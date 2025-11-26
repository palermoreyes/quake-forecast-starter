import os
import json
import datetime as dt
import math
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import tensorflow as tf
import reverse_geocoder as rg 
from dataclasses import dataclass

# ==========================
#  CONFIGURACIÓN
# ==========================

@dataclass
class Config:
    db_host: str = os.getenv("DB_HOST", "db")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "quake")
    db_user: str = os.getenv("DB_USER", "quake")
    db_password: str = os.getenv("DB_PASSWORD", "changeme")
    
    models_dir: str = os.getenv("MODELS_DIR", "/app/artifacts/models")
    
    # CAMBIO EXPERTO: Subimos a 25 para tener margen de error y capturar más sismos.
    # 10 era muy estricto (1:1 con la realidad). 25 da holgura (2.5:1).
    topk_k: int = 25
    
    lookback_days: int = 90 

def get_conn(cfg: Config):
    return psycopg2.connect(
        host=cfg.db_host, port=cfg.db_port, dbname=cfg.db_name, 
        user=cfg.db_user, password=cfg.db_password
    )

# ==========================
#  UTILIDADES GEOGRÁFICAS CIENTÍFICAS
# ==========================

def haversine_km(lat1, lon1, lat2, lon2):
    """Calcula distancia en km entre dos puntos (Fórmula Haversine)."""
    R = 6371.0 # Radio Tierra
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_cardinal_direction(lat1, lon1, lat2, lon2):
    """Calcula la dirección (N, NE, E...) del sismo respecto a la ciudad."""
    dLon = math.radians(lon1 - lon2)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dLon) * math.cos(lat1)
    x = math.cos(lat2) * math.sin(lat1) - math.sin(lat2) * math.cos(lat1) * math.cos(dLon)
    bearing = math.degrees(math.atan2(y, x))
    bearing = (bearing + 360) % 360
    dirs = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    idx = round(bearing / 45) % 8
    return dirs[idx]

def format_location(event_lat, event_lon, geo_info):
    """Genera texto rico: 'A 15 km al SO de Mala, Lima'"""
    city_name = geo_info.get('name', 'Zona')
    region = geo_info.get('admin1', '')
    
    city_lat = float(geo_info['lat'])
    city_lon = float(geo_info['lon'])
    
    dist = haversine_km(event_lat, event_lon, city_lat, city_lon)
    
    if dist < 5:
        return f"En {city_name}, {region}"
    
    direction = get_cardinal_direction(event_lat, event_lon, city_lat, city_lon)
    return f"A {int(dist)} km al {direction} de {city_name}, {region}"

# ==========================
#  LÓGICA DEL MODELO
# ==========================

def get_active_model(cfg: Config):
    with get_conn(cfg) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT model_id, params_json, horizons, mag_min 
                FROM public.model_registry WHERE is_active = true 
                ORDER BY created_at DESC LIMIT 1
            """)
            row = cur.fetchone()
    if not row: raise RuntimeError("No hay modelo activo.")
    params = row["params_json"]
    path_rel = params.get("model_path")
    if not path_rel:
         files = sorted([f for f in os.listdir(cfg.models_dir) if f.endswith(".keras")])
         if files: path_rel = files[-1]
         else: raise RuntimeError("No se encontró archivo .keras.")
    path = os.path.join(cfg.models_dir, path_rel)
    return row["model_id"], path, row["horizons"], float(row["mag_min"])

def build_national_tensor(cfg: Config, mag_min: float, window_days: int):
    print("[PRED] Cargando historial reciente...")
    query = """
    SELECT c.cell_id, date_trunc('day', e.event_time_utc)::date as date, COUNT(e.id) as count
    FROM public.prediction_cells c
    LEFT JOIN public.events_clean e 
        ON ST_Intersects(c.geom, e.geom) AND e.magnitude >= %s AND e.event_time_utc >= (current_date - interval '%s days')
    GROUP BY c.cell_id, date
    """
    with get_conn(cfg) as conn:
        df = pd.read_sql_query(query, conn, params=(mag_min, cfg.lookback_days))
    
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="cell_id", values="count").fillna(0)
    
    # --- FECHA AUTOMÁTICA (PRODUCCIÓN) ---
    end_date = pd.Timestamp.now().date() - pd.Timedelta(days=1) 
    
    # --- FECHA MANUAL (BACKTESTING - Descomentar para pruebas pasadas) ---
    #end_date = pd.Timestamp("2025-11-01").date() 
    
    full_idx = pd.date_range(end=end_date, periods=cfg.lookback_days, freq="D")
    pivot = pivot.reindex(full_idx, fill_value=0)
    
    recent_matrix = pivot.iloc[-window_days:].values
    cell_ids = pivot.columns.to_numpy()
    last_input_date = pivot.index[-1] 
    
    if len(recent_matrix) < window_days: raise RuntimeError("Datos insuficientes.")

    feat_counts = recent_matrix.T 
    means = feat_counts.mean(axis=1, keepdims=True)
    stds = feat_counts.std(axis=1, keepdims=True) + 1e-5
    feat_norm = (feat_counts - means) / stds
    feat_bin = (feat_counts > 0).astype(np.float32)
    X = np.stack([feat_norm, feat_bin], axis=-1)
    
    return X, cell_ids, last_input_date

# ==========================
#  MAIN
# ==========================

def main():
    cfg = Config()
    try:
        model_id, model_path, horizons, mag_min = get_active_model(cfg)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    window_days = 30 
    print(f"[PRED] Iniciando inferencia. Modelo: {model_id} | Top-K: {cfg.topk_k}")
    
    X, cell_ids, last_input_date = build_national_tensor(cfg, mag_min, window_days)
    print(f"[PRED] Ejecutando modelo...")
    
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X, batch_size=2048, verbose=1).flatten()
    
    last_input_ts = pd.to_datetime(last_input_date).to_pydatetime()
    pred_start = last_input_ts + dt.timedelta(days=1) 
    pred_end   = pred_start + dt.timedelta(days=horizons[0] - 1) + dt.timedelta(hours=23, minutes=59, seconds=59)
    
    print("[PRED] Procesando resultados...")
    df_res = pd.DataFrame({"cell_id": cell_ids, "prob": probs})
    df_res["rank_pct"] = df_res["prob"].rank(pct=True)
    
    topk = df_res.nlargest(cfg.topk_k, "prob")
    
    with get_conn(cfg) as conn:
        ids = tuple(topk["cell_id"].tolist())
        q_geo = f"SELECT cell_id, ST_Y(centroid) as lat, ST_X(centroid) as lon FROM prediction_cells WHERE cell_id IN {ids}"
        df_geo = pd.read_sql_query(q_geo, conn)
        
        coords = list(zip(df_geo["lat"], df_geo["lon"]))
        results = rg.search(coords)
        
        place_map = {}
        for i, row in df_geo.iterrows():
            ref_text = format_location(row["lat"], row["lon"], results[i])
            place_map[row["cell_id"]] = ref_text

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO prediction_run (generated_at, model_id, input_max_time, horizons, mag_min, cell_km, topk_k)
                VALUES (now(), %s, %s, %s, %s, 10, %s) RETURNING run_id
            """, (model_id, last_input_ts, list(horizons), mag_min, cfg.topk_k))
            run_id = cur.fetchone()[0]
            
            data_probs = [
                (run_id, int(r.cell_id), horizons[0], float(r.prob), float(r.prob), float(r.rank_pct))
                for r in df_res.itertuples()
            ]
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO prediction_cell_prob (run_id, cell_id, horizon_days, prob, density, rank_pct) VALUES %s",
                data_probs
            )
            
            for i, row in enumerate(topk.itertuples(), 1):
                place_txt = place_map.get(row.cell_id, "Zona Remota")
                cur.execute("""
                    INSERT INTO prediction_topk 
                    (run_id, horizon_days, rank, t_pred_start, t_pred_end, lat, lon, mag_pred, prob, place)
                    SELECT %s, %s, %s, %s, %s, ST_Y(centroid), ST_X(centroid), %s, %s, %s
                    FROM prediction_cells WHERE cell_id = %s
                """, (
                    run_id, horizons[0], i, pred_start, pred_end, 
                    mag_min + 0.5, row.prob, place_txt, row.cell_id
                ))
                
            conn.commit()
            print(f"[PRED] ¡Éxito! Run ID: {run_id}. {cfg.topk_k} Alertas generadas.")

if __name__ == "__main__":
    main()