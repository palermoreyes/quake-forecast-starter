# forecast_predict.py

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

    # Top-K de celdas con mayor probabilidad para dashboard
    topk_k: int = int(os.getenv("PRED_TOPK_K", "25"))

    # Umbral mínimo para considerar una celda candidata a alerta
    min_prob_alert: float = float(os.getenv("PRED_MIN_PROB_ALERT", "0.10"))

    # Días hacia atrás para construir el tensor nacional
    lookback_days: int = int(os.getenv("PRED_LOOKBACK_DAYS", "90"))


def get_conn(cfg: Config):
    return psycopg2.connect(
        host=cfg.db_host,
        port=cfg.db_port,
        dbname=cfg.db_name,
        user=cfg.db_user,
        password=cfg.db_password,
    )


# ==========================
#  UTILIDADES GEOGRÁFICAS
# ==========================

def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en km entre dos puntos (Fórmula de Haversine)."""
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (
        math.sin(dLat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dLon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_cardinal_direction(lat1, lon1, lat2, lon2):
    """Dirección cardinal (N, NE, E...) del punto (lat1,lon1) respecto a (lat2,lon2)."""
    dLon = math.radians(lon1 - lon2)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dLon) * math.cos(lat1)
    x = (
        math.cos(lat2) * math.sin(lat1)
        - math.sin(lat2) * math.cos(lat1) * math.cos(dLon)
    )
    bearing = math.degrees(math.atan2(y, x))
    bearing = (bearing + 360) % 360
    dirs = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    idx = round(bearing / 45) % 8
    return dirs[idx]


def format_location(event_lat, event_lon, geo_info):
    """
    Devuelve un texto tipo: 'A 15 km al SO de Mala, Lima'
    o 'En Mala, Lima' si la distancia < 5 km.
    """
    city_name = geo_info.get("name", "Zona")
    region = geo_info.get("admin1", "")

    city_lat = float(geo_info["lat"])
    city_lon = float(geo_info["lon"])

    dist = haversine_km(event_lat, event_lon, city_lat, city_lon)

    if dist < 5:
        return f"En {city_name}, {region}".strip(", ")

    direction = get_cardinal_direction(event_lat, event_lon, city_lat, city_lon)
    return f"A {int(dist)} km al {direction} de {city_name}, {region}".strip(", ")


# ==========================
#  LÓGICA DE MODELO / CARGA
# ==========================

def get_active_model(cfg: Config):
    """
    Devuelve:
      - model_id
      - ruta del archivo .keras
      - horizons
      - mag_min
      - params_json (dict) con norm_means, norm_stds, cell_ids, window_days, etc.
    """
    with get_conn(cfg) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT model_id, params_json, horizons, mag_min
                FROM public.model_registry
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()

    if not row:
        raise RuntimeError("No hay modelo activo en model_registry.")

    params = row["params_json"]
    if isinstance(params, str):
        params = json.loads(params)

    path_rel = params.get("model_path")
    if not path_rel:
        # Fallback: último .keras en la carpeta
        files = sorted(
            [f for f in os.listdir(cfg.models_dir) if f.endswith(".keras")]
        )
        if files:
            path_rel = files[-1]
        else:
            raise RuntimeError("No se encontró archivo .keras para el modelo activo.")

    path = os.path.join(cfg.models_dir, path_rel)
    return row["model_id"], path, row["horizons"], float(row["mag_min"]), params


def build_national_tensor(
    cfg: Config,
    mag_min: float,
    window_days: int,
    cell_ids_train: list,
    norm_means: np.ndarray,
    norm_stds: np.ndarray,
):
    """
    Construye el tensor nacional [num_cells, window_days, 2] usando:
      - el mismo orden de cell_ids que en entrenamiento
      - las mismas medias/desviaciones de normalización
    """
    print("[PRED] Cargando historial reciente...")

    query = """
    SELECT 
        c.cell_id,
        date_trunc('day', e.event_time_utc)::date AS date,
        COUNT(e.id) AS count
    FROM public.prediction_cells c
    LEFT JOIN public.events_clean e
        ON ST_Intersects(c.geom, e.geom)
       AND e.magnitude >= %s
       AND e.event_time_utc >= (current_date - interval '%s days')
    GROUP BY c.cell_id, date
    """

    with get_conn(cfg) as conn:
        df = pd.read_sql_query(
            query, conn, params=(mag_min, cfg.lookback_days)
        )

    df["date"] = pd.to_datetime(df["date"])
    pivot = (
        df.pivot(index="date", columns="cell_id", values="count")
        .fillna(0)
        .astype(np.float32)
    )

    # Producción: último día disponible (ayer)
    end_date = pd.Timestamp.now().date() - pd.Timedelta(days=0)

    # --- FECHA MANUAL (BACKTESTING - Descomentar para pruebas pasadas) ---
    #end_date = pd.Timestamp("2025-11-10").date() 

    full_idx = pd.date_range(
        end=end_date, periods=cfg.lookback_days, freq="D"
    )
    pivot = pivot.reindex(full_idx, fill_value=0).astype(np.float32)

    # Reordenar columnas para que coincidan con entrenamiento
    pivot = pivot.reindex(columns=cell_ids_train, fill_value=0).astype(np.float32)

    if pivot.shape[0] < window_days:
        raise RuntimeError(
            f"Datos insuficientes: se requieren al menos {window_days} días."
        )

    recent_matrix = pivot.iloc[-window_days:].values  # [window_days, num_cells]
    last_input_date = pivot.index[-1]

    # Normalización consistente (dims: [num_cells, window_days])
    feat_counts = recent_matrix.T  # [C, W]
    if norm_means.shape[0] != feat_counts.shape[0]:
        raise RuntimeError("Dimensiones de norm_means no coinciden con num_cells.")
    feat_norm = (feat_counts - norm_means[:, None]) / norm_stds[:, None]
    feat_bin = (feat_counts > 0).astype(np.float32)

    X = np.stack([feat_norm, feat_bin], axis=-1).astype(np.float32)

    return X, np.array(cell_ids_train, dtype=int), last_input_date


# ==========================
#  MAIN DE PREDICCIÓN
# ==========================

def main():
    cfg = Config()

    try:
        (
            model_id,
            model_path,
            horizons,
            mag_min,
            params,
        ) = get_active_model(cfg)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Parámetros guardados en entrenamiento
    cell_ids_train = params.get("cell_ids")
    norm_means = params.get("norm_means")
    norm_stds = params.get("norm_stds")
    window_days = params.get("window_days", 30)

    if not (cell_ids_train and norm_means and norm_stds):
        raise RuntimeError(
            "El modelo activo no contiene parámetros de normalización V3 "
            "(cell_ids / norm_means / norm_stds)."
        )

    cell_ids_train = list(cell_ids_train)
    norm_means = np.array(norm_means, dtype=np.float32)
    norm_stds = np.array(norm_stds, dtype=np.float32)

    print(
        f"[PRED] Iniciando inferencia. Modelo: {model_id} | "
        f"Top-K={cfg.topk_k} | min_prob_alert={cfg.min_prob_alert:.2f}"
    )

    X, cell_ids, last_input_date = build_national_tensor(
        cfg,
        mag_min,
        window_days,
        cell_ids_train=cell_ids_train,
        norm_means=norm_means,
        norm_stds=norm_stds,
    )

    print("[PRED] Ejecutando modelo...")
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X, batch_size=2048, verbose=1).flatten()

    # Horizonte (asumimos un único horizonte para ahora)
    last_input_ts = pd.to_datetime(last_input_date).to_pydatetime()
    horizon_days = int(horizons[0])

    pred_start = last_input_ts + dt.timedelta(days=1)
    pred_end = pred_start + dt.timedelta(days=horizon_days - 1) + dt.timedelta(
        hours=23, minutes=59, seconds=59
    )

    print("[PRED] Procesando resultados probabilísticos...")
    df_res = pd.DataFrame({"cell_id": cell_ids, "prob": probs})
    df_res["rank_pct"] = df_res["prob"].rank(pct=True)

    # Filtro por probabilidad mínima
    df_filtered = df_res[df_res["prob"] >= cfg.min_prob_alert].copy()
    if df_filtered.empty:
        print(
            "[PRED] Ninguna celda supera el umbral de probabilidad. "
            "Se usarán las top-K absolutas como fallback."
        )
        df_filtered = df_res.copy()

    topk = df_filtered.nlargest(cfg.topk_k, "prob")

    # ========================
    #  PERSISTENCIA EN BD
    # ========================
    with get_conn(cfg) as conn:
        # 1. Geometría de las celdas top-K
        ids_tuple = tuple(int(c) for c in topk["cell_id"].tolist())
        q_geo = (
            "SELECT cell_id, ST_Y(centroid) AS lat, ST_X(centroid) AS lon "
            "FROM prediction_cells WHERE cell_id IN %s"
        )
        df_geo = pd.read_sql_query(q_geo, conn, params=(ids_tuple,))

        # Reverse geocoding en el mismo orden
        coords = list(zip(df_geo["lat"], df_geo["lon"]))
        results = rg.search(coords)

        place_map = {}
        for idx, row in enumerate(df_geo.itertuples(index=False)):
            ref_text = format_location(row.lat, row.lon, results[idx])
            place_map[int(row.cell_id)] = ref_text

        with conn.cursor() as cur:
            # 2. Registrar corrida de predicción
            cur.execute(
                """
                INSERT INTO prediction_run
                (generated_at, model_id, input_max_time, horizons, mag_min, cell_km, topk_k)
                VALUES (now(), %s, %s, %s, %s, 10, %s)
                RETURNING run_id
                """,
                (
                    model_id,
                    last_input_ts,
                    list(horizons),
                    mag_min,
                    cfg.topk_k,
                ),
            )
            run_id = cur.fetchone()[0]

            # 3. Guardar TODAS las probabilidades por celda (grid completo)
            data_probs = [
                (
                    run_id,
                    int(r.cell_id),
                    horizon_days,
                    float(r.prob),
                    float(r.prob),   # density ~ prob por ahora
                    float(r.rank_pct),
                )
                for r in df_res.itertuples()
            ]

            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO prediction_cell_prob
                    (run_id, cell_id, horizon_days, prob, density, rank_pct)
                VALUES %s
                """,
                data_probs,
            )

            # 4. Guardar SOLO top-K (incluyendo cell_id y place)
            for rank_idx, row in enumerate(topk.itertuples(), 1):
                cell_id = int(row.cell_id)
                place_txt = place_map.get(cell_id, "Zona remota")
                cur.execute(
                    """
                    INSERT INTO prediction_topk
                    (run_id, horizon_days, rank, 
                     t_pred_start, t_pred_end,
                     lat, lon, mag_pred, prob, place, cell_id)
                    SELECT %s, %s, %s, %s, %s,
                           ST_Y(centroid), ST_X(centroid),
                           %s, %s, %s, %s
                    FROM prediction_cells
                    WHERE cell_id = %s
                    """,
                    (
                        run_id,
                        horizon_days,
                        rank_idx,
                        pred_start,
                        pred_end,
                        mag_min + 0.5,      # magnitud estimada (ej: mag_min + 0.5)
                        float(row.prob),
                        place_txt,
                        cell_id,             # valor para columna cell_id
                        cell_id,             # valor para WHERE cell_id = ...
                    ),
                )

            conn.commit()
            print(
                f"[PRED] ¡Éxito! Run ID: {run_id}. "
                f"{len(topk)} alertas generadas (K={cfg.topk_k})."
            )


if __name__ == "__main__":
    main()
