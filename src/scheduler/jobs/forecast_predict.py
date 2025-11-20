import os
import json
import datetime as dt

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import tensorflow as tf

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    db_host: str = os.getenv("DB_HOST", "db")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "quake")
    db_user: str = os.getenv("DB_USER", "quake")
    db_password: str = os.getenv("DB_PASSWORD", "changeme")

    mag_min: float = float(os.getenv("PRED_MAG_MIN", "4.0"))
    horizons: Tuple[int, ...] = tuple(int(x) for x in os.getenv("PRED_HORIZONS", "7,14").split(","))

    window_days: int = int(os.getenv("TRAIN_WINDOW_DAYS", "30"))
    models_dir: str = os.getenv("MODELS_DIR", "/app/artifacts/models")
    topk_k: int = int(os.getenv("TOPK_K", "10"))

    # límite de celdas para que no explote en inferencia
    max_cells_infer: int = int(os.getenv("INFER_MAX_CELLS", "2000"))


def get_conn(cfg: Config):
    return psycopg2.connect(
        host=cfg.db_host,
        port=cfg.db_port,
        dbname=cfg.db_name,
        user=cfg.db_user,
        password=cfg.db_password,
    )


# ============================
# 1. OBTENER MODELO ACTIVO
# ============================

def get_active_model_meta(cfg: Config):
    with get_conn(cfg) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT model_id, params_json, horizons, mag_min
                FROM public.model_registry
                WHERE is_active = true
                ORDER BY created_at DESC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
    if row is None:
        raise RuntimeError("No hay modelo activo en model_registry")

    model_id = row["model_id"]
    params = row["params_json"] or {}
    model_path_rel = params.get("model_path")
    if not model_path_rel:
        raise RuntimeError("El modelo activo no tiene 'model_path' en params_json")

    model_path = os.path.join(cfg.models_dir, model_path_rel)
    horizons = row["horizons"]
    mag_min = float(row["mag_min"])

    return model_id, model_path, horizons, mag_min


# ============================
# 2. SERIE DIARIA PARA INFERENCIA
# ============================

def load_daily_cell_series_for_inference(cfg: Config) -> pd.DataFrame:
    """
    Construye la serie diaria SOLO para los últimos N días (INFER_LOOKBACK_DAYS),
    en lugar de usar toda la historia desde 1960. Esto reduce muchísimo el tamaño
    del DataFrame y acelera la inferencia.
    """

    lookback_days = int(os.getenv("INFER_LOOKBACK_DAYS", "365"))

    query = """
    WITH events_with_cell AS (
        SELECT
            date_trunc('day', e.event_time_utc)::date AS event_date,
            c.cell_id,
            COUNT(*) AS event_count
        FROM public.events_clean e
        JOIN public.prediction_cells c
          ON ST_Contains(
                c.geom,
                ST_SetSRID(ST_MakePoint(e.lon, e.lat), 4326)
             )
        WHERE e.magnitude >= %s
        GROUP BY event_date, c.cell_id
    )
    SELECT
        event_date AS date,
        cell_id,
        event_count
    FROM events_with_cell
    ORDER BY date, cell_id;
    """

    with get_conn(cfg) as conn:
        df_events = pd.read_sql_query(query, conn, params=(cfg.mag_min,))

    if df_events.empty:
        raise RuntimeError("No se encontraron eventos para el umbral de magnitud dado (inference).")

    df_events["date"] = pd.to_datetime(df_events["date"])

    max_date = df_events["date"].max()
    min_date = max_date - pd.Timedelta(days=lookback_days - 1)

    # Nos quedamos solo con el rango reciente
    df_events = df_events[(df_events["date"] >= min_date) & (df_events["date"] <= max_date)]

    print(f"[PRED] Rango fechas (reciente): {min_date} -> {max_date}, filas={len(df_events)}")

    all_cells = df_events["cell_id"].unique()
    print(f"[PRED] Celdas con al menos un evento en lookback: {len(all_cells)}")

    frames = []
    full_idx = pd.date_range(start=min_date, end=max_date, freq="D")

    for cell_id in all_cells:
        g = df_events[df_events["cell_id"] == cell_id].copy()
        g = g.set_index("date").sort_index()

        g = g.reindex(full_idx)
        g.index.name = "date"

        g["cell_id"] = cell_id
        g["event_count"] = g["event_count"].fillna(0).astype(int)
        g["y_bin"] = (g["event_count"] > 0).astype(int)

        frames.append(g.reset_index())

    df = pd.concat(frames, ignore_index=True)
    print(f"[PRED] Serie diaria completada (reciente): filas={len(df)}")

    return df[["date", "cell_id", "event_count", "y_bin"]]


# ============================
# 3. ARMAR INPUT PARA INFERENCIA
# ============================

def build_inference_tensor(df: pd.DataFrame, cfg: Config):
    df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)

    # Normalización por celda (igual que en training)
    groups = []
    for cell_id, g in df.groupby("cell_id", group_keys=False):
        g = g.sort_values("date")
        g["event_count_norm"] = (g["event_count"] - g["event_count"].mean()) / (g["event_count"].std() + 1e-6)
        groups.append(g)
    df_norm = pd.concat(groups, ignore_index=True)

    last_date = df_norm["date"].max()

    X_list = []
    cell_ids = []

    for cell_id, g in df_norm.groupby("cell_id"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < cfg.window_days:
            continue

        g_last = g.iloc[-cfg.window_days:]
        values = g_last[["event_count_norm", "y_bin"]].values

        X_list.append(values)
        cell_ids.append(cell_id)

    X = np.array(X_list, dtype=np.float32)
    print(f"[PRED] Tensor completo: X.shape={X.shape}, celdas={len(cell_ids)}, last_date={last_date}")

    # limitar número de celdas para inferencia (por memoria/tiempo)
    if len(cell_ids) > cfg.max_cells_infer:
        idx = np.random.choice(len(cell_ids), cfg.max_cells_infer, replace=False)
        X = X[idx]
        cell_ids = [cell_ids[i] for i in idx]
        print(f"[PRED] Submuestreo de celdas para inferencia: {cfg.max_cells_infer} celdas")

    return X, cell_ids, last_date


# ============================
# 4. INSERTAR EN BD
# ============================

def insert_prediction_run(cfg: Config, model_id: int, horizons, input_max_time: dt.datetime) -> int:
    with get_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.prediction_run (
                    generated_at,
                    model_id,
                    input_max_time,
                    horizons,
                    mag_min,
                    cell_km,
                    topk_k,
                    code_version,
                    notes
                )
                SELECT
                    now(),
                    %s,
                    %s,
                    %s,
                    %s,
                    (SELECT cell_km FROM public.prediction_cells LIMIT 1),
                    %s,
                    'forecast_predict_v1',
                    'Run automático de inferencia LSTM'
                RETURNING run_id;
                """,
                (model_id, input_max_time, list(horizons), cfg.mag_min, cfg.topk_k)
            )
            run_id = cur.fetchone()[0]
        conn.commit()
    return run_id


def insert_cell_probs_and_topk(cfg: Config, run_id: int, horizon_days: int, cell_ids, probs, last_date: dt.datetime):
    df = pd.DataFrame({"cell_id": cell_ids, "prob": probs.reshape(-1)})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)
    df["rank_pct"] = 100.0 * (1.0 - (df["rank"] - 1) / max(1, len(df) - 1))
    df["density"] = df["prob"]

    inserted_cells = 0
    inserted_topk = 0

    with get_conn(cfg) as conn:
        with conn.cursor() as cur:

            # prediction_cell_prob
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO public.prediction_cell_prob (
                        run_id, cell_id, horizon_days, prob, density, rank_pct
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        int(row["cell_id"]),
                        horizon_days,
                        float(row["prob"]),
                        float(row["density"]),
                        float(row["rank_pct"]),
                    )
                )
                inserted_cells += 1

            # Top-K
            topk = df.head(cfg.topk_k)
            for _, row in topk.iterrows():
                cur.execute(
                    """
                    INSERT INTO public.prediction_topk (
                        run_id,
                        horizon_days,
                        rank,
                        t_pred_start,
                        t_pred_end,
                        lat,
                        lon,
                        mag_pred,
                        depth_pred,
                        prob,
                        time_conf_h,
                        space_conf_km
                    )
                    SELECT
                        %s, %s, %s,
                        %s,
                        %s,
                        ST_Y(centroid) AS lat,
                        ST_X(centroid) AS lon,
                        %s,
                        NULL,
                        %s,
                        48,
                        75
                    FROM public.prediction_cells
                    WHERE cell_id = %s;
                    """,
                    (
                        run_id,
                        horizon_days,
                        int(row["rank"]),
                        last_date + dt.timedelta(seconds=1),
                        last_date + dt.timedelta(days=horizon_days),
                        cfg.mag_min + 0.5,
                        float(row["prob"]),
                        int(row["cell_id"]),
                    )
                )
                inserted_topk += 1

        conn.commit()

    print(f"[PRED] Insertadas {inserted_cells} filas en prediction_cell_prob")
    print(f"[PRED] Insertadas {inserted_topk} filas en prediction_topk (Top-{cfg.topk_k})")


# ============================
# 5. MAIN
# ============================

def main():
    cfg = Config()
    print("[PRED] Config:", cfg)

    model_id, model_path, horizons, mag_min = get_active_model_meta(cfg)
    print(f"[PRED] Modelo activo: model_id={model_id}, path={model_path}, horizons={horizons}, mag_min={mag_min}")

    print("[PRED] Cargando serie diaria...")
    df = load_daily_cell_series_for_inference(cfg)
    df["date"] = pd.to_datetime(df["date"])

    X, cell_ids, last_date = build_inference_tensor(df, cfg)
    print(f"[PRED] Tensor final para inferencia: X.shape={X.shape}, celdas={len(cell_ids)}, last_date={last_date}")

    print("[PRED] Cargando modelo LSTM...")
    model = tf.keras.models.load_model(model_path)

    print("[PRED] Ejecutando predicciones...")
    probs = model.predict(X, batch_size=1024, verbose=1)
    print("[PRED] Predicciones completadas.")

    horizon_days = int(horizons[0])  # por ahora solo 7 días 

    input_max_time = pd.to_datetime(last_date).to_pydatetime()
    run_id = insert_prediction_run(cfg, model_id, horizons, input_max_time)
    print(f"[PRED] run_id creado: {run_id}")

    insert_cell_probs_and_topk(cfg, run_id, horizon_days, cell_ids, probs, last_date)
    print("[PRED] Predicciones guardadas en prediction_cell_prob y prediction_topk")


if __name__ == "__main__":
    main()
