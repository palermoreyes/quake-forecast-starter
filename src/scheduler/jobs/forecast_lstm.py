import os
import json
import datetime as dt
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import tensorflow as tf


# ==========================
#  CONFIGURACIÓN DEL MODELO
# ==========================

@dataclass
class Config:
    db_host: str = os.getenv("DB_HOST", "db")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "quake")
    db_user: str = os.getenv("DB_USER", "quake")
    db_password: str = os.getenv("DB_PASSWORD", "changeme")

    mag_min: float = float(os.getenv("PRED_MAG_MIN", "4.0"))
    horizons: Tuple[int, ...] = tuple(int(x) for x in os.getenv("PRED_HORIZONS", "7,14").split(","))
    train_end: dt.date = dt.date.fromisoformat(os.getenv("SPLIT_TRAIN_END", "2018-12-31"))
    val_end: dt.date = dt.date.fromisoformat(os.getenv("SPLIT_VAL_END", "2021-12-31"))
    test_end: dt.date = dt.date.fromisoformat(os.getenv("SPLIT_TEST_END", "2023-12-31"))

    window_days: int = int(os.getenv("TRAIN_WINDOW_DAYS", "30"))  # longitud de la secuencia
    batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "64"))
    epochs: int = int(os.getenv("TRAIN_EPOCHS", "20"))

    models_dir: str = os.getenv("MODELS_DIR", "/app/artifacts/models")


def get_conn(cfg: Config):
    return psycopg2.connect(
        host=cfg.db_host,
        port=cfg.db_port,
        dbname=cfg.db_name,
        user=cfg.db_user,
        password=cfg.db_password,
    )


# =========================================
#  1. EXTRAER SERIE DIARIA POR CELDA (LABELS)
# =========================================

def load_daily_cell_series(cfg: Config) -> pd.DataFrame:
    """
    Construye una tabla día x celda SOLO en Python, evitando el cross join masivo en SQL.
    En SQL solo agregamos los eventos por fecha y celda que realmente tienen sismos.
    Luego, por cada celda, completamos los días faltantes con 0 en pandas.
    """

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
        raise RuntimeError("No se encontraron eventos para el umbral de magnitud dado.")

    # Rango global de fechas
    min_date = df_events["date"].min()
    max_date = df_events["date"].max()

    # Para cada celda, completar fechas faltantes con 0
    all_cells = df_events["cell_id"].unique()
    frames = []

    # Opcional: limitar a N celdas para pruebas iniciales
    max_cells = int(os.getenv("TRAIN_MAX_CELLS", "500"))  # por ejemplo 500
    if len(all_cells) > max_cells:
        all_cells = all_cells[:max_cells]

    for cell_id in all_cells:
        g = df_events[df_events["cell_id"] == cell_id].copy()
        g = g.set_index("date").sort_index()

        # Crear índice diario y rellenar
        full_idx = pd.date_range(start=min_date, end=max_date, freq="D")
        g = g.reindex(full_idx)
        g.index.name = "date"

        g["cell_id"] = cell_id
        g["event_count"] = g["event_count"].fillna(0).astype(int)
        g["y_bin"] = (g["event_count"] > 0).astype(int)

        frames.append(g.reset_index())

    df = pd.concat(frames, ignore_index=True)

    return df[["date", "cell_id", "event_count", "y_bin"]]


# =========================================
#  2. CREAR SECUENCIAS (VENTANAS DESLIZANTES)
# =========================================

def make_sequences(df: pd.DataFrame, cfg: Config):
    """
    df: columnas [date, cell_id, event_count, y_bin]
    Salida: X, y para un horizonte (por ahora usaremos solo 7 días para simplificar).
    """

    horizon = cfg.horizons[0]  # por ahora solo el primero (ej. 7 días)

    # Ordenamos por celda y fecha
    df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)

    # Normalizamos por celda
    groups = []
    for cell_id, g in df.groupby("cell_id", group_keys=False):
        g = g.sort_values("date")
        g["event_count_norm"] = (g["event_count"] - g["event_count"].mean()) / (g["event_count"].std() + 1e-6)
        groups.append(g)
    df_norm = pd.concat(groups, ignore_index=True)

    # Crear secuencias
    X_list = []
    y_list = []

    for cell_id, g in df_norm.groupby("cell_id"):
        g = g.sort_values("date").reset_index(drop=True)

        # target: si hay algún evento en los próximos 'horizon' días
        y_target = []
        y_series = g["y_bin"].values
        for i in range(len(g)):
            end = min(len(g), i + horizon + 1)
            y_target.append(1 if y_series[i+1:end].sum() > 0 else 0)
        g["y_target"] = y_target

        values = g[["event_count_norm", "y_bin"]].values
        targets = g["y_target"].values

        for i in range(len(g) - cfg.window_days - horizon):
            X_list.append(values[i : i + cfg.window_days])
            y_list.append(targets[i + cfg.window_days])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


# ==========================
#  3. MODELO LSTM EN KERAS
# ==========================

def build_lstm_model(cfg: Config, n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cfg.window_days, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),  # probabilidad de evento en horizonte
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==================================
#  4. ENTRENAR, EVALUAR Y REGISTRAR
# ==================================

def train_and_register(cfg: Config):
    os.makedirs(cfg.models_dir, exist_ok=True)

    print("[TRAIN] Cargando serie diaria por celda...")
    df = load_daily_cell_series(cfg)

    # 🔹 Normalizar tipo de fecha para comparaciones
    df["date"] = pd.to_datetime(df["date"])

    # Split por fechas
    df_train = df[df["date"] <= pd.Timestamp(cfg.train_end)]
    df_val   = df[(df["date"] > pd.Timestamp(cfg.train_end)) & (df["date"] <= pd.Timestamp(cfg.val_end))]
    df_test  = df[(df["date"] > pd.Timestamp(cfg.val_end))   & (df["date"] <= pd.Timestamp(cfg.test_end))]

    print(f"[TRAIN] Train: {df_train['date'].min()} -> {df_train['date'].max()}")
    print(f"[TRAIN] Val:   {df_val['date'].min()} -> {df_val['date'].max()}")
    print(f"[TRAIN] Test:  {df_test['date'].min()} -> {df_test['date'].max()}")

    X_train, y_train = make_sequences(df_train, cfg)
    X_val, y_val = make_sequences(df_val, cfg)
    X_test, y_test = make_sequences(df_test, cfg)

    print(f"[TRAIN] X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    n_features = X_train.shape[-1]

    model = build_lstm_model(cfg, n_features)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
        callbacks=callbacks
    )

    print("[TRAIN] Evaluando en conjunto de prueba...")
    test_metrics = model.evaluate(X_test, y_test, verbose=0)
    test_loss = float(test_metrics[0])
    test_acc = float(test_metrics[1])

    print(f"[TRAIN] Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    # Guardar modelo en disco
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = os.path.join(cfg.models_dir, f"model_lstm_{ts}.keras")
    model.save(model_path)
    print(f"[TRAIN] Modelo guardado en {model_path}")

    # Registrar en model_registry
    metrics_json = {
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    params_json = {
        "window_days": cfg.window_days,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
    }

    features_json = {
        "inputs": ["event_count_norm", "y_bin"],
        "target": "evento_en_horizonte",
        "horizon_days": cfg.horizons[0],
    }

    with get_conn(cfg) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO public.model_registry (
                    is_active,
                    in_staging,
                    framework,
                    tag,
                    train_start,
                    train_end,
                    horizons,
                    mag_min,
                    params_json,
                    features_json,
                    metrics_json,
                    git_commit,
                    image_tag,
                    data_cutoff,
                    notes
                ) VALUES (
                    FALSE, TRUE, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s::jsonb,
                    NULL, NULL, %s, %s
                )
                RETURNING model_id;
                """,
                (
                    "keras",
                    f"LSTM_window{cfg.window_days}_M{cfg.mag_min}",
                    df_train["date"].min(),
                    df_train["date"].max(),
                    list(cfg.horizons),
                    cfg.mag_min,
                    json.dumps(params_json),
                    json.dumps(features_json),
                    json.dumps(metrics_json),
                    cfg.test_end,
                    "Modelo base LSTM binario por celda/horizonte",
                )
            )
            row = cur.fetchone()
            model_id = row["model_id"]

    print(f"[TRAIN] Modelo registrado en model_registry con model_id={model_id}")


def main():
    cfg = Config()
    print("[TRAIN] Config:", cfg)
    train_and_register(cfg)


if __name__ == "__main__":
    main()
