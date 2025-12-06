# forecast_lstm.py (V3.5 - Final: Bi-LSTM 30d + log1p + GeoShuffle)

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
from numpy.lib.stride_tricks import sliding_window_view

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

    mag_min: float = float(os.getenv("PRED_MAG_MIN", "4.0"))
    horizons: Tuple[int, ...] = tuple(
        int(x) for x in os.getenv("PRED_HORIZONS", "7,14").split(",")
    )

    train_end: dt.date = dt.date.fromisoformat(
        os.getenv("SPLIT_TRAIN_END", "2018-12-31")
    )
    val_end: dt.date = dt.date.fromisoformat(
        os.getenv("SPLIT_VAL_END", "2021-12-31")
    )
    test_end: dt.date = dt.date.fromisoformat(
        os.getenv("SPLIT_TEST_END", "2023-12-31")
    )

    # Ventana donde sabemos que hay señal: 30 días
    window_days: int = int(os.getenv("TRAIN_WINDOW_DAYS", "30"))

    # Tamaño de batch equilibrado para tu VM
    batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "512"))
    # Más épocas, pero con EarlyStopping
    epochs: int = int(os.getenv("TRAIN_EPOCHS", "40"))

    # Sampling 2% de negativos: sweet spot que ya probaste (V3.3.1)
    keep_neg_rate: float = float(os.getenv("TRAIN_KEEP_NEG_RATE", "0.02"))

    # Límites seguros (con 64 GB estás sobrado)
    max_train_samples: int = int(os.getenv("MAX_TRAIN_SAMPLES", "4000000"))
    max_eval_samples: int = int(os.getenv("MAX_EVAL_SAMPLES", "1000000"))

    models_dir: str = os.getenv("MODELS_DIR", "/app/artifacts/models")


def get_conn(cfg: Config):
    return psycopg2.connect(
        host=cfg.db_host,
        port=cfg.db_port,
        dbname=cfg.db_name,
        user=cfg.db_user,
        password=cfg.db_password,
    )

# ==========================
#  1. CARGA DE DATOS (log1p)
# ==========================

def load_wide_matrix(cfg: Config) -> pd.DataFrame:
    print("[TRAIN] Query SQL (historial sísmico)...")
    query = """
    SELECT 
        c.cell_id,
        date_trunc('day', e.event_time_utc)::date AS date,
        COUNT(*) AS event_count
    FROM public.prediction_cells c
    JOIN public.events_clean e
        ON ST_Intersects(c.geom, e.geom)
    WHERE e.magnitude >= %s
      AND e.event_time_utc >= '1960-01-01'
    GROUP BY c.cell_id, date
    """
    with get_conn(cfg) as conn:
        df_events = pd.read_sql_query(query, conn, params=(cfg.mag_min,))

    if df_events.empty:
        raise RuntimeError("No hay datos de entrenamiento.")

    df_events["date"] = pd.to_datetime(df_events["date"])

    # log1p: estabiliza días con muchos eventos
    df_events["event_count"] = np.log1p(df_events["event_count"])

    wide_df = (
        df_events.pivot(index="date", columns="cell_id", values="event_count")
        .fillna(0)
        .astype(np.float32)
    )

    full_range = pd.date_range(wide_df.index.min(), wide_df.index.max(), freq="D")
    wide_df = wide_df.reindex(full_range, fill_value=0).astype(np.float32)

    print(f"[TRAIN] Matriz Log-Transformed: {wide_df.shape} (días x celdas)")
    return wide_df

# ==========================
#  2. NORMALIZACIÓN + SECUENCIAS
# ==========================

def compute_norm_params(train_df: pd.DataFrame):
    """
    Importante: devolver Series de pandas (no numpy) para
    que evaluate_lstm pueda usarlas directamente también.
    """
    means = train_df.mean(axis=0).astype(np.float32)
    stds = (train_df.std(axis=0) + 1e-5).astype(np.float32)
    return means, stds


def make_sequences_fast(
    wide_df: pd.DataFrame,
    cfg: Config,
    means,
    stds,
    is_train: bool = True,
):
    horizon = cfg.horizons[0]
    window = cfg.window_days

    vals = wide_df.values
    # Acepta tanto Series como np.ndarray
    means_arr = np.asarray(means, dtype=np.float32)
    stds_arr = np.asarray(stds, dtype=np.float32)

    norm_vals = ((vals - means_arr) / stds_arr).astype(np.float32)
    binary_vals = (vals > 0).astype(np.float32)

    # Target: hubo al menos 1 sismo en los próximos "horizon" días
    df_bin = pd.DataFrame(binary_vals)
    target_vals = (
        df_bin.iloc[::-1]
        .rolling(window=horizon, min_periods=1)
        .max()
        .iloc[::-1]
        .shift(-horizon)
        .fillna(0)
        .values
        .astype(np.float32)
    )

    X_arrays, y_arrays = [], []
    num_cells = norm_vals.shape[1]

    limit = cfg.max_train_samples if is_train else cfg.max_eval_samples
    current_count = 0
    modo = "TRAIN" if is_train else "EVAL"

    # Geo-shuffle: evita sesgo por orden de cell_id
    cell_indices = np.arange(num_cells)
    if is_train:
        np.random.shuffle(cell_indices)

    print(f"[TRAIN] Generando ({modo}) límite={limit} (Shuffle ON)...")

    for c in cell_indices:
        if current_count >= limit:
            break

        col_data = norm_vals[:, c]
        col_bin = binary_vals[:, c]
        col_target = target_vals[:, c]

        w_data = sliding_window_view(col_data, window_shape=window)
        w_bin = sliding_window_view(col_bin, window_shape=window)

        valid_len = len(w_data) - horizon
        if valid_len <= 0:
            continue

        X_f1 = w_data[:valid_len]
        X_f2 = w_bin[:valid_len]
        y_c = col_target[window - 1 : window - 1 + valid_len]

        pos_idx = np.where(y_c > 0.5)[0]
        neg_idx = np.where(y_c <= 0.5)[0]

        if is_train:
            num_neg = int(len(neg_idx) * cfg.keep_neg_rate)
            neg_keep = (
                np.random.choice(neg_idx, num_neg, replace=False)
                if num_neg > 0
                else np.array([], dtype=int)
            )
        else:
            neg_keep = neg_idx

        idx_keep = np.concatenate([pos_idx, neg_keep]).astype(int)
        if idx_keep.size == 0:
            continue

        X_cell = np.stack([X_f1[idx_keep], X_f2[idx_keep]], axis=-1)
        y_cell = y_c[idx_keep]

        # Límite global
        rem = limit - current_count
        if rem <= 0:
            break

        if len(y_cell) > rem:
            # Priorizar positivos
            p_local = np.where(y_cell > 0.5)[0]
            n_local = np.where(y_cell <= 0.5)[0]

            if len(p_local) >= rem:
                final_idx = np.random.choice(p_local, rem, replace=False)
            else:
                rem_n = rem - len(p_local)
                chosen_n = (
                    np.random.choice(n_local, rem_n, replace=False)
                    if rem_n > 0 and len(n_local) > 0
                    else np.array([], dtype=int)
                )
                final_idx = np.concatenate([p_local, chosen_n])

            X_cell = X_cell[final_idx]
            y_cell = y_cell[final_idx]

        X_arrays.append(X_cell)
        y_arrays.append(y_cell)
        current_count += len(y_cell)

    if not X_arrays:
        raise RuntimeError("No se generaron secuencias.")

    X = np.concatenate(X_arrays, axis=0).astype(np.float32)
    y = np.concatenate(y_arrays, axis=0).astype(np.float32)

    n_pos = int((y > 0.5).sum())
    n_neg = len(y) - n_pos
    ratio = n_neg / n_pos if n_pos > 0 else float("inf")
    print(
        f"[TRAIN] Total {modo}: {X.shape} | Pos={n_pos} Neg={n_neg} "
        f"(ratio ~1:{ratio:.1f})"
    )

    return X, y

# ==========================
#  3. MODELO (Bi-LSTM 30d)
# ==========================

def build_model_v3_5(cfg: Config, n_features: int) -> tf.keras.Model:
    """
    Arquitectura basada en la V3.3 (la que mejor funcionó en hit-rate),
    con mejoras de estabilidad (clipnorm) pero sin complicarla demasiado.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(cfg.window_days, n_features)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
            ),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # LR intermedio + clipnorm para estabilidad
    opt = tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

# ==========================
#  4. TRAIN + REGISTRO
# ==========================

def train_and_register(cfg: Config):
    os.makedirs(cfg.models_dir, exist_ok=True)
    wide_df = load_wide_matrix(cfg)

    train_mask = wide_df.index <= pd.Timestamp(cfg.train_end)
    val_mask = (wide_df.index > pd.Timestamp(cfg.train_end)) & (
        wide_df.index <= pd.Timestamp(cfg.val_end)
    )
    test_mask = (wide_df.index > pd.Timestamp(cfg.val_end)) & (
        wide_df.index <= pd.Timestamp(cfg.test_end)
    )

    print("[TRAIN] Calculando parámetros de normalización (TRAIN)...")
    wide_train = wide_df[train_mask]
    means, stds = compute_norm_params(wide_train)
    cell_ids = wide_df.columns.to_numpy().tolist()

    print("[TRAIN] Generando Train (V3.5)...")
    X_train, y_train = make_sequences_fast(
        wide_train, cfg, means, stds, is_train=True
    )

    print("[TRAIN] Generando Val...")
    wide_val = wide_df[val_mask]
    X_val, y_val = make_sequences_fast(
        wide_val, cfg, means, stds, is_train=False
    )

    print("[TRAIN] Generando Test...")
    wide_test = wide_df[test_mask]
    X_test, y_test = make_sequences_fast(
        wide_test, cfg, means, stds, is_train=False
    )

    print("[TRAIN] Iniciando entrenamiento V3.5...")
    model = build_model_v3_5(cfg, n_features=X_train.shape[-1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("[TRAIN] Evaluando en TEST...")
    res = model.evaluate(
        X_test, y_test, return_dict=True, batch_size=cfg.batch_size
    )
    print(f"[TEST RESULT] AUC: {res['auc']:.4f} | Recall: {res['recall']:.4f}")

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_filename = f"model_lstm_v3.5_final_{ts}.keras"
    model.save(os.path.join(cfg.models_dir, model_filename))

    params = {
        "model_path": model_filename,
        "class_weights": "None (log1p + 2% neg + GeoShuffle)",
        "norm_means": np.asarray(means, dtype=float).tolist(),
        "norm_stds": np.asarray(stds, dtype=float).tolist(),
        "cell_ids": cell_ids,
        "window_days": cfg.window_days,
    }
    metrics = {"test": {k: float(v) for k, v in res.items()}}

    with get_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE public.model_registry SET is_active = FALSE WHERE is_active = TRUE;"
            )
            cur.execute(
                """
                INSERT INTO public.model_registry 
                (is_active, in_staging, framework, tag, train_start, train_end, 
                 horizons, mag_min, params_json, metrics_json, data_cutoff, notes)
                VALUES (
                    TRUE, FALSE, 'keras', 'LSTM_V3.5_Final',
                    '1960-01-01', %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s,
                    'V3.5: Bi-LSTM 30d + log1p + GeoShuffle + 2% neg'
                )
                """,
                (
                    cfg.train_end,
                    list(cfg.horizons),
                    cfg.mag_min,
                    json.dumps(params),
                    json.dumps(metrics),
                    cfg.test_end,
                ),
            )
            conn.commit()
            print("[TRAIN] Modelo V3.5 registrado y activado.")


if __name__ == "__main__":
    train_and_register(Config())
