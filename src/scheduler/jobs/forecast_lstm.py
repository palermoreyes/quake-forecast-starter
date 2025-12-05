# forecast_lstm.py (V3.4 - Stabilized Deep Bi-LSTM)

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

    train_end: dt.date = dt.date.fromisoformat(os.getenv("SPLIT_TRAIN_END", "2018-12-31"))
    val_end: dt.date = dt.date.fromisoformat(os.getenv("SPLIT_VAL_END", "2021-12-31"))
    test_end: dt.date = dt.date.fromisoformat(os.getenv("SPLIT_TEST_END", "2023-12-31"))

    # Mantenemos 30 días (La ventana probada y robusta)
    window_days: int = 30
    
    # Batch Size moderado para estabilidad
    batch_size: int = 512
    
    # Más épocas, el Clipping hace que aprenda "lento pero seguro"
    epochs: int = 50 

    # Sampling 2% (Tu balance ideal)
    keep_neg_rate: float = 0.02

    max_train_samples: int = 4000000
    max_eval_samples: int = 1000000

    models_dir: str = os.getenv("MODELS_DIR", "/app/artifacts/models")


def get_conn(cfg: Config):
    return psycopg2.connect(
        host=cfg.db_host, port=cfg.db_port, dbname=cfg.db_name,
        user=cfg.db_user, password=cfg.db_password,
    )

# ==========================
#  1. CARGA DE DATOS
# ==========================

def load_wide_matrix(cfg: Config) -> pd.DataFrame:
    print("[TRAIN] Query SQL...")
    query = """
    SELECT c.cell_id, date_trunc('day', e.event_time_utc)::date AS date, COUNT(*) AS event_count
    FROM public.prediction_cells c
    JOIN public.events_clean e ON ST_Intersects(c.geom, e.geom)
    WHERE e.magnitude >= %s AND e.event_time_utc >= '1960-01-01'
    GROUP BY c.cell_id, date
    """
    with get_conn(cfg) as conn:
        df_events = pd.read_sql_query(query, conn, params=(cfg.mag_min,))

    df_events["date"] = pd.to_datetime(df_events["date"])
    
    # LOG-TRANSFORM: Clave para reducir la escala de valores extremos
    df_events["event_count"] = np.log1p(df_events["event_count"])
    
    wide_df = df_events.pivot(index="date", columns="cell_id", values="event_count").fillna(0).astype(np.float32)
    
    full_range = pd.date_range(wide_df.index.min(), wide_df.index.max(), freq="D")
    wide_df = wide_df.reindex(full_range, fill_value=0).astype(np.float32)

    print(f"[TRAIN] Matriz Log-Transformed: {wide_df.shape}")
    return wide_df

# ==========================
#  2. NORMALIZACIÓN
# ==========================

def compute_norm_params(train_df: pd.DataFrame):
    means = train_df.mean(axis=0).astype(np.float32)
    stds = (train_df.std(axis=0) + 1e-5).astype(np.float32)
    return means, stds

def make_sequences_fast(wide_df, cfg, means, stds, is_train=True):
    horizon = cfg.horizons[0]; window = cfg.window_days
    
    vals = wide_df.values
    norm_vals = ((vals - means) / stds).astype(np.float32)
    binary_vals = (vals > 0).astype(np.float32)
    
    df_bin = pd.DataFrame(binary_vals)
    target_vals = df_bin.iloc[::-1].rolling(window=horizon).max().iloc[::-1].shift(-horizon).fillna(0).values.astype(np.float32)
    
    X_arrays, y_arrays = [], []
    num_cells = norm_vals.shape[1]
    limit = cfg.max_train_samples if is_train else cfg.max_eval_samples
    current_count = 0
    modo = "TRAIN" if is_train else "EVAL"

    cell_indices = np.arange(num_cells)
    if is_train: np.random.shuffle(cell_indices)

    print(f"[TRAIN] Generando ({modo}) límite={limit}...")

    for c in cell_indices:
        if current_count >= limit: break

        w_data = sliding_window_view(norm_vals[:, c], window)
        w_bin = sliding_window_view(binary_vals[:, c], window)
        
        valid = len(w_data) - horizon
        if valid <= 0: continue
        
        X_f1 = w_data[:valid]; X_f2 = w_bin[:valid]
        y_c = target_vals[window-1 : window-1+valid, c]
        
        pos_idx = np.where(y_c > 0.5)[0]
        neg_idx = np.where(y_c <= 0.5)[0]
        
        if is_train:
            num_neg = int(len(neg_idx) * cfg.keep_neg_rate)
            neg_keep = np.random.choice(neg_idx, num_neg, replace=False) if num_neg > 0 else []
        else:
            neg_keep = neg_idx
            
        idx_keep = np.concatenate([pos_idx, neg_keep]).astype(int)
        if len(idx_keep) == 0: continue
        
        X_cell = np.stack([X_f1[idx_keep], X_f2[idx_keep]], axis=-1)
        y_cell = y_c[idx_keep]
        
        rem = limit - current_count
        if rem <= 0: break
        
        if len(y_cell) > rem:
            p_local = np.where(y_cell > 0.5)[0]
            n_local = np.where(y_cell <= 0.5)[0]
            if len(p_local) >= rem:
                final_idx = np.random.choice(p_local, rem, replace=False)
            else:
                rem_n = rem - len(p_local)
                chosen_n = np.random.choice(n_local, rem_n, replace=False)
                final_idx = np.concatenate([p_local, chosen_n])
            X_cell = X_cell[final_idx]; y_cell = y_cell[final_idx]

        X_arrays.append(X_cell)
        y_arrays.append(y_cell)
        current_count += len(y_cell)
        
    if not X_arrays: raise RuntimeError("No secuencias.")
    
    return np.concatenate(X_arrays).astype(np.float32), np.concatenate(y_arrays).astype(np.float32)

# ==========================
#  3. MODELO (V3.4 - STABILIZED DEEP BI-LSTM)
# ==========================

def build_model_v3_4(cfg: Config, n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cfg.window_days, n_features)),
        
        # CAPA 1: Bi-LSTM (128 unidades)
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)
        ),
        # BatchNormalization: La clave para que no colapse
        tf.keras.layers.BatchNormalization(),
        
        # CAPA 2: Bi-LSTM Compresión (64 unidades)
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.3)
        ),
        tf.keras.layers.BatchNormalization(),
        
        # CAPA DENSA "Wide"
        tf.keras.layers.Dense(64, activation="gelu"), # GELU es más moderno y suave que ReLU
        tf.keras.layers.Dropout(0.4),
        
        # SALIDA
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # OPTIMIZADOR CON GRADIENT CLIPPING
    # clipnorm=1.0 corta los picos de error para que no se vaya a infinito
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    
    model.compile(
        optimizer=opt, 
        loss="binary_crossentropy", 
        metrics=[
            tf.keras.metrics.AUC(name="auc"), 
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    return model

# ==========================
#  4. TRAIN
# ==========================

def train_and_register(cfg: Config):
    os.makedirs(cfg.models_dir, exist_ok=True)
    wide_df = load_wide_matrix(cfg)

    train_mask = wide_df.index <= pd.Timestamp(cfg.train_end)
    val_mask = (wide_df.index > pd.Timestamp(cfg.train_end)) & (wide_df.index <= pd.Timestamp(cfg.val_end))
    test_mask = (wide_df.index > pd.Timestamp(cfg.val_end)) & (wide_df.index <= pd.Timestamp(cfg.test_end))

    print("[TRAIN] Calculando parámetros...")
    wide_train = wide_df[train_mask]
    means, stds = compute_norm_params(wide_train)
    cell_ids = wide_df.columns.to_numpy().tolist()

    print("[TRAIN] Generando Train (V3.4 Stable)...")
    X_train, y_train = make_sequences_fast(wide_train, cfg, means.values, stds.values, is_train=True)
    print(f"   -> Train: {X_train.shape} (Positivos: {int((y_train>0.5).sum())})")

    print("[TRAIN] Generando Val...")
    wide_val = wide_df[val_mask]
    X_val, y_val = make_sequences_fast(wide_val, cfg, means.values, stds.values, is_train=False)

    print("[TRAIN] Generando Test...")
    wide_test = wide_df[test_mask]
    X_test, y_test = make_sequences_fast(wide_test, cfg, means.values, stds.values, is_train=False)

    print("[TRAIN] Iniciando entrenamiento V3.4...")
    model = build_model_v3_4(cfg, n_features=X_train.shape[-1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs, batch_size=cfg.batch_size,
        callbacks=callbacks, verbose=1,
    )

    print("[TRAIN] Evaluando...")
    res = model.evaluate(X_test, y_test, return_dict=True, batch_size=cfg.batch_size)
    print(f"[TEST RESULT] AUC: {res['auc']:.4f}")

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_filename = f"model_lstm_v3.4_stabilized_{ts}.keras"
    model.save(os.path.join(cfg.models_dir, model_filename))

    params = {
        "model_path": model_filename,
        "class_weights": "None (Log + Clipping + BatchNorm)",
        "norm_means": means.values.astype(float).tolist(),
        "norm_stds": stds.values.astype(float).tolist(),
        "cell_ids": cell_ids,
        "window_days": cfg.window_days
    }
    metrics = {"test": {k: float(v) for k, v in res.items()}}

    with get_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE public.model_registry SET is_active = FALSE WHERE is_active = TRUE;")
            cur.execute("""
                INSERT INTO public.model_registry 
                (is_active, in_staging, framework, tag, train_start, train_end, horizons, mag_min, params_json, metrics_json, data_cutoff, notes)
                VALUES (TRUE, FALSE, 'keras', 'LSTM_V3.4_Stabilized', '1960-01-01', %s, %s, %s, %s::jsonb, %s::jsonb, %s, 'V3.4: Deep Bi-LSTM Stabilized (BN + Clip)')
            """, (cfg.train_end, list(cfg.horizons), cfg.mag_min, json.dumps(params), json.dumps(metrics), cfg.test_end))
            conn.commit()
            print("[TRAIN] Modelo V3.4 Registrado.")

if __name__ == "__main__":
    train_and_register(Config())