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
from sklearn.utils import class_weight
from numpy.lib.stride_tricks import sliding_window_view

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

    window_days: int = int(os.getenv("TRAIN_WINDOW_DAYS", "30"))
    batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "256")) 
    epochs: int = int(os.getenv("TRAIN_EPOCHS", "20"))
    
    # V2: Subimos a 10% los negativos para que el modelo aprenda mejor la "calma"
    keep_neg_rate: float = 0.10 

    models_dir: str = os.getenv("MODELS_DIR", "/app/artifacts/models")

def get_conn(cfg: Config):
    return psycopg2.connect(
        host=cfg.db_host, port=cfg.db_port, dbname=cfg.db_name,
        user=cfg.db_user, password=cfg.db_password,
    )

# =========================================
#  1. CARGA DE DATOS (WIDE FORMAT)
# =========================================

def load_wide_matrix(cfg: Config) -> pd.DataFrame:
    print("[TRAIN] Query SQL (Solo celdas activas)...")
    
    query = """
    SELECT
        c.cell_id,
        date_trunc('day', e.event_time_utc)::date AS date,
        COUNT(*) as event_count
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

    print(f"[TRAIN] Datos cargados. Construyendo Matriz Pivot...")
    df_events["date"] = pd.to_datetime(df_events["date"])
    
    # Pivotar y rellenar (Días x Celdas)
    wide_df = df_events.pivot(index="date", columns="cell_id", values="event_count").fillna(0)
    
    # Rango completo de fechas
    full_range = pd.date_range(wide_df.index.min(), wide_df.index.max(), freq="D")
    wide_df = wide_df.reindex(full_range, fill_value=0)
    wide_df = wide_df.astype(np.float32)

    print(f"[TRAIN] Matriz base: {wide_df.shape} (RAM optimizada).")
    return wide_df

# =========================================
#  2. SECUENCIAS (OPTIMIZADAS)
# =========================================

def make_sequences_fast(wide_df: pd.DataFrame, cfg: Config, is_train: bool = True):
    horizon = cfg.horizons[0]
    window = cfg.window_days
    
    means = wide_df.mean()
    stds = wide_df.std() + 1e-5
    norm_vals = ((wide_df - means) / stds).values.astype(np.float32)
    binary_vals = (wide_df > 0).values.astype(np.float32)
    
    print("[TRAIN] Calculando targets futuros...")
    df_bin = pd.DataFrame(binary_vals)
    target_df = df_bin.iloc[::-1].rolling(window=horizon, min_periods=1).max().iloc[::-1].shift(-horizon).fillna(0)
    target_vals = target_df.values.astype(np.float32)
    
    X_arrays = []
    y_arrays = []
    
    num_cells = norm_vals.shape[1]
    print(f"[TRAIN] Extrayendo ventanas de {num_cells} celdas (Modo {'TRAIN' if is_train else 'TEST'})...")

    for c in range(num_cells):
        col_data = norm_vals[:, c]
        col_target = target_vals[:, c]
        col_bin = binary_vals[:, c]
        
        windows_feat1 = sliding_window_view(col_data, window_shape=window)
        windows_feat2 = sliding_window_view(col_bin, window_shape=window)
        
        valid_len = len(windows_feat1) - horizon 
        if valid_len <= 0: continue
        
        X_f1 = windows_feat1[:valid_len]
        X_f2 = windows_feat2[:valid_len]
        y_c  = col_target[window-1 : window-1+valid_len]
        
        if is_train:
            mask_sismos = (y_c > 0.5)
            mask_keep_zeros = (np.random.rand(len(y_c)) < cfg.keep_neg_rate)
            mask_final = mask_sismos | mask_keep_zeros
        else:
            mask_final = np.ones(len(y_c), dtype=bool)

        if mask_final.sum() == 0: continue
            
        X_cell = np.stack([X_f1[mask_final], X_f2[mask_final]], axis=-1)
        y_cell = y_c[mask_final]
        
        X_arrays.append(X_cell)
        y_arrays.append(y_cell)
    
    if not X_arrays: raise RuntimeError("No se generaron secuencias.")
    return np.concatenate(X_arrays, axis=0), np.concatenate(y_arrays, axis=0)

# ==========================
#  3. MODELO V2
# ==========================

def build_lstm_model_v2(cfg: Config, n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cfg.window_days, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=metrics
    )
    return model

# ==================================
#  4. PIPELINE DE ENTRENAMIENTO
# ==================================

def train_and_register(cfg: Config):
    os.makedirs(cfg.models_dir, exist_ok=True)

    # 1. Carga
    wide_df = load_wide_matrix(cfg) 

    # 2. Split
    train_mask = wide_df.index <= pd.Timestamp(cfg.train_end)
    val_mask   = (wide_df.index > pd.Timestamp(cfg.train_end)) & (wide_df.index <= pd.Timestamp(cfg.val_end))
    test_mask  = (wide_df.index > pd.Timestamp(cfg.val_end))   & (wide_df.index <= pd.Timestamp(cfg.test_end))

    # 3. Generación
    print("[TRAIN] Generando X_train...")
    X_train, y_train = make_sequences_fast(wide_df[train_mask], cfg, is_train=True)
    print(f"[TRAIN] X_train shape: {X_train.shape}")
    
    print("[TRAIN] Generando X_val...")
    X_val, y_val = make_sequences_fast(wide_df[val_mask], cfg, is_train=False)
    
    print("[TRAIN] Generando X_test...")
    X_test, y_test = make_sequences_fast(wide_df[test_mask], cfg, is_train=False)

    # 4. PESOS MANUALES (LA CLAVE DE LA V2)
    # Usamos 1:10. Suficiente para detectar, pero no para alucinar.
    class_weights_dict = {0: 1.0, 1: 10.0}
    print(f"[TRAIN] Pesos Manuales V2: {class_weights_dict}")

    # 5. Entrenar
    model = build_lstm_model_v2(cfg, n_features=X_train.shape[-1])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Evaluar
    print("[TRAIN] Evaluando en Test...")
    res = model.evaluate(X_test, y_test, return_dict=True)
    print(f"[RESULTADOS] AUC: {res['auc']:.4f} | Recall: {res['recall']:.4f} | Prec: {res['precision']:.4f}")

    # Guardar
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_filename = f"model_lstm_v2_{ts}.keras"
    model.save(os.path.join(cfg.models_dir, model_filename))

    # Registro DB (V2)
    params = {"model_path": model_filename, "class_weights": class_weights_dict}
    metrics = {"test": {k: float(v) for k, v in res.items()}}
    
    with get_conn(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE public.model_registry SET is_active = FALSE WHERE is_active = TRUE;")
            
            # Insertamos como V2
            cur.execute("""
                INSERT INTO public.model_registry 
                (is_active, in_staging, framework, tag, train_start, train_end, horizons, mag_min, params_json, metrics_json, data_cutoff, notes)
                VALUES (TRUE, FALSE, 'keras', 'LSTM_V2_Optimized', '1960-01-01', %s, %s, %s, %s::jsonb, %s::jsonb, %s, 'V2: Pesos Manuales 1:10 para reducir falsas alarmas')
            """, (cfg.train_end, list(cfg.horizons), cfg.mag_min, json.dumps(params), json.dumps(metrics), cfg.test_end))
            
            conn.commit()
            print("[TRAIN] Modelo V2 registrado y activado exitosamente.")

if __name__ == "__main__":
    train_and_register(Config())