# forecast_lstm.py (V3.1 - Stable / Undersampling + Sin ClassWeight)

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

    window_days: int = int(os.getenv("TRAIN_WINDOW_DAYS", "30"))
    batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "512"))
    epochs: int = int(os.getenv("TRAIN_EPOCHS", "30"))

    # Undersampling agresivo de negativos en entrenamiento
    keep_neg_rate: float = float(os.getenv("TRAIN_KEEP_NEG_RATE", "0.01"))

    # Límite global de muestras para TRAIN (RAM)
    max_train_samples: int = int(os.getenv("TRAIN_MAX_SAMPLES", "2000000"))

    # Límite para VAL/TEST (para no explotar RAM)
    max_eval_samples: int = int(os.getenv("EVAL_MAX_SAMPLES", "500000"))

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
#  1. CARGA
# ==========================


def load_wide_matrix(cfg: Config) -> pd.DataFrame:
    print("[TRAIN] Query SQL...")
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
        raise RuntimeError("No hay datos.")

    print("[TRAIN] Pivotando matriz...")
    df_events["date"] = pd.to_datetime(df_events["date"])
    wide_df = (
        df_events.pivot(index="date", columns="cell_id", values="event_count")
        .fillna(0)
        .astype(np.float32)
    )
    full_range = pd.date_range(
        wide_df.index.min(), wide_df.index.max(), freq="D"
    )
    wide_df = wide_df.reindex(full_range, fill_value=0).astype(np.float32)

    print(f"[TRAIN] Matriz base: {wide_df.shape}")
    return wide_df

# ==========================
#  2. NORMALIZACIÓN + SECUENCIAS
# ==========================


def compute_norm_params(train_df: pd.DataFrame):
    means = train_df.mean(axis=0).astype(np.float32)
    stds = (train_df.std(axis=0) + 1e-5).astype(np.float32)
    return means, stds


def make_sequences_fast(
    wide_df: pd.DataFrame,
    cfg: Config,
    means: np.ndarray,
    stds: np.ndarray,
    is_train: bool = True,
):
    """
    Genera X, y a partir de wide_df usando:
      - normalización consistente con 'means' y 'stds'
      - undersampling agresivo en TRAIN (keep_neg_rate)
      - límite global de muestras (train/eval)
      - nunca tira positivos por límite global (siempre prioritarios)
    """
    horizon = cfg.horizons[0]
    window = cfg.window_days

    vals = wide_df.values.astype(np.float32)  # [T, C]
    norm_vals = ((vals - means) / stds).astype(np.float32)  # [T, C]
    binary_vals = (vals > 0).astype(np.float32)

    # Targets futuros
    df_bin = pd.DataFrame(binary_vals)
    target_df = (
        df_bin.iloc[::-1]
        .rolling(window=horizon, min_periods=1)
        .max()
        .iloc[::-1]
        .shift(-horizon)
        .fillna(0)
    )
    target_vals = target_df.values.astype(np.float32)  # [T, C]

    X_arrays, y_arrays = [], []
    num_cells = norm_vals.shape[1]

    limit = cfg.max_train_samples if is_train else cfg.max_eval_samples
    current_count = 0

    modo = "TRAIN" if is_train else "EVAL"
    print(f"[TRAIN] Generando ventanas ({modo}) con límite={limit}...")

    for c in range(num_cells):
        if current_count >= limit:
            break

        col_data = norm_vals[:, c]       # [T]
        col_target = target_vals[:, c]   # [T]
        col_bin = binary_vals[:, c]      # [T]

        windows_feat1 = sliding_window_view(col_data, window_shape=window)
        windows_feat2 = sliding_window_view(col_bin, window_shape=window)

        valid_len = len(windows_feat1) - horizon
        if valid_len <= 0:
            continue

        X_f1 = windows_feat1[:valid_len]       # [valid_len, window]
        X_f2 = windows_feat2[:valid_len]       # [valid_len, window]
        y_c = col_target[window - 1 : window - 1 + valid_len]  # [valid_len]

        # Construimos todas las ventanas de esa celda
        X_all = np.stack([X_f1, X_f2], axis=-1)  # [valid_len, window, 2]

        # Positivos y negativos
        pos_mask = y_c > 0.5
        neg_mask = ~pos_mask

        # Extraemos los índices
        pos_idx = np.where(pos_mask)[0]
        neg_idx = np.where(neg_mask)[0]

        # TRAIN: undersampling agresivo de negativos vía keep_neg_rate
        if is_train:
            num_neg_keep = int(len(neg_idx) * cfg.keep_neg_rate)
            if num_neg_keep > 0:
                neg_idx_keep = np.random.choice(
                    neg_idx, num_neg_keep, replace=False
                )
            else:
                neg_idx_keep = np.array([], dtype=int)
        else:
            # EVAL: no hacemos undersampling por probabilidad,
            # solo limitamos luego por el límite global.
            neg_idx_keep = neg_idx

        # Siempre consideramos todos los positivos candidatos (no se tiran aquí)
        idx_cell = np.concatenate([pos_idx, neg_idx_keep])
        if idx_cell.size == 0:
            continue

        X_cell = X_all[idx_cell]
        y_cell = y_c[idx_cell]

        # Aplicamos el límite global priorizando positivos
        remaining = limit - current_count
        if remaining <= 0:
            break

        if len(y_cell) > remaining:
            # Separamos positivos y negativos locales
            pos_mask_local = y_cell > 0.5
            neg_mask_local = ~pos_mask_local

            X_pos = X_cell[pos_mask_local]
            y_pos = y_cell[pos_mask_local]
            X_neg = X_cell[neg_mask_local]
            y_neg = y_cell[neg_mask_local]

            n_pos = len(y_pos)
            n_neg = len(y_neg)

            if remaining <= n_pos:
                # No alcanza para todos los positivos -> sample de positivos
                idx = np.random.choice(n_pos, remaining, replace=False)
                X_cell = X_pos[idx]
                y_cell = y_pos[idx]
            else:
                # Guardamos todos los positivos
                remaining_for_negs = remaining - n_pos
                if n_neg > 0 and remaining_for_negs < n_neg:
                    idx = np.random.choice(
                        n_neg, remaining_for_negs, replace=False
                    )
                    X_neg = X_neg[idx]
                    y_neg = y_neg[idx]

                X_cell = (
                    np.concatenate([X_pos, X_neg], axis=0)
                    if n_neg > 0
                    else X_pos
                )
                y_cell = (
                    np.concatenate([y_pos, y_neg], axis=0)
                    if n_neg > 0
                    else y_pos
                )

        X_arrays.append(X_cell)
        y_arrays.append(y_cell)
        current_count += len(y_cell)

        if (c + 1) % 500 == 0:
            print(
                f"[TRAIN] Procesadas {c+1}/{num_cells} celdas | "
                f"muestras acumuladas: {current_count}"
            )

    if not X_arrays:
        raise RuntimeError("No se generaron secuencias.")

    X = np.concatenate(X_arrays, axis=0).astype(np.float32)
    y = np.concatenate(y_arrays, axis=0).astype(np.float32)

    print(
        f"[TRAIN] Secuencias generadas ({modo}): {X.shape[0]} muestras "
        f"(límite {limit})"
    )

    return X, y

# ==========================
#  3. MODELO (Simplificado y Estable)
# ==========================


def build_lstm_model_v3(cfg: Config, n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(cfg.window_days, n_features)),
            # Menos complejidad para evitar oscilaciones raras
            tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2),
            tf.keras.layers.GlobalMaxPooling1D(),  # captura el pico de señal
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
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

    # Splits temporales
    train_mask = wide_df.index <= pd.Timestamp(cfg.train_end)
    val_mask = (wide_df.index > pd.Timestamp(cfg.train_end)) & (
        wide_df.index <= pd.Timestamp(cfg.val_end)
    )
    test_mask = (wide_df.index > pd.Timestamp(cfg.val_end)) & (
        wide_df.index <= pd.Timestamp(cfg.test_end)
    )

    wide_train = wide_df[train_mask]
    wide_val = wide_df[val_mask]
    # wide_test se usará en evaluate_lstm, no aquí.

    print("[TRAIN] Calculando medias (Train Only)...")
    means, stds = compute_norm_params(wide_train)
    cell_ids = wide_df.columns.to_numpy().tolist()

    print("[TRAIN] Generando Datasets...")
    X_train, y_train = make_sequences_fast(
        wide_train, cfg, means.values, stds.values, is_train=True
    )
    num_pos_train = int((y_train > 0.5).sum())
    num_neg_train = int((y_train <= 0.5).sum())
    print(
        f"   -> X_train: {X_train.shape} | Positivos: {num_pos_train} | "
        f"Negativos: {num_neg_train} | ratio neg/pos={num_neg_train / max(num_pos_train,1):.1f}"
    )

    X_val, y_val = make_sequences_fast(
        wide_val, cfg, means.values, stds.values, is_train=False
    )
    num_pos_val = int((y_val > 0.5).sum())
    num_neg_val = int((y_val <= 0.5).sum())
    print(
        f"   -> X_val:   {X_val.shape} | Positivos: {num_pos_val} | "
        f"Negativos: {num_neg_val} | ratio neg/pos={num_neg_val / max(num_pos_val,1):.1f}"
    )

    print(
        "[TRAIN] Iniciando entrenamiento sin class_weight "
        "(balance natural por undersampling)..."
    )

    model = build_lstm_model_v3(cfg, n_features=X_train.shape[-1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
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

    # Evaluación final en VALIDACIÓN (con mejores pesos restaurados)
    print("[TRAIN] Evaluando en Validación...")
    res_val = model.evaluate(X_val, y_val, return_dict=True, batch_size=cfg.batch_size)
    print(
        "[RESULTADOS] Val -> "
        f"AUC: {res_val['auc']:.4f} | "
        f"Prec: {res_val['precision']:.4f} | "
        f"Recall: {res_val['recall']:.4f}"
    )

    # Guardar modelo
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_filename = f"model_lstm_v3_stable_{ts}.keras"
    model_path = os.path.join(cfg.models_dir, model_filename)
    model.save(model_path)
    print(f"[TRAIN] Modelo guardado en: {model_path}")

    # Registro en DB
    params = {
        "model_path": model_filename,
        "class_weights": "None (Natural Sampling)",
        "norm_means": means.values.astype(float).tolist(),
        "norm_stds": stds.values.astype(float).tolist(),
        "cell_ids": cell_ids,
        "window_days": cfg.window_days,
    }
    metrics = {"val": {k: float(v) for k, v in res_val.items()}}

    with get_conn(cfg) as conn:
        with conn.cursor() as cur:
            # Desactivar modelos anteriores
            cur.execute(
                "UPDATE public.model_registry SET is_active = FALSE "
                "WHERE is_active = TRUE;"
            )

            # Insertar como V3 estable
            cur.execute(
                """
                INSERT INTO public.model_registry 
                (is_active, in_staging, framework, tag, 
                 train_start, train_end, horizons, mag_min, 
                 params_json, metrics_json, data_cutoff, notes)
                VALUES (
                    TRUE, FALSE, 'keras', 'LSTM_V3_Stable',
                    '1960-01-01', %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s,
                    'V3.1: Undersampling 1%% + Sin ClassWeight + MaxPooling + límites de muestras'
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
            print("[TRAIN] Modelo V3 (Estable) registrado y activado.")


if __name__ == "__main__":
    train_and_register(Config())