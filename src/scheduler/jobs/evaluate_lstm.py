import os
import json
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import psycopg2
import psycopg2.extras
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report
)
from dataclasses import dataclass

# Importamos funciones del training V3 para consistencia
from scheduler.jobs.forecast_lstm import (
    Config,
    load_wide_matrix,
    compute_norm_params,
    make_sequences_fast,
    get_conn
)

# Configuración de Gráficos
plt.style.use("ggplot")
ARTIFACTS_DIR = "/app/artifacts/evaluation"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def get_active_model_path(cfg):
    """Obtiene la ruta del modelo ACTIVO desde la BD, no por fecha."""
    with get_conn(cfg) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT params_json 
                FROM public.model_registry 
                WHERE is_active = TRUE 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            row = cur.fetchone()
            
    if not row:
        raise RuntimeError("No hay modelo activo en la base de datos.")
        
    params = row['params_json']
    if isinstance(params, str):
        params = json.loads(params)
        
    filename = params.get('model_path')
    if not filename:
        raise RuntimeError("El modelo activo no tiene 'model_path' en sus parámetros.")
        
    return os.path.join(cfg.models_dir, filename)

def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    print(f"[EVAL] Mejor F1-Score: {f1_scores[best_idx]:.4f} en Umbral: {best_thresh:.4f}")
    return best_thresh

def plot_roc(y_true, y_probs, filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Curva ROC - Predicción de Sismos (Test)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(ARTIFACTS_DIR, filename))
    plt.close()
    print(f"[GRAPH] Guardado {filename}")

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicción")
    plt.ylabel("Realidad")
    plt.title("Matriz de Confusión (Test)")
    plt.savefig(os.path.join(ARTIFACTS_DIR, filename))
    plt.close()
    print(f"[GRAPH] Guardado {filename}")

def plot_pr_curve(y_true, y_probs, filename="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall (Test)")
    plt.savefig(os.path.join(ARTIFACTS_DIR, filename))
    plt.close()
    print(f"[GRAPH] Guardado {filename}")

def main():
    print("=== INICIANDO EVALUACIÓN FORMAL (MODELO ACTIVO) ===")
    cfg = Config()

    # 1. Cargar datos
    print("[EVAL] Cargando datos...")
    wide_df = load_wide_matrix(cfg)

    # 2. Splits
    train_mask = wide_df.index <= pd.Timestamp(cfg.train_end)
    test_mask = (wide_df.index > pd.Timestamp(cfg.val_end)) & (wide_df.index <= pd.Timestamp(cfg.test_end))
    df_train = wide_df[train_mask]
    df_test = wide_df[test_mask]

    # 3. Normalización (Train Only)
    print("[EVAL] Calculando parámetros de normalización...")
    means, stds = compute_norm_params(df_train)

    # 4. Generar Test
    print("[EVAL] Generando secuencias de prueba...")
    X_test, y_test = make_sequences_fast(df_test, cfg, means=means.values, stds=stds.values, is_train=False)
    print(f"[EVAL] Datos Test: {X_test.shape}")

    # 5. CARGAR MODELO ACTIVO (CORRECCIÓN)
    model_path = get_active_model_path(cfg)
    print(f"[EVAL] Evaluando modelo ACTIVO: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(model_path)

    # 6. Inferencia
    print("[EVAL] Ejecutando inferencia...")
    y_probs = model.predict(X_test, batch_size=1024, verbose=1).flatten()

    # 7. Métricas
    print("[EVAL] Buscando Umbral Óptimo...")
    optimal_thresh = find_optimal_threshold(y_test, y_probs)
    y_pred_opt = (y_probs >= optimal_thresh).astype(int)

    # 8. Reporte
    print("\n" + "=" * 50)
    print(f" RESULTADOS FINALES (Umbral {optimal_thresh:.4f})")
    print("=" * 50)
    print(classification_report(y_test, y_pred_opt, target_names=["No Sismo", "Sismo"]))
    
    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")

    # 9. Gráficos
    plot_roc(y_test, y_probs)
    plot_pr_curve(y_test, y_probs)
    plot_confusion_matrix(y_test, y_pred_opt)
    print(f"\n[EXITO] Gráficos guardados en {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()