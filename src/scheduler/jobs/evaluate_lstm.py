import os
import json
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix, 
    classification_report, f1_score
)
from dataclasses import dataclass
from typing import Tuple

# Importamos funciones de carga del script de entrenamiento para consistencia
# (Asegúrate de que forecast_lstm.py tenga las funciones load_wide_matrix y make_sequences_fast)
from scheduler.jobs.forecast_lstm import Config, load_wide_matrix, make_sequences_fast

# Configuración de Gráficos
plt.style.use('ggplot')
ARTIFACTS_DIR = "/app/artifacts/evaluation"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def find_optimal_threshold(y_true, y_probs):
    """
    Encuentra el umbral que maximiza el F1-Score (Balance entre Precision y Recall).
    """
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
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Curva ROC - Predicción de Sismos')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(ARTIFACTS_DIR, filename))
    plt.close()
    print(f"[GRAPH] Guardado {filename}")

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title('Matriz de Confusión')
    plt.savefig(os.path.join(ARTIFACTS_DIR, filename))
    plt.close()
    print(f"[GRAPH] Guardado {filename}")

def plot_pr_curve(y_true, y_probs, filename="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.savefig(os.path.join(ARTIFACTS_DIR, filename))
    plt.close()
    print(f"[GRAPH] Guardado {filename}")

def main():
    print("=== INICIANDO EVALUACIÓN FORMAL V2 ===")
    cfg = Config()
    
    # 1. Cargar Datos (Mismo proceso que Training)
    print("[EVAL] Cargando datos de Test...")
    wide_df = load_wide_matrix(cfg)
    
    # Filtrar solo Test
    test_mask = (wide_df.index > pd.Timestamp(cfg.val_end)) & (wide_df.index <= pd.Timestamp(cfg.test_end))
    df_test = wide_df[test_mask]
    
    if df_test.empty:
        raise RuntimeError("No hay datos en el rango de Test.")

    # 2. Generar Secuencias (Sin undersampling, queremos ver TODA la realidad)
    print("[EVAL] Generando secuencias de prueba (esto puede tardar un poco)...")
    # Nota: is_train=False para NO borrar nada. Evaluamos sobre el 100% de los datos.
    X_test, y_test = make_sequences_fast(df_test, cfg, is_train=False)
    
    print(f"[EVAL] Datos de Test listos: {X_test.shape}")

    # 3. Cargar Modelo Activo
    # Busca el archivo más reciente en la carpeta de modelos
    models = [f for f in os.listdir(cfg.models_dir) if f.endswith(".keras")]
    if not models:
        raise RuntimeError("No hay modelos .keras en artifacts/models")
    
    latest_model = max(models, key=lambda f: os.path.getctime(os.path.join(cfg.models_dir, f)))
    model_path = os.path.join(cfg.models_dir, latest_model)
    print(f"[EVAL] Evaluando modelo: {latest_model}")
    
    model = tf.keras.models.load_model(model_path)

    # 4. Predecir
    print("[EVAL] Ejecutando inferencia...")
    y_probs = model.predict(X_test, batch_size=1024, verbose=1).flatten()

    # 5. Análisis de Umbral
    print("[EVAL] Buscando Umbral Óptimo...")
    optimal_thresh = find_optimal_threshold(y_test, y_probs)
    
    # Aplicar umbral
    y_pred_opt = (y_probs >= optimal_thresh).astype(int)
    
    # 6. Reporte de Métricas
    print("\n" + "="*40)
    print(f" RESULTADOS FINALES (Umbral: {optimal_thresh:.4f})")
    print("="*40)
    print(classification_report(y_test, y_pred_opt, target_names=['No Sismo', 'Sismo']))
    
    cm = confusion_matrix(y_test, y_pred_opt)
    print("Matriz de Confusión:")
    print(cm)
    
    # Métricas aisladas
    tn, fp, fn, tp = cm.ravel()
    print(f"\nDetalle:")
    print(f" - Sismos Reales Detectados (TP): {tp}")
    print(f" - Sismos Reales Perdidos   (FN): {fn}")
    print(f" - Falsas Alarmas           (FP): {fp}")
    print(f" - Silencios Correctos      (TN): {tn}")
    
    # 7. Generar Gráficos
    plot_roc(y_test, y_probs)
    plot_pr_curve(y_test, y_probs)
    plot_confusion_matrix(y_test, y_pred_opt)
    
    print(f"\n[EXITO] Evaluación completada. Gráficos en {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()