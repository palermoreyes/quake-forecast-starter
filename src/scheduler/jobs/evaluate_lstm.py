import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import psycopg2
import psycopg2.extras

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    average_precision_score
)

# Importamos funciones del training para consistencia
from scheduler.jobs.forecast_lstm import (
    Config,
    load_wide_matrix,
    compute_norm_params,
    make_sequences_fast,
    get_conn
)

# =========================
# CONFIGURACIÓN GENERAL
# =========================
ARTIFACTS_DIR = "/app/artifacts/evaluation"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Gráficos “publicables”
FIGSIZE = (6.8, 4.2)     # proporción cómoda para Word
DPI = 300                # alta resolución para tesis
GRID_ALPHA = 0.15        # rejilla sutil (no borrador)

def _save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()

def _fmt_int(x: int) -> str:
    # separador de miles para que sea legible en tesis
    return f"{x:,}".replace(",", " ")

# =========================
# MODELO ACTIVO
# =========================
def get_active_model_path(cfg):
    """Obtiene la ruta del modelo ACTIVO desde la BD (registry), no por fecha."""
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

    params = row["params_json"]
    if isinstance(params, str):
        params = json.loads(params)

    filename = params.get("model_path")
    if not filename:
        raise RuntimeError("El modelo activo no tiene 'model_path' en sus parámetros.")

    return os.path.join(cfg.models_dir, filename)

# =========================
# SELECCIÓN DE UMBRAL
# =========================
def find_optimal_threshold_fbeta(y_true, y_probs, beta=1.0):
    """
    Selección de umbral maximizando F-beta (beta>1 prioriza Recall).
    Importante en desbalance extremo (eventos raros).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    # thresholds tiene N-1; precision/recall tienen N
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    beta2 = beta ** 2
    fbeta = (1 + beta2) * (precision_t * recall_t) / (beta2 * precision_t + recall_t + 1e-12)

    best_idx = int(np.argmax(fbeta))
    best_thresh = float(thresholds[best_idx])
    best_fbeta = float(fbeta[best_idx])
    return best_thresh, best_fbeta

# =========================
# GRÁFICOS “PUBLICABLES”
# =========================
def plot_roc_publicable(y_true, y_probs, thr, outdir, filename="roc_test.png"):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    idx = int(np.argmin(np.abs(thresholds - thr)))

    plt.figure(figsize=FIGSIZE)
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Azar")
    plt.scatter(fpr[idx], tpr[idx], s=45, label=f"Umbral = {thr:.4f}")

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR / Recall)")
    plt.title("Curva ROC (conjunto de prueba)")
    plt.legend(frameon=False, loc="lower right")
    plt.grid(alpha=GRID_ALPHA)
    _save_fig(os.path.join(outdir, filename))

    return float(roc_auc)

def plot_pr_publicable(y_true, y_probs, outdir, filename="pr_test.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    base = float(np.mean(y_true))

    # Escala “real” (para que la curva se vea en desbalance extremo)
    # Usamos cuantiles para no “inflar” por puntos aislados.
    p99 = float(np.quantile(precision, 0.995))
    y_max = max(base * 8, p99) * 1.10
    y_max = min(y_max, 0.02)  # techo razonable (ajustable)

    plt.figure(figsize=FIGSIZE)
    plt.step(recall, precision, where="post", linewidth=2, label=f"AP = {ap:.4f}")
    plt.fill_between(recall, precision, step="post", alpha=0.12)
    plt.axhline(base, linestyle="--", linewidth=1.5, label=f"Base = {base:.6f}")

    plt.xlim(0, 1)
    plt.ylim(0, y_max)
    plt.xlabel("Exhaustividad (Recall)")
    plt.ylabel("Precisión (Precision)")
    plt.title("Curva Precisión–Recall (conjunto de prueba)")
    plt.legend(frameon=False, loc="upper right")
    plt.grid(alpha=GRID_ALPHA)
    _save_fig(os.path.join(outdir, filename))

    return float(ap), float(base), float(y_max)

def plot_pr_log_publicable(y_true, y_probs, outdir, filename="pr_test_log.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    base = float(np.mean(y_true))

    # límites para escala log
    p99 = float(np.quantile(precision, 0.995))
    y_min = max(base / 10.0, 1e-6)
    y_max = max(p99, base) * 2.0

    plt.figure(figsize=FIGSIZE)
    plt.step(recall, precision, where="post", linewidth=2, label=f"AP = {ap:.4f}")
    plt.axhline(base, linestyle="--", linewidth=1.5, label=f"Base = {base:.6f}")

    plt.yscale("log")
    plt.xlim(0, 1)
    plt.ylim(y_min, y_max)
    plt.xlabel("Exhaustividad (Recall)")
    plt.ylabel("Precisión (escala log)")
    plt.title("Curva Precisión–Recall (log)")
    plt.legend(frameon=False, loc="lower left")
    plt.grid(alpha=GRID_ALPHA, which="both")
    _save_fig(os.path.join(outdir, filename))

def plot_lift_curve(y_true, y_probs, outdir, filename="lift_test.png"):
    """
    Lift = Precision / Prevalencia
    Útil y muy defendible en tesis cuando el evento es raro (muestra cuántas veces mejor que azar).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    base = float(np.mean(y_true))
    lift = precision / (base + 1e-12)

    # rango razonable
    l99 = float(np.quantile(lift, 0.995))
    y_max = max(2.0, l99) * 1.10

    plt.figure(figsize=FIGSIZE)
    plt.step(recall, lift, where="post", linewidth=2)
    plt.axhline(1.0, linestyle="--", linewidth=1.5, label="Azar (Lift = 1)")

    plt.xlim(0, 1)
    plt.ylim(0, y_max)
    plt.xlabel("Exhaustividad (Recall)")
    plt.ylabel("Lift (Precisión / Prevalencia)")
    plt.title("Curva de Lift (conjunto de prueba)")
    plt.legend(frameon=False, loc="upper right")
    plt.grid(alpha=GRID_ALPHA)
    _save_fig(os.path.join(outdir, filename))

def plot_confusion_counts_and_norm(y_true, y_pred, outdir,
                                   fn_counts="cm_test_conteos.png",
                                   fn_norm="cm_test_normalizada.png"):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # --- Conteos
    plt.figure(figsize=(6.0, 4.6))
    plt.imshow(cm)
    plt.title("Matriz de confusión (conteos)")
    plt.xlabel("Predicción")
    plt.ylabel("Realidad")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, _fmt_int(int(v)), ha="center", va="center")

    plt.grid(False)
    _save_fig(os.path.join(outdir, fn_counts))

    # --- Normalizada por realidad (filas)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6.0, 4.6))
    plt.imshow(cm_norm, vmin=0, vmax=1)
    plt.title("Matriz de confusión (normalizada por realidad)")
    plt.xlabel("Predicción")
    plt.ylabel("Realidad")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for (i, j), v in np.ndenumerate(cm_norm):
        plt.text(j, i, f"{v:.3f}", ha="center", va="center")

    plt.grid(False)
    _save_fig(os.path.join(outdir, fn_norm))

    return int(tn), int(fp), int(fn), int(tp)

# =========================
# MAIN
# =========================
def main():
    print("=== INICIANDO EVALUACIÓN FORMAL (MODELO ACTIVO) ===")
    cfg = Config()

    # 1) Cargar datos
    print("[EVAL] Cargando matriz wide...")
    wide_df = load_wide_matrix(cfg)

    # 2) Splits (cronológicos)
    train_mask = wide_df.index <= pd.Timestamp(cfg.train_end)
    test_mask = (wide_df.index > pd.Timestamp(cfg.val_end)) & (wide_df.index <= pd.Timestamp(cfg.test_end))

    df_train = wide_df[train_mask]
    df_test = wide_df[test_mask]

    # 3) Normalización (solo Train)
    print("[EVAL] Calculando parámetros de normalización (Train)...")
    means, stds = compute_norm_params(df_train)

    # 4) Secuencias Test
    print("[EVAL] Generando secuencias del conjunto de prueba...")
    X_test, y_test = make_sequences_fast(df_test, cfg, means=means, stds=stds, is_train=False)
    print(f"[EVAL] X_test: {X_test.shape} | y_test: {y_test.shape}")

    # 5) Cargar modelo activo
    model_path = get_active_model_path(cfg)
    print(f"[EVAL] Modelo activo: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(model_path)

    # 6) Inferencia
    print("[EVAL] Ejecutando inferencia...")
    y_probs = model.predict(X_test, batch_size=1024, verbose=1).flatten()

    # Info de prevalencia
    prevalence = float(np.mean(y_test))
    print(f"[INFO] Prevalencia en Test: {prevalence:.6f} ({prevalence*100:.4f}%)")
    print(f"[INFO] Positivos en Test: {int(np.sum(y_test))} | Total: {len(y_test)}")

    # 7) Umbral óptimo (elige beta=1 para F1 o beta=2 para priorizar Recall)
    # Recomendación en eventos raros: beta=2
    beta = 1.0
    thr, best_fbeta = find_optimal_threshold_fbeta(y_test, y_probs, beta=beta)
    print(f"[EVAL] Mejor F{beta:.0f}-Score: {best_fbeta:.6f} en Umbral: {thr:.4f}")

    y_pred = (y_probs >= thr).astype(int)

    # 8) Reporte
    print("\n" + "=" * 60)
    print(f"RESULTADOS (Umbral = {thr:.4f} | F{beta:.0f} = {best_fbeta:.6f})")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["No Sismo", "Sismo"], digits=4))

    # Matriz para métricas clave
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision_pos = tp / (tp + fp + 1e-12)
    recall_pos = tp / (tp + fn + 1e-12)
    f1 = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + 1e-12)

    print(f"[INFO] TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print(f"[INFO] Precision(+)={precision_pos:.6f} | Recall(+)={recall_pos:.6f} | F1={f1:.6f}")

    # 9) Gráficos “publicables”
    print("[GRAPH] Generando gráficos publicables...")
    roc_auc = plot_roc_publicable(y_test, y_probs, thr, ARTIFACTS_DIR, filename="roc_test.png")
    ap, base, pr_ymax = plot_pr_publicable(y_test, y_probs, ARTIFACTS_DIR, filename="pr_test.png")
    plot_pr_log_publicable(y_test, y_probs, ARTIFACTS_DIR, filename="pr_test_log.png")
    plot_lift_curve(y_test, y_probs, ARTIFACTS_DIR, filename="lift_test.png")
    _tn, _fp, _fn, _tp = plot_confusion_counts_and_norm(
        y_test, y_pred, ARTIFACTS_DIR,
        fn_counts="cm_test_conteos.png",
        fn_norm="cm_test_normalizada.png"
    )

    print("\n" + "=" * 60)
    print("RESUMEN PARA TESIS")
    print("=" * 60)
    print(f"AUC (ROC) = {roc_auc:.4f}")
    print(f"AP (PR)   = {ap:.4f} | Base (prevalencia) = {base:.6f}")
    print(f"Umbral    = {thr:.4f}")
    print(f"CM        = TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Gráficos en: {ARTIFACTS_DIR}")
    print("Archivos recomendados para pegar en tesis:")
    print(" - roc_test.png")
    print(" - cm_test_conteos.png")
    print(" - cm_test_normalizada.png")
    print(" - pr_test_log.png  (o pr_test.png si tu eje lineal ya se aprecia)")
    print(" - lift_test.png    (muy recomendable por interpretabilidad)")
    print("=" * 60)

if __name__ == "__main__":
    main()
