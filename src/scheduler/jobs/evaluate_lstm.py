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
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

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

# Gráficos "publicables"
FIGSIZE    = (6.8, 4.2)   # proporción cómoda para Word / LaTeX
DPI        = 300           # alta resolución para tesis
GRID_ALPHA = 0.15          # rejilla sutil

def _save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()

def _fmt_int(x: int) -> str:
    """Separador de miles legible en tesis (e.g. 1 234 567)."""
    return f"{x:,}".replace(",", "\u202f")   # narrow no-break space


# =========================
# REPRODUCIBILIDAD
# =========================
def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# MODELO ACTIVO
# =========================
def get_active_model_path(cfg) -> str:
    """Obtiene la ruta del modelo ACTIVO desde el registry en BD."""
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
def find_optimal_threshold_fbeta(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    beta: float = 1.0,
) -> tuple[float, float]:
    """
    Selecciona el umbral que maximiza F-beta en la curva PR.

    beta > 1 → prioriza Recall (recomendado para eventos raros como sismos).
    beta = 1 → F1 estándar (igual peso a Precision y Recall).

    Returns
    -------
    best_thresh : float
    best_fbeta  : float
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    # thresholds tiene N-1 elementos; precision/recall tienen N
    precision_t = precision[:-1]
    recall_t    = recall[:-1]

    # Máscara para evitar que argmax elija puntos degenerados
    valid = (precision_t + recall_t) > 1e-6

    beta2 = beta ** 2
    fbeta = np.where(
        valid,
        (1 + beta2) * (precision_t * recall_t) / (beta2 * precision_t + recall_t + 1e-12),
        0.0,
    )

    best_idx    = int(np.argmax(fbeta))
    best_thresh = float(thresholds[best_idx])
    best_fbeta  = float(fbeta[best_idx])
    return best_thresh, best_fbeta


# =========================
# GRÁFICOS PUBLICABLES
# =========================

def plot_roc_publicable(
    y_true:   np.ndarray,
    y_probs:  np.ndarray,
    thr:      float,
    outdir:   str,
    filename: str = "roc_test.png",
) -> float:
    """Curva ROC con punto del umbral óptimo marcado."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Índice del umbral más cercano al óptimo (con clamp por seguridad)
    idx = int(np.argmin(np.abs(thresholds - thr)))
    idx = int(np.clip(idx, 0, len(fpr) - 1))

    plt.figure(figsize=FIGSIZE)
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Azar")
    plt.scatter(fpr[idx], tpr[idx], s=55, zorder=5,
                label=f"Umbral = {thr:.4f}")

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR / Recall)")
    plt.title("Curva ROC (conjunto de prueba)")
    plt.legend(frameon=False, loc="lower right")
    plt.grid(alpha=GRID_ALPHA)
    _save_fig(os.path.join(outdir, filename))

    return float(roc_auc)


def plot_pr_publicable(
    y_true:   np.ndarray,
    y_probs:  np.ndarray,
    outdir:   str,
    filename: str = "pr_test.png",
) -> tuple[float, float, float]:
    """Curva Precisión–Recall (escala lineal) con baseline de prevalencia."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap   = average_precision_score(y_true, y_probs)
    base = float(np.mean(y_true))

    # CORRECCIÓN: techo dinámico que no aplana la curva en desbalance extremo
    p99   = float(np.quantile(precision, 0.995))
    y_max = max(p99 * 1.5, base * 10) * 1.10

    plt.figure(figsize=FIGSIZE)
    plt.step(recall, precision, where="post", linewidth=2,
             label=f"AP = {ap:.4f}")
    plt.fill_between(recall, precision, step="post", alpha=0.12)
    plt.axhline(base, linestyle="--", linewidth=1.5,
                label=f"Base = {base:.6f}")

    plt.xlim(0, 1)
    plt.ylim(0, y_max)
    plt.xlabel("Exhaustividad (Recall)")
    plt.ylabel("Precisión (Precision)")
    plt.title("Curva Precisión–Recall (conjunto de prueba)")
    plt.legend(frameon=False, loc="upper right")
    plt.grid(alpha=GRID_ALPHA)
    _save_fig(os.path.join(outdir, filename))

    return float(ap), float(base), float(y_max)


def plot_pr_log_publicable(
    y_true:   np.ndarray,
    y_probs:  np.ndarray,
    outdir:   str,
    filename: str = "pr_test_log.png",
) -> None:
    """Curva PR en escala log (útil cuando la prevalencia es < 0.1 %)."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap   = average_precision_score(y_true, y_probs)
    base = float(np.mean(y_true))

    p99   = float(np.quantile(precision, 0.995))
    y_min = max(base / 10.0, 1e-6)
    y_max = max(p99, base) * 2.0

    plt.figure(figsize=FIGSIZE)
    plt.step(recall, precision, where="post", linewidth=2,
             label=f"AP = {ap:.4f}")
    plt.axhline(base, linestyle="--", linewidth=1.5,
                label=f"Base = {base:.6f}")

    plt.yscale("log")
    plt.xlim(0, 1)
    plt.ylim(y_min, y_max)
    plt.xlabel("Exhaustividad (Recall)")
    plt.ylabel("Precisión (escala log)")
    plt.title("Curva Precisión–Recall — escala log (conjunto de prueba)")
    plt.legend(frameon=False, loc="lower left")
    plt.grid(alpha=GRID_ALPHA, which="both")
    _save_fig(os.path.join(outdir, filename))


def plot_lift_curve(
    y_true:   np.ndarray,
    y_probs:  np.ndarray,
    outdir:   str,
    filename: str = "lift_test.png",
) -> None:
    """
    Lift = Precision / Prevalencia.
    Muy interpretable en tesis: cuántas veces mejor que selección aleatoria.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    base = float(np.mean(y_true))
    lift = precision / (base + 1e-12)

    l99   = float(np.quantile(lift, 0.995))
    y_max = max(2.0, l99) * 1.10

    plt.figure(figsize=FIGSIZE)
    plt.step(recall, lift, where="post", linewidth=2)
    plt.axhline(1.0, linestyle="--", linewidth=1.5,
                label="Azar (Lift = 1)")

    plt.xlim(0, 1)
    plt.ylim(0, y_max)
    plt.xlabel("Exhaustividad (Recall)")
    plt.ylabel("Lift  (Precisión / Prevalencia)")
    plt.title("Curva de Lift (conjunto de prueba)")
    plt.legend(frameon=False, loc="upper right")
    plt.grid(alpha=GRID_ALPHA)
    _save_fig(os.path.join(outdir, filename))


def plot_confusion_counts_and_norm(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    outdir:    str,
    fn_counts: str = "cm_test_conteos.png",
    fn_norm:   str = "cm_test_normalizada.png",
) -> tuple[int, int, int, int]:
    """Genera las matrices de confusión: conteos y normalizada por fila."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # --- Conteos (colormap Blues para tesis en blanco/negro)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title("Matriz de confusión (conteos)")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Realidad")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Sismo", "Sismo"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Sismo", "Sismo"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, _fmt_int(int(v)), ha="center", va="center", fontsize=12)
    ax.grid(False)
    _save_fig(os.path.join(outdir, fn_counts))

    # --- Normalizada por realidad (filas)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title("Matriz de confusión (normalizada por realidad)")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Realidad")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Sismo", "Sismo"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Sismo", "Sismo"])
    for (i, j), v in np.ndenumerate(cm_norm):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=12)
    ax.grid(False)
    _save_fig(os.path.join(outdir, fn_norm))

    return int(tn), int(fp), int(fn), int(tp)


def plot_calibration_curve(
    y_true:   np.ndarray,
    y_probs:  np.ndarray,
    outdir:   str,
    n_bins:   int = 10,
    filename: str = "calibration_test.png",
) -> float:
    """
    Curva de calibración (reliability diagram).
    Muestra si las probabilidades predichas reflejan las frecuencias reales.
    Brier Score incluido: 0 = perfecto, 0.25 = azar en binario balanceado.
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_probs, n_bins=n_bins, strategy="quantile"
    )
    brier = brier_score_loss(y_true, y_probs)

    plt.figure(figsize=FIGSIZE)
    plt.plot(mean_pred, fraction_pos, marker="o", linewidth=2,
             label=f"Modelo  (Brier = {brier:.6f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5,
             label="Calibración perfecta")

    plt.xlabel("Probabilidad predicha promedio")
    plt.ylabel("Fracción de positivos reales")
    plt.title("Curva de calibración (conjunto de prueba)")
    plt.legend(frameon=False, loc="upper left")
    plt.grid(alpha=GRID_ALPHA)
    _save_fig(os.path.join(outdir, filename))

    return float(brier)


def plot_score_distribution(
    y_true:   np.ndarray,
    y_probs:  np.ndarray,
    thr:      float,
    outdir:   str,
    filename: str = "score_dist_test.png",
) -> None:
    """
    Histograma de las probabilidades predichas, separado por clase.
    Útil para visualizar separabilidad y justificar el umbral elegido.
    """
    probs_neg = y_probs[y_true == 0]
    probs_pos = y_probs[y_true == 1]

    plt.figure(figsize=FIGSIZE)
    bins = np.linspace(0, 1, 60)
    plt.hist(probs_neg, bins=bins, alpha=0.55, density=True,
             label="No Sismo (clase 0)")
    plt.hist(probs_pos, bins=bins, alpha=0.55, density=True,
             label="Sismo (clase 1)")
    plt.axvline(thr, linestyle="--", linewidth=1.8,
                label=f"Umbral = {thr:.4f}")

    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Densidad")
    plt.title("Distribución de scores por clase (conjunto de prueba)")
    plt.legend(frameon=False)
    plt.grid(alpha=GRID_ALPHA)
    _save_fig(os.path.join(outdir, filename))

# =========================
# PERSISTENCIA DE MÉTRICAS
# =========================
class _NumpyEncoder(json.JSONEncoder):
    """Convierte tipos NumPy a tipos nativos Python para que json.dump no falle."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_metrics_json(metrics: dict, outdir: str, filename: str = "metrics.json") -> None:
    """Guarda todas las métricas en JSON para reproducibilidad y tablas en tesis."""
    path = os.path.join(outdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    print(f"[INFO] Métricas guardadas en: {path}")

# =========================
# MAIN
# =========================
def main() -> None:
    print("=== INICIANDO EVALUACIÓN FORMAL (MODELO ACTIVO) ===")

    # Reproducibilidad
    set_seeds(42)

    cfg = Config()

    # 1) Cargar datos
    print("[EVAL] Cargando matriz wide...")
    wide_df = load_wide_matrix(cfg)

    # 2) Splits cronológicos
    train_mask = wide_df.index <= pd.Timestamp(cfg.train_end)
    test_mask  = (
        (wide_df.index > pd.Timestamp(cfg.val_end)) &
        (wide_df.index <= pd.Timestamp(cfg.test_end))
    )

    df_train = wide_df[train_mask]
    df_test  = wide_df[test_mask]

    # 3) Normalización (parámetros sólo de Train → sin data leakage)
    print("[EVAL] Calculando parámetros de normalización (Train)...")
    means, stds = compute_norm_params(df_train)
    # .values convierte pandas Series → numpy array (requerido por make_sequences_fast)
    means_np = means.values.astype('float32')
    stds_np  = stds.values.astype('float32')

    # 4) Secuencias de Test
    print("[EVAL] Generando secuencias del conjunto de prueba...")
    X_test, y_test = make_sequences_fast(
        df_test, cfg, means=means_np, stds=stds_np, is_train=False
    )
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
    n_pos      = int(np.sum(y_test))
    n_total    = len(y_test)
    print(f"[INFO] Prevalencia en Test : {prevalence:.6f} ({prevalence * 100:.4f} %)")
    print(f"[INFO] Positivos / Total   : {n_pos} / {n_total}")

    # 7) Umbral óptimo por F-beta
    #    beta=2 → prioriza Recall (recomendado para detección de sismos raros)
    #    beta=1 → F1 estándar
    beta = 1.0
    thr, best_fbeta = find_optimal_threshold_fbeta(y_test, y_probs, beta=beta)
    print(f"[EVAL] Mejor F{beta:.0f}-Score : {best_fbeta:.6f}  (umbral = {thr:.4f})")

    y_pred = (y_probs >= thr).astype(int)

    # 8) Reporte textual
    print("\n" + "=" * 65)
    print(f"RESULTADOS  (Umbral = {thr:.4f} | F{beta:.0f} = {best_fbeta:.6f})")
    print("=" * 65)
    print(classification_report(
        y_test, y_pred,
        target_names=["No Sismo", "Sismo"],
        digits=4,
    ))

    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision_pos = tp / (tp + fp + 1e-12)
    recall_pos    = tp / (tp + fn + 1e-12)
    f1            = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + 1e-12)

    print(f"[INFO] TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"[INFO] Precision(+) = {precision_pos:.6f} | Recall(+) = {recall_pos:.6f} | F1 = {f1:.6f}")

    # 9) Gráficos publicables
    print("\n[GRAPH] Generando gráficos publicables...")
    roc_auc       = plot_roc_publicable(y_test, y_probs, thr, ARTIFACTS_DIR)
    ap, base, _   = plot_pr_publicable(y_test, y_probs, ARTIFACTS_DIR)
    plot_pr_log_publicable(y_test, y_probs, ARTIFACTS_DIR)
    plot_lift_curve(y_test, y_probs, ARTIFACTS_DIR)
    plot_confusion_counts_and_norm(y_test, y_pred, ARTIFACTS_DIR)
    brier         = plot_calibration_curve(y_test, y_probs, ARTIFACTS_DIR)
    plot_score_distribution(y_test, y_probs, thr, ARTIFACTS_DIR)
    print("[GRAPH] ✓ Todos los gráficos guardados.")

    # 10) Guardar métricas en JSON
    metrics = {
        "model_file"      : os.path.basename(model_path),
        "n_test_samples"  : n_total,
        "n_positives"     : n_pos,
        "prevalence"      : round(prevalence, 8),
        "threshold"       : round(thr, 6),
        f"f{beta:.0f}_score": round(best_fbeta, 6),
        "auc_roc"         : round(roc_auc, 6),
        "average_precision": round(ap, 6),
        "brier_score"     : round(brier, 8),
        "precision_pos"   : round(precision_pos, 6),
        "recall_pos"      : round(recall_pos, 6),
        "f1_pos"          : round(f1, 6),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    save_metrics_json(metrics, ARTIFACTS_DIR)

    # 11) Resumen para tesis
    print("\n" + "=" * 65)
    print("RESUMEN PARA TESIS")
    print("=" * 65)
    print(f"  AUC-ROC            = {roc_auc:.4f}")
    print(f"  AP (Precision-Recall) = {ap:.4f}  (base = {base:.6f})")
    print(f"  Brier Score        = {brier:.6f}")
    print(f"  Umbral óptimo      = {thr:.4f}  (F{beta:.0f} = {best_fbeta:.6f})")
    print(f"  Confusion matrix   : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Precision / Recall / F1 = {precision_pos:.4f} / {recall_pos:.4f} / {f1:.4f}")
    print(f"\nArtifacts en: {ARTIFACTS_DIR}")
    print("Archivos recomendados para la tesis:")
    print("  - roc_test.png             → discriminación global")
    print("  - pr_test_log.png          → rendimiento en clase minoritaria")
    print("  - lift_test.png            → interpretabilidad del modelo")
    print("  - cm_test_conteos.png      → errores absolutos")
    print("  - cm_test_normalizada.png  → tasas de error por clase")
    print("  - calibration_test.png     → confianza en las probabilidades")
    print("  - score_dist_test.png      → separabilidad de clases")
    print("  - metrics.json             → tabla de métricas reproducible")
    print("=" * 65)


if __name__ == "__main__":
    main()
