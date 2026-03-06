# =============================================================================
# evaluate_v36_completo.py
# =============================================================================
# Evaluación completa de V3.6 según recomendaciones del asesor:
#
#   MÉTRICAS:
#     - Recall por EVENTO   : detectar cada sismo individual (métrica estricta)
#     - Recall por CELDA    : detectar la zona activa (coherente con el target)
#
#   TOLERANCIAS:
#     -   0 km → celda exacta (más estricto)
#     -  50 km → PRINCIPAL (≈2 celdas de 22km, recomendada por asesor)
#     - 100 km → sobreestimación, solo sensibilidad
#
#   K (Top-K):
#     -  25 → actual
#     -  50 → análisis
#     - 100 → análisis
#
# Lee predicciones ya guardadas en BD (prediction_topk + prediction_cell_prob).
# NO requiere reentrenar ni volver a correr el backtest.
#
# USO:
#   docker cp evaluate_v36_completo.py \
#       quake_sched:/app/scheduler/jobs/evaluate_v36_completo.py
#   docker compose exec scheduler \
#       python -m scheduler.jobs.evaluate_v36_completo
# =============================================================================

import os
import math
import datetime as dt
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dataclasses import dataclass
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class Config:
    db_host:     str = os.getenv("DB_HOST",     "db")
    db_port:     int = int(os.getenv("DB_PORT", "5432"))
    db_name:     str = os.getenv("DB_NAME",     "quake")
    db_user:     str = os.getenv("DB_USER",     "quake")
    db_password: str = os.getenv("DB_PASSWORD", "changeme")

# Parámetros de evaluación según asesor
TOLERANCIAS_KM  = [0, 50, 100]       # 50km = principal
TOP_K_LIST      = [25, 50, 100]      # 25 = actual
UMBRAL_OFICIAL  = 4.0
UMBRAL_PARCIAL  = 3.0


def get_engine(cfg: Config):
    return create_engine(
        f"postgresql+psycopg2://{cfg.db_user}:{cfg.db_password}"
        f"@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
    )


# =============================================================================
# UTILIDADES
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dLon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def dist_minima_a_predicciones(sismo_lat, sismo_lon, df_preds):
    """Distancia mínima entre un sismo y cualquier celda predicha."""
    if df_preds.empty:
        return float("inf"), None
    dists = df_preds.apply(
        lambda r: haversine_km(sismo_lat, sismo_lon, r["lat"], r["lon"]),
        axis=1
    )
    idx_min = dists.idxmin()
    return dists[idx_min], idx_min


# =============================================================================
# CARGA DE DATOS DESDE BD
# =============================================================================

def cargar_runs(engine) -> pd.DataFrame:
    """Todas las corridas guardadas con sus ventanas temporales."""
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT DISTINCT
                pt.run_id,
                MIN(pt.t_pred_start) AS pred_start,
                MAX(pt.t_pred_end)   AS pred_end,
                pr.model_id
            FROM prediction_topk pt
            JOIN prediction_run  pr USING (run_id)
            GROUP BY pt.run_id, pr.model_id
            ORDER BY pt.run_id
        """), conn)
    return df


def cargar_predicciones_run(engine, run_id: int) -> pd.DataFrame:
    """Todas las predicciones de un run (hasta top-100 para cubrir todos los K)."""
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT rank, lat, lon, prob, cell_id
            FROM prediction_topk
            WHERE run_id = :r
            ORDER BY rank ASC
            LIMIT 100
        """), conn, params={"r": run_id})
    return df


def cargar_sismos_ventana(engine, pred_start, pred_end) -> pd.DataFrame:
    """Sismos reales Mw >= UMBRAL_PARCIAL en la ventana."""
    ps = pd.Timestamp(pred_start)
    pe = pd.Timestamp(pred_end)
    if pe.hour == 0 and pe.minute == 0:
        pe = pe.replace(hour=23, minute=59, second=59)

    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT id, event_time_utc, lat, lon, magnitude, place
            FROM events_clean
            WHERE event_time_utc BETWEEN :s AND :e
              AND magnitude >= :m
        """), conn, params={"s": ps, "e": pe, "m": UMBRAL_PARCIAL})
    return df


# =============================================================================
# MÉTRICAS POR EVENTO (estricta — métrica actual)
# =============================================================================

def recall_por_evento(df_sismos_oficiales: pd.DataFrame,
                      df_preds: pd.DataFrame,
                      topk: int,
                      tolerancia_km: float) -> dict:
    """
    Métrica estricta: cada sismo individual cuenta.
    Un sismo se detecta si alguna celda del top-K está a <= tolerancia_km.
    Matching 1-a-1: cada celda solo puede detectar un sismo.
    """
    preds_k = df_preds[df_preds["rank"] <= topk].copy()
    sismos  = df_sismos_oficiales.copy()

    if sismos.empty:
        return {"hits": 0, "total": 0, "recall": 0.0, "precision": 0.0}

    # Para 0 km: usamos 11km (radio de centroide de celda 0.2° ≈ 22km)
    if tolerancia_km == 0:
        tolerancia_km = 11.0

    used_pred_idxs = set()
    hits = 0

    for _, sismo in sismos.iterrows():
        best_dist = float("inf")
        best_idx  = None
        for idx, pred in preds_k.iterrows():
            if idx in used_pred_idxs:
                continue
            d = haversine_km(sismo["lat"], sismo["lon"], pred["lat"], pred["lon"])
            if d <= tolerancia_km and d < best_dist:
                best_dist = d
                best_idx  = idx
        if best_idx is not None:
            used_pred_idxs.add(best_idx)
            hits += 1

    total = len(sismos)
    return {
        "hits"     : hits,
        "total"    : total,
        "recall"   : hits / total * 100 if total else 0.0,
        "precision": hits / topk  * 100 if topk  else 0.0,
    }


# =============================================================================
# MÉTRICAS POR CELDA (coherente con el target — recomendada por asesor)
# =============================================================================

def _clustering_sismos(lats: np.ndarray, lons: np.ndarray,
                       resolucion_km: float = 22.0) -> np.ndarray:
    """
    Agrupa sismos en zonas activas usando Union-Find (más robusto que loop manual).
    Dos sismos se unen si su distancia es <= resolucion_km.
    Retorna array de cluster_id (enteros >= 0) de longitud N.
    Garantizado: nunca retorna -1, siempre N clusters >= 1.
    """
    n = len(lats)
    parent = np.arange(n, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Unir sismos dentro de resolucion_km
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            if d <= resolucion_km:
                union(i, j)

    # Normalizar: reasignar IDs consecutivos 0, 1, 2...
    roots   = {find(i) for i in range(n)}
    id_map  = {r: idx for idx, r in enumerate(sorted(roots))}
    labels  = np.array([id_map[find(i)] for i in range(n)], dtype=int)
    return labels


def recall_por_celda(df_sismos_oficiales: pd.DataFrame,
                     df_preds: pd.DataFrame,
                     topk: int,
                     tolerancia_km: float) -> dict:
    """
    Métrica coherente con el target de entrenamiento.

    Lógica:
      1. Agrupa sismos en ZONAS ACTIVAS usando Union-Find con resolución 22km.
         Dos sismos en la misma zona solo cuentan como una zona activa.
      2. Una zona se detecta si el top-K incluye alguna celda a <= tolerancia_km
         del centroide de esa zona.
      3. Recall_celda = zonas_detectadas / zonas_activas_totales.

    Ventaja vs recall por evento: 10 sismos en la misma zona = 1 zona activa,
    no 10 penalizaciones independientes. Alineado con el target binario del modelo.
    """
    preds_k = df_preds[df_preds["rank"] <= topk].reset_index(drop=True).copy()
    sismos  = df_sismos_oficiales.reset_index(drop=True).copy()

    VACIO = {"celdas_activas": 0, "celdas_detectadas": 0,
             "recall": 0.0, "precision": 0.0}

    if sismos.empty:
        return VACIO

    # Tolerancia 0km → usar radio del centroide de celda (~11km)
    tol = 11.0 if tolerancia_km == 0 else tolerancia_km

    # ── Paso 1: clustering con Union-Find ────────────────────────────────────
    lats   = sismos["lat"].values.astype(float)
    lons   = sismos["lon"].values.astype(float)
    labels = _clustering_sismos(lats, lons, resolucion_km=22.0)
    sismos = sismos.copy()
    sismos["_zona"] = labels

    # Centroide de cada zona activa
    zonas = (
        sismos.groupby("_zona", sort=False)
        .agg(lat_c=("lat", "mean"), lon_c=("lon", "mean"))
        .reset_index(drop=True)
    )

    n_activas = len(zonas)
    if n_activas == 0:
        # Nunca debería ocurrir con sismos no vacíos
        return VACIO

    if preds_k.empty:
        return {"celdas_activas": n_activas, "celdas_detectadas": 0,
                "recall": 0.0, "precision": 0.0}

    pred_lats = preds_k["lat"].values.astype(float)
    pred_lons = preds_k["lon"].values.astype(float)

    # ── Paso 2: detectar zonas usando vectorización numpy ────────────────────
    detectadas = 0
    for _, zona in zonas.iterrows():
        # Calcular distancia a todas las predicciones a la vez
        dists = np.array([
            haversine_km(zona["lat_c"], zona["lon_c"], pl, plo)
            for pl, plo in zip(pred_lats, pred_lons)
        ])
        if np.any(dists <= tol):
            detectadas += 1

    recall    = detectadas / n_activas * 100
    precision = detectadas / topk      * 100

    return {
        "celdas_activas"   : n_activas,
        "celdas_detectadas": detectadas,
        "recall"           : recall,
        "precision"        : precision,
    }


# =============================================================================
# EVALUACIÓN COMPLETA DE UN RUN
# =============================================================================

def evaluar_run(engine, run_id: int, pred_start, pred_end) -> dict:
    """
    Calcula todas las combinaciones (métrica × tolerancia × K) para un run.
    Retorna dict con todos los resultados.
    """
    df_preds = cargar_predicciones_run(engine, run_id)
    df_todos = cargar_sismos_ventana(engine, pred_start, pred_end)

    df_oficiales = df_todos[df_todos["magnitude"] >= UMBRAL_OFICIAL].copy()
    df_parciales = df_todos[
        (df_todos["magnitude"] >= UMBRAL_PARCIAL) &
        (df_todos["magnitude"] <  UMBRAL_OFICIAL)
    ].copy()

    resultado = {
        "run_id"         : run_id,
        "pred_start"     : pd.Timestamp(pred_start).date(),
        "pred_end"       : pd.Timestamp(pred_end).date(),
        "dias"           : (pd.Timestamp(pred_end).date()
                            - pd.Timestamp(pred_start).date()).days + 1,
        "n_sismos_mw4"   : len(df_oficiales),
        "n_sismos_mw3"   : len(df_parciales),
        "n_predicciones" : len(df_preds),
    }

    # ── Calcular celdas activas una sola vez por tolerancia (no depende de K) ─
    # IMPORTANTE: cel_activas es propiedad del ground-truth, no de las preds.
    # Se calcula siempre aunque no haya suficientes predicciones para ese K.
    celdas_activas_por_tol = {}
    for tol in TOLERANCIAS_KM:
        cel_base = recall_por_celda(df_oficiales, pd.DataFrame(
            columns=["rank","lat","lon","prob","cell_id"]), 0, tol)
        celdas_activas_por_tol[tol] = cel_base["celdas_activas"]

    # ── Calcular todas las combinaciones ─────────────────────────────────────
    for tol in TOLERANCIAS_KM:
        for k in TOP_K_LIST:
            # Siempre guardar celdas_activas (ground truth, independiente de K)
            resultado[f"cel_activas_k{k}_t{tol}"] = celdas_activas_por_tol[tol]

            if k > len(df_preds):
                # Sin suficientes predicciones: hits=0, resto calculado
                resultado[f"ev_hits_k{k}_t{tol}"]  = 0
                resultado[f"ev_recall_k{k}_t{tol}"] = 0.0
                resultado[f"ev_prec_k{k}_t{tol}"]   = 0.0
                resultado[f"cel_det_k{k}_t{tol}"]   = 0
                resultado[f"cel_recall_k{k}_t{tol}"]= 0.0
                resultado[f"cel_prec_k{k}_t{tol}"]  = 0.0
                continue

            # Métrica por evento
            ev = recall_por_evento(df_oficiales, df_preds, k, tol)
            resultado[f"ev_hits_k{k}_t{tol}"]   = ev["hits"]
            resultado[f"ev_recall_k{k}_t{tol}"]  = round(ev["recall"],  2)
            resultado[f"ev_prec_k{k}_t{tol}"]    = round(ev["precision"], 2)

            # Métrica por celda
            cel = recall_por_celda(df_oficiales, df_preds, k, tol)
            resultado[f"cel_det_k{k}_t{tol}"]   = cel["celdas_detectadas"]
            resultado[f"cel_recall_k{k}_t{tol}"] = round(cel["recall"],    2)
            resultado[f"cel_prec_k{k}_t{tol}"]   = round(cel["precision"],  2)

    return resultado


# =============================================================================
# IMPRESIÓN DE RESULTADOS
# =============================================================================

def imprimir_tabla(resultados: list, metrica: str, tol: int, k: int,
                   campo_recall: str, campo_hits: str, campo_total: str):
    """Imprime tabla semanal para una combinación métrica/tolerancia/K."""
    total_hits = total_sis = 0
    filas = []

    for r in resultados:
        hits  = r.get(campo_hits,  0)
        total = r.get(campo_total, 0)
        rec   = r.get(campo_recall, 0.0)
        filas.append((r["pred_start"], r["pred_end"],
                      r["dias"], total, hits, rec))
        total_hits += hits
        total_sis  += total

    recall_g = total_hits / total_sis * 100 if total_sis else 0.0
    prec_g   = total_hits / (len(resultados) * k) * 100 if resultados else 0.0

    print(f"\n  [{metrica.upper()}] Tolerancia={tol}km | Top-{k}")
    print(f"  {'Ventana':<24} {'Días':>4} {'Total':>6} "
          f"{'Hits':>5} {'Recall':>8}")
    print(f"  {'─'*55}")
    for ps, pe, dias, total, hits, rec in filas:
        bar = "█" * int(rec / 5)
        print(f"  {str(ps)+'→'+str(pe):<24} {dias:>4} {total:>6} "
              f"{hits:>5} {rec:>7.1f}%  {bar}")
    print(f"  {'─'*55}")
    print(f"  {'TOTAL':<24} {'':>4} {total_sis:>6} "
          f"{total_hits:>5} {recall_g:>7.1f}%")
    return recall_g, prec_g


def imprimir_resumen_comparativo(resultados: list):
    """Tabla resumen con todas las combinaciones para comparación rápida."""
    print("\n" + "=" * 75)
    print("  TABLA COMPARATIVA — TODAS LAS COMBINACIONES")
    print("  (Recall global acumulado sobre las 9 semanas)")
    print("=" * 75)
    print(f"  {'':20} {'Top-25':>10} {'Top-50':>10} {'Top-100':>10}")
    print(f"  {'─'*52}")

    for tol in TOLERANCIAS_KM:
        marker = " ← PRINCIPAL" if tol == 50 else ""
        # Por evento
        row_ev  = f"  Recall/Evento  @{tol:>3}km{marker}"
        # Por celda
        row_cel = f"  Recall/Celda   @{tol:>3}km{marker}"

        vals_ev  = []
        vals_cel = []
        for k in TOP_K_LIST:
            th_ev  = sum(r.get(f"ev_hits_k{k}_t{tol}",  0) for r in resultados)
            ts_ev  = sum(r.get("n_sismos_mw4", 0) for r in resultados)
            th_cel = sum(r.get(f"cel_det_k{k}_t{tol}",  0) for r in resultados)
            ts_cel = sum(r.get(f"cel_activas_k{k}_t{tol}", 0)
                         for r in resultados if f"cel_activas_k{k}_t{tol}" in r)
            vals_ev.append(f"{th_ev/ts_ev*100:>8.1f}%" if ts_ev else "     N/A")
            vals_cel.append(f"{th_cel/ts_cel*100:>8.1f}%" if ts_cel else "     N/A")

        print(f"{row_ev:<40} {'':>2}" + "  ".join(vals_ev))
        print(f"{row_cel:<40} {'':>2}" + "  ".join(vals_cel))
        print(f"  {'·'*52}")

    print("=" * 75)


def imprimir_detalle_principal(resultados: list):
    """Detalle semanal de la métrica principal recomendada por el asesor."""
    print("\n" + "=" * 65)
    print("  DETALLE SEMANAL — MÉTRICA PRINCIPAL")
    print("  Recall por CELDA | Tolerancia=50km | Top-25")
    print("  (Coherente con target de entrenamiento)")
    print("=" * 65)

    total_cel_det = total_cel_act = 0
    total_ev_hits = total_ev_sis  = 0

    print(f"  {'Ventana':<22} {'Días':>4} {'SisM4':>5} "
          f"{'ZonasA':>6} {'ZonasD':>6} "
          f"{'R/Celda':>8} {'R/Evento':>9}")
    print(f"  {'─'*65}")

    for r in resultados:
        cel_act = r.get("cel_activas_k25_t50",  0)
        cel_det = r.get("cel_det_k25_t50",      0)
        rc      = r.get("cel_recall_k25_t50",   0.0)
        re      = r.get("ev_recall_k25_t50",    0.0)
        sis     = r.get("n_sismos_mw4",         0)

        total_cel_det += cel_det
        total_cel_act += cel_act
        total_ev_hits += r.get("ev_hits_k25_t50", 0)
        total_ev_sis  += sis

        bar = "█" * int(rc / 5)
        ventana = f"{r['pred_start']}→{r['pred_end']}"
        print(f"  {ventana:<22} {r['dias']:>4} {sis:>5} "
              f"{cel_act:>6} {cel_det:>6} "
              f"{rc:>7.1f}% {re:>8.1f}%  {bar}")

    print(f"  {'─'*65}")
    rc_g = total_cel_det / total_cel_act * 100 if total_cel_act else 0.0
    re_g = total_ev_hits / total_ev_sis   * 100 if total_ev_sis  else 0.0
    print(f"  {'TOTAL ACUMULADO':<22} {'':>4} {total_ev_sis:>5} "
          f"{total_cel_act:>6} {total_cel_det:>6} "
          f"{rc_g:>7.1f}% {re_g:>8.1f}%")

    # Veredicto
    print(f"\n  {'='*65}")
    print(f"  VEREDICTO V3.6")
    print(f"  {'='*65}")
    print(f"  Recall por evento (estricto, @50km, K=25)  : {re_g:.1f}%")
    print(f"  Recall por celda  (principal, @50km, K=25) : {rc_g:.1f}%")

    # Baseline aleatorio
    total_cells   = 9265  # celdas totales del modelo
    rand_ev_25    = 25 / total_cells * 100
    rand_cel_25   = 25 / total_cells * 100
    ratio_ev      = re_g / rand_ev_25  if rand_ev_25  else 0
    ratio_cel     = rc_g / rand_cel_25 if rand_cel_25 else 0

    print(f"\n  Baseline aleatorio (25/9265 celdas)       : {rand_ev_25:.2f}%")
    print(f"  Ratio mejora vs aleatorio (por evento)    : {ratio_ev:.0f}x")
    print(f"  Ratio mejora vs aleatorio (por celda)     : {ratio_cel:.0f}x")

    if rc_g >= 70:
        print(f"\n  ✅ Recall por celda ≥ 70% — objetivo superado")
    elif rc_g >= 50:
        print(f"\n  ✅ Recall por celda ≥ 50% — resultado sólido")
    else:
        print(f"\n  ⚠️  Recall por celda < 50% — revisar modelo")

    print(f"  {'='*65}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg    = Config()
    engine = get_engine(cfg)

    print("=" * 65)
    print("EVALUACIÓN COMPLETA V3.6 — METODOLOGÍA DEL ASESOR")
    print("=" * 65)
    print(f"Tolerancias : {TOLERANCIAS_KM} km  (principal=50km)")
    print(f"Top-K       : {TOP_K_LIST}")
    print(f"Métricas    : por evento (estricta) + por celda (principal)")

    # ── Cargar corridas ───────────────────────────────────────────────────────
    df_runs = cargar_runs(engine)

    if df_runs.empty:
        print("\n[ERROR] No hay corridas en prediction_topk.")
        print("Ejecuta primero: python -m scheduler.jobs.backtest_2026")
        return

    print(f"\nCorridas encontradas: {len(df_runs)}")
    print(f"{'Run':>5}  {'Pred inicio':12} {'Pred fin':12}")
    print("─" * 35)
    for _, r in df_runs.iterrows():
        print(f"{int(r['run_id']):>5}  "
              f"{str(pd.Timestamp(r['pred_start']).date()):12} "
              f"{str(pd.Timestamp(r['pred_end']).date()):12}")

    # ── Evaluar cada run ──────────────────────────────────────────────────────
    print("\n[EVAL] Calculando métricas para cada ventana...")
    resultados = []

    for _, run in df_runs.iterrows():
        r = evaluar_run(
            engine,
            int(run["run_id"]),
            run["pred_start"],
            run["pred_end"],
        )
        resultados.append(r)
        print(f"  ✓ Run {int(run['run_id']):>3} | "
              f"{r['pred_start']} → {r['pred_end']} | "
              f"Mw≥4: {r['n_sismos_mw4']:>2} | "
              f"Celda@50km/K25: {r.get('cel_recall_k25_t50', 0):.1f}% | "
              f"Evento@50km/K25: {r.get('ev_recall_k25_t50', 0):.1f}%")

    # ── Tabla principal (métrica del asesor) ──────────────────────────────────
    imprimir_detalle_principal(resultados)

    # ── Tabla comparativa completa ────────────────────────────────────────────
    imprimir_resumen_comparativo(resultados)

    # ── Detalle por tolerancia y K (por celda, métrica principal) ─────────────
    print("\n" + "=" * 65)
    print("  DETALLE RECALL POR CELDA — SENSIBILIDAD A K Y TOLERANCIA")
    print("=" * 65)
    for tol in TOLERANCIAS_KM:
        for k in TOP_K_LIST:
            campo_rec   = f"cel_recall_k{k}_t{tol}"
            campo_det   = f"cel_det_k{k}_t{tol}"
            campo_activ = f"cel_activas_k{k}_t{tol}"
            if campo_rec not in resultados[0]:
                continue
            th = sum(r.get(campo_det,   0) for r in resultados)
            ts = sum(r.get(campo_activ, 0) for r in resultados)
            rg = th / ts * 100 if ts else 0.0
            marker = "  ← PRINCIPAL" if tol == 50 and k == 25 else ""
            print(f"  Celda @{tol:>3}km | Top-{k:<3} : "
                  f"{th:>3}/{ts:<3} zonas = {rg:>5.1f}%{marker}")

    print("\n" + "=" * 65)
    print("  DETALLE RECALL POR EVENTO — SENSIBILIDAD A K Y TOLERANCIA")
    print("=" * 65)
    for tol in TOLERANCIAS_KM:
        for k in TOP_K_LIST:
            campo_rec   = f"ev_recall_k{k}_t{tol}"
            campo_hits  = f"ev_hits_k{k}_t{tol}"
            if campo_rec not in resultados[0]:
                continue
            th  = sum(r.get(campo_hits,       0) for r in resultados)
            ts  = sum(r.get("n_sismos_mw4",   0) for r in resultados)
            rg  = th / ts * 100 if ts else 0.0
            marker = "  ← estricta actual" if tol == 100 and k == 25 else ""
            print(f"  Evento @{tol:>3}km | Top-{k:<3} : "
                  f"{th:>3}/{ts:<3} sismos = {rg:>5.1f}%{marker}")

    print("\n[OK] Evaluación completa finalizada.")


if __name__ == "__main__":
    main()