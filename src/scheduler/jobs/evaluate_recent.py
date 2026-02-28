import pandas as pd
import math
import json
from sqlalchemy import text
from .common import get_engine, log

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA
# =============================================================================

TOLERANCIA_KM  = 100
EVALUAR_TOP_K  = 25

# Umbrales científicos
UMBRAL_OFICIAL = 4.0   # Acierto completo
UMBRAL_PARCIAL = 3.0   # Activación sub-umbral


# =============================================================================
# UTILIDADES
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (
        math.sin(dLat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dLon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def find_best_match(pred_lat, pred_lon, df_full, df_partial,
                    used_full_ids, used_partial_ids):
    """
    Matching con dos pools completamente separados:

      Pool A — df_full   : sismos Mw ≥ UMBRAL_OFICIAL (4.0) → full_hit
      Pool B — df_partial: sismos UMBRAL_PARCIAL ≤ Mw < UMBRAL_OFICIAL → partial_hit

    Reglas:
      1. Se intenta primero el pool A (full). Si hay candidato disponible → full_hit.
      2. Solo si no hay ningún full disponible se busca en pool B → partial_hit.
      3. Dentro de cada pool se elige el de MENOR distancia.
      4. Cada sismo solo puede ser asignado una vez (matching 1-a-1 por pool).

    Esto garantiza que un sismo ≥4.0 NUNCA puede registrarse como partial_hit
    y que un sismo <4.0 NUNCA puede inflar los full_hits.

    Returns
    -------
    (sismo_row, distancia_km, hit_type)  o  (None, None, 'no_hit')
    """
    # --- Intentar full_hit primero ---
    full_candidates = []
    for _, sismo in df_full.iterrows():
        if sismo["id"] in used_full_ids:
            continue
        dist = haversine_km(pred_lat, pred_lon, sismo["lat"], sismo["lon"])
        if dist <= TOLERANCIA_KM:
            full_candidates.append((dist, sismo))

    if full_candidates:
        full_candidates.sort(key=lambda x: x[0])   # menor distancia primero
        best_dist, best_sismo = full_candidates[0]
        return best_sismo, best_dist, "full_hit"

    # --- Solo si no hubo full, intentar partial_hit ---
    partial_candidates = []
    for _, sismo in df_partial.iterrows():
        if sismo["id"] in used_partial_ids:
            continue
        dist = haversine_km(pred_lat, pred_lon, sismo["lat"], sismo["lon"])
        if dist <= TOLERANCIA_KM:
            partial_candidates.append((dist, sismo))

    if partial_candidates:
        partial_candidates.sort(key=lambda x: x[0])
        best_dist, best_sismo = partial_candidates[0]
        return best_sismo, best_dist, "partial_hit"

    return None, None, "no_hit"


# =============================================================================
# PROCESO PRINCIPAL
# =============================================================================

def main():
    log("=== INICIANDO VALIDACIÓN (HISTÓRICO EXTENDIDO) ===")
    engine = get_engine()

    with engine.connect() as conn:

        # ---------------------------------------------------------------------
        # 1. OBTENER ÚLTIMA CORRIDA
        # ---------------------------------------------------------------------
        run = conn.execute(
            text("""
                SELECT run_id, input_max_time
                FROM prediction_run
                ORDER BY run_id DESC
                LIMIT 1
            """)
        ).mappings().first()

        if not run:
            log("No hay predicciones para evaluar.")
            return

        run_id = run["run_id"]

        # Evitar duplicados
        exists = conn.execute(
            text("SELECT 1 FROM validation_realworld WHERE run_id = :r"),
            {"r": run_id}
        ).first()

        if exists:
            log(f"⚠️ Run #{run_id} ya fue evaluado.")
            return

        # ---------------------------------------------------------------------
        # 2. VENTANA TEMPORAL
        # ---------------------------------------------------------------------
        pred_start = pd.to_datetime(run["input_max_time"]) + pd.Timedelta(days=1)
        pred_end   = pred_start + pd.Timedelta(days=7)

        log(f"Evaluando Run #{run_id} ({pred_start.date()} → {pred_end.date()})")

        # ---------------------------------------------------------------------
        # 3. CARGAR PREDICCIONES TOP-K
        # ---------------------------------------------------------------------
        df_preds = pd.read_sql(
            text("""
                SELECT rank, lat, lon, place
                FROM prediction_topk
                WHERE run_id = :r
                ORDER BY rank ASC
                LIMIT :k
            """),
            conn,
            params={"r": run_id, "k": EVALUAR_TOP_K}
        )

        # ---------------------------------------------------------------------
        # 4. CARGAR SISMOS REALES (≥ UMBRAL_PARCIAL)
        # ---------------------------------------------------------------------
        df_real = pd.read_sql(
            text("""
                SELECT id, event_time_utc, lat, lon, magnitude, place
                FROM events_clean
                WHERE event_time_utc BETWEEN :s AND :e
                  AND magnitude >= :min_mag
            """),
            conn,
            params={"s": pred_start, "e": pred_end, "min_mag": UMBRAL_PARCIAL}
        )

        # Separar sismos en dos pools completamente independientes
        # Pool A: sismos oficiales Mw ≥ 4.0
        df_full    = df_real[df_real["magnitude"] >= UMBRAL_OFICIAL].copy()
        # Pool B: sismos sub-umbral  3.0 ≤ Mw < 4.0  (NUNCA incluye ≥4.0)
        df_partial = df_real[
            (df_real["magnitude"] >= UMBRAL_PARCIAL) &
            (df_real["magnitude"] <  UMBRAL_OFICIAL)
        ].copy()

        total_sismos_oficiales = int(len(df_full))
        total_alertas          = int(len(df_preds))

        full_hits    = 0
        partial_hits = 0

        detalle_full    = []
        detalle_partial = []

        # IDs usados por pool separado → un sismo ≥4.0 nunca puede
        # ser reclamado como partial, y viceversa
        used_full_ids    = set()
        used_partial_ids = set()

        # ---------------------------------------------------------------------
        # 5. MATCHING  (1 predicción → máximo 1 sismo, 1 sismo → máximo 1 predicción)
        # ---------------------------------------------------------------------
        log(f"[MATCH] Pool full    : {len(df_full)} sismos Mw≥{UMBRAL_OFICIAL}")
        log(f"[MATCH] Pool partial : {len(df_partial)} sismos {UMBRAL_PARCIAL}≤Mw<{UMBRAL_OFICIAL}")
        log(f"[MATCH] Predicciones : {total_alertas} (top-{EVALUAR_TOP_K})")

        trace_rows = []

        for _, pred in df_preds.iterrows():

            best_sismo, best_dist, hit_type = find_best_match(
                pred["lat"], pred["lon"],
                df_full, df_partial,
                used_full_ids, used_partial_ids
            )

            # Marcar el sismo como usado en su pool correspondiente
            if best_sismo is not None:
                if hit_type == "full_hit":
                    used_full_ids.add(best_sismo["id"])
                    log(
                        f"  [FULL]    rank={int(pred['rank']):>2} → "
                        f"Mw{float(best_sismo['magnitude']):.1f} "
                        f"'{best_sismo['place']}' @ {best_dist:.1f} km"
                    )
                else:
                    used_partial_ids.add(best_sismo["id"])
                    log(
                        f"  [PARTIAL] rank={int(pred['rank']):>2} → "
                        f"Mw{float(best_sismo['magnitude']):.1f} "
                        f"'{best_sismo['place']}' @ {best_dist:.1f} km"
                    )
            else:
                log(f"  [MISS]    rank={int(pred['rank']):>2} → sin sismo en radio {TOLERANCIA_KM} km")

            if hit_type == "full_hit":
                full_hits += 1
                detalle_full.append({
                    "rank"     : int(pred["rank"]),
                    "evento"   : best_sismo["place"],
                    "magnitud" : float(best_sismo["magnitude"]),
                    "dist_km"  : round(best_dist, 2)
                })
            elif hit_type == "partial_hit":
                partial_hits += 1
                detalle_partial.append({
                    "rank"     : int(pred["rank"]),
                    "evento"   : best_sismo["place"],
                    "magnitud" : float(best_sismo["magnitude"]),
                    "dist_km"  : round(best_dist, 2)
                })

            # Acumular para inserción en batch
            trace_rows.append({
                "run_id"  : run_id,
                "rank"    : int(pred["rank"]),
                "lat"     : pred["lat"],
                "lon"     : pred["lon"],
                "place"   : pred["place"],
                "ws"      : pred_start.date(),
                "we"      : pred_end.date(),
                "matched" : hit_type != "no_hit",
                "eid"     : int(best_sismo["id"])       if best_sismo is not None else None,
                "etime"   : best_sismo["event_time_utc"] if best_sismo is not None else None,
                "mag"     : float(best_sismo["magnitude"]) if best_sismo is not None else None,
                "eplace"  : best_sismo["place"]         if best_sismo is not None else None,
                "dist"    : round(best_dist, 2)         if best_dist  is not None else None,
                "tol"     : TOLERANCIA_KM,
                "hit_type": hit_type,
            })

        # ---------------------------------------------------------------------
        # 6. PERSISTIR TRAZAS EN BATCH
        # ---------------------------------------------------------------------
        with engine.begin() as tx:
            tx.execute(
                text("""
                    INSERT INTO prediction_trace (
                        run_id, rank, lat, lon, place,
                        predicted_window_start, predicted_window_end,
                        matched_event, matched_event_id,
                        event_time_utc, event_magnitude, event_place,
                        distance_km, tolerance_km, hit_type
                    ) VALUES (
                        :run_id, :rank, :lat, :lon, :place,
                        :ws, :we,
                        :matched, :eid,
                        :etime, :mag, :eplace,
                        :dist, :tol, :hit_type
                    )
                """),
                trace_rows   # SQLAlchemy ejecuta todos de una sola vez
            )

        # ---------------------------------------------------------------------
        # 7. MÉTRICAS
        # ---------------------------------------------------------------------
        recall = (
            full_hits / total_sismos_oficiales * 100
            if total_sismos_oficiales else 0.0
        )
        precision = (
            full_hits / total_alertas * 100
            if total_alertas else 0.0
        )
        # F1 sobre métricas oficiales (full_hits únicamente)
        f1 = (
            2 * (precision / 100) * (recall / 100)
            / ((precision / 100) + (recall / 100))
            * 100
            if (precision + recall) > 0 else 0.0
        )

        # ---------------------------------------------------------------------
        # 8. GUARDAR RESUMEN
        # ---------------------------------------------------------------------
        with engine.begin() as tx:
            tx.execute(
                text("""
                    INSERT INTO validation_realworld (
                        run_id, window_start, window_end,
                        total_sismos, sismos_detectados, partial_hits,
                        recall_pct, precision_pct, f1_pct,
                        aciertos_json, partial_hits_json
                    ) VALUES (
                        :r, :s, :e,
                        :ts, :full, :partial,
                        :rec, :prec, :f1,
                        :json_full, :json_partial
                    )
                """),
                {
                    "r"           : run_id,
                    "s"           : pred_start.date(),
                    "e"           : pred_end.date(),
                    "ts"          : total_sismos_oficiales,
                    "full"        : full_hits,
                    "partial"     : partial_hits,
                    "rec"         : round(recall, 4),
                    "prec"        : round(precision, 4),
                    "f1"          : round(f1, 4),
                    "json_full"   : json.dumps(detalle_full),
                    "json_partial": json.dumps(detalle_partial),
                }
            )

        # ---------------------------------------------------------------------
        # 9. REPORTE
        # ---------------------------------------------------------------------
        no_hits = total_alertas - full_hits - partial_hits

        print("\n=== RESULTADO VALIDACIÓN REAL-WORLD ===")
        print(f"Ventana evaluada:           {pred_start.date()} → {pred_end.date()}")
        print(f"Sismos reales (Mw≥{UMBRAL_OFICIAL}):   {total_sismos_oficiales}")
        print(f"Sismos sub-umbral ({UMBRAL_PARCIAL}–{UMBRAL_OFICIAL}): {len(df_partial)}")
        print(f"─────────────────────────────────────")
        print(f"Alertas evaluadas:          {total_alertas}")
        print(f"  Aciertos completos:       {full_hits}")
        print(f"  Aciertos parciales:       {partial_hits}")
        print(f"  Sin match:                {no_hits}")
        print(f"─────────────────────────────────────")
        print(f"Recall    (full):           {recall:.1f}%")
        print(f"Precision (full):           {precision:.1f}%")
        print(f"F1-score  (full):           {f1:.1f}%")
        print("========================================")

        # Sanity-checks — imposibles con matching 1-a-1 y pools separados
        if full_hits > total_sismos_oficiales:
            log(
                f"[WARNING] full_hits ({full_hits}) > sismos oficiales "
                f"({total_sismos_oficiales}). Revisar lógica de matching."
            )
        if partial_hits > len(df_partial):
            log(
                f"[WARNING] partial_hits ({partial_hits}) > sismos sub-umbral "
                f"({len(df_partial)}). Revisar lógica de matching."
            )
        if recall > 100.0:
            log("[WARNING] Recall > 100% — imposible con matching 1-a-1. Revisar datos.")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    main()

#Marca para validar cambios cargados