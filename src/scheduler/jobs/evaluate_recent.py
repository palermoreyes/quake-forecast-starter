# evaluate_recent.py

import pandas as pd
import math
import json
from sqlalchemy import text
from .common import get_engine, log

# =============================================================================
# CONFIGURACIÓN (PRODUCCIÓN: FIJO)
# =============================================================================

TOLERANCIA_KM = 50
EVALUAR_TOP_K = 25

# Umbrales científicos
UMBRAL_OFICIAL = 4.0   # Acierto completo (métrica principal oficial)
UMBRAL_PARCIAL = 3.0   # Activación sub-umbral (complementario)

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
    Matching con dos pools separados:
      A) df_full    : Mw ≥ UMBRAL_OFICIAL  → full_hit
      B) df_partial : UMBRAL_PARCIAL ≤ Mw < UMBRAL_OFICIAL → partial_hit
    1) Siempre intenta full primero.
    2) Solo si no hay full disponible, intenta partial.
    3) Dentro del pool: menor distancia.
    4) 1-a-1 por pool (cada sismo se usa una sola vez por pool).
    """
    full_candidates = []
    for _, sismo in df_full.iterrows():
        if sismo["id"] in used_full_ids:
            continue
        dist = haversine_km(pred_lat, pred_lon, sismo["lat"], sismo["lon"])
        if dist <= TOLERANCIA_KM:
            full_candidates.append((dist, sismo))

    if full_candidates:
        full_candidates.sort(key=lambda x: x[0])
        best_dist, best_sismo = full_candidates[0]
        return best_sismo, best_dist, "full_hit"

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


def table_has_column(conn, table_name: str, column_name: str) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = :t
          AND column_name = :c
        LIMIT 1
    """)
    return conn.execute(q, {"t": table_name, "c": column_name}).first() is not None


# =============================================================================
# PROCESO PRINCIPAL
# =============================================================================

def main():
    log("=== INICIANDO VALIDACIÓN (PRODUCCIÓN: TOP-25, 50km) ===")
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
        # 2. VENTANA TEMPORAL (se toma DESDE prediction_topk para alinear 100%)
        # ---------------------------------------------------------------------
        win = conn.execute(
            text("""
                SELECT
                  MIN(t_pred_start) AS t_start,
                  MAX(t_pred_end)   AS t_end
                FROM prediction_topk
                WHERE run_id = :r
            """),
            {"r": run_id}
        ).mappings().first()

        if not win or win["t_start"] is None or win["t_end"] is None:
            # Fallback: 7 días desde input_max_time (pero preferimos topk siempre)
            pred_start = pd.to_datetime(run["input_max_time"]) + pd.Timedelta(days=1)
            pred_end_excl = pred_start + pd.Timedelta(days=7)  # [start, end)
            pred_end_incl = pred_end_excl - pd.Timedelta(seconds=1)
            log("⚠️ No se encontró ventana en prediction_topk. Usando fallback 7 días desde input_max_time.")
        else:
            pred_start = pd.to_datetime(win["t_start"])
            pred_end_incl = pd.to_datetime(win["t_end"])
            pred_end_excl = pred_end_incl + pd.Timedelta(seconds=1)

        log(f"Evaluando Run #{run_id} ({pred_start} → {pred_end_incl}) | K={EVALUAR_TOP_K} | tol={TOLERANCIA_KM}km")

        # ---------------------------------------------------------------------
        # 3. CARGAR PREDICCIONES TOP-K
        # ---------------------------------------------------------------------
        df_preds = pd.read_sql(
            text("""
                SELECT rank, lat, lon, place, cell_id
                FROM prediction_topk
                WHERE run_id = :r
                ORDER BY rank ASC
                LIMIT :k
            """),
            conn,
            params={"r": run_id, "k": EVALUAR_TOP_K}
        )

        total_alertas = int(len(df_preds))
        if total_alertas == 0:
            log("No hay filas en prediction_topk para esta corrida. Abortando evaluación.")
            return

        # ---------------------------------------------------------------------
        # 4. CARGAR SISMOS REALES (≥ UMBRAL_PARCIAL) EN LA VENTANA EXACTA
        #     Ventana: [pred_start, pred_end_excl)  => 7 días exactos
        # ---------------------------------------------------------------------
        df_real = pd.read_sql(
            text("""
                SELECT id, event_time_utc, lat, lon, magnitude, place
                FROM events_clean
                WHERE event_time_utc >= :s
                  AND event_time_utc <  :e
                  AND magnitude >= :min_mag
            """),
            conn,
            params={"s": pred_start, "e": pred_end_excl, "min_mag": UMBRAL_PARCIAL}
        )

        # Pools
        df_full = df_real[df_real["magnitude"] >= UMBRAL_OFICIAL].copy()
        df_partial = df_real[
            (df_real["magnitude"] >= UMBRAL_PARCIAL) &
            (df_real["magnitude"] < UMBRAL_OFICIAL)
        ].copy()

        total_sismos_oficiales = int(len(df_full))

        # ---------------------------------------------------------------------
        # 5. MATCHING ESTRICTO POR EVENTO (1-a-1) — como métrica exigente
        # ---------------------------------------------------------------------
        log(f"[MATCH] Pool full    : {len(df_full)} sismos Mw≥{UMBRAL_OFICIAL}")
        log(f"[MATCH] Pool partial : {len(df_partial)} sismos {UMBRAL_PARCIAL}≤Mw<{UMBRAL_OFICIAL}")
        log(f"[MATCH] Predicciones : {total_alertas} (top-{EVALUAR_TOP_K})")

        full_hits = 0
        partial_hits = 0

        detalle_full = []
        detalle_partial = []

        used_full_ids = set()
        used_partial_ids = set()

        trace_rows = []

        for _, pred in df_preds.iterrows():
            best_sismo, best_dist, hit_type = find_best_match(
                pred["lat"], pred["lon"],
                df_full, df_partial,
                used_full_ids, used_partial_ids
            )

            if best_sismo is not None:
                if hit_type == "full_hit":
                    used_full_ids.add(best_sismo["id"])
                    log(
                        f"  [FULL]    rank={int(pred['rank']):>2} → "
                        f"Mw{float(best_sismo['magnitude']):.1f} "
                        f"'{best_sismo['place']}' @ {best_dist:.1f} km"
                    )
                elif hit_type == "partial_hit":
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
                    "rank": int(pred["rank"]),
                    "evento": best_sismo["place"],
                    "magnitud": float(best_sismo["magnitude"]),
                    "dist_km": round(best_dist, 2)
                })
            elif hit_type == "partial_hit":
                partial_hits += 1
                detalle_partial.append({
                    "rank": int(pred["rank"]),
                    "evento": best_sismo["place"],
                    "magnitud": float(best_sismo["magnitude"]),
                    "dist_km": round(best_dist, 2)
                })

            # Trace
            trace_rows.append({
                "run_id": run_id,
                "rank": int(pred["rank"]),
                "lat": float(pred["lat"]),
                "lon": float(pred["lon"]),
                "place": pred["place"],
                "ws": pred_start.date(),
                "we": pred_end_incl.date(),
                "matched": hit_type != "no_hit",
                "eid": int(best_sismo["id"]) if best_sismo is not None else None,
                "etime": best_sismo["event_time_utc"] if best_sismo is not None else None,
                "mag": float(best_sismo["magnitude"]) if best_sismo is not None else None,
                "eplace": best_sismo["place"] if best_sismo is not None else None,
                "dist": round(best_dist, 2) if best_dist is not None else None,
                "tol": TOLERANCIA_KM,
                "hit_type": hit_type,
            })

        # ---------------------------------------------------------------------
        # 5B. MÉTRICA COHERENTE POR CELDA–VENTANA (para tesis / coherencia)
        #     Definición: una celda real "activa" si ocurre ≥1 evento Mw≥4.0 en esa celda durante la ventana.
        #     Acierto si esa celda activa está dentro del Top-K (match por cell_id exacto).
        # ---------------------------------------------------------------------
        # 1) celdas reales activas (Mw≥4.0) durante ventana:
        df_real_cells = pd.read_sql(
            text("""
                SELECT DISTINCT c.cell_id
                FROM prediction_cells c
                JOIN events_clean e
                  ON ST_Intersects(c.geom, e.geom)
                WHERE e.event_time_utc >= :s
                  AND e.event_time_utc <  :e
                  AND e.magnitude >= :min_mag
            """),
            conn,
            params={"s": pred_start, "e": pred_end_excl, "min_mag": UMBRAL_OFICIAL}
        )

        real_cells = set(int(x) for x in df_real_cells["cell_id"].tolist())
        pred_cells = set(int(x) for x in df_preds["cell_id"].tolist())

        hit_real_cells = real_cells.intersection(pred_cells)

        real_cells_n = int(len(real_cells))
        hit_real_cells_n = int(len(hit_real_cells))
        hit_pred_cells_n = int(len(pred_cells.intersection(real_cells)))

        cell_recall_pct = (hit_real_cells_n / real_cells_n * 100.0) if real_cells_n else 0.0
        cell_precision_pct = (hit_pred_cells_n / total_alertas * 100.0) if total_alertas else 0.0

        log(
            f"[CELL] real_cells={real_cells_n} | hit_real_cells={hit_real_cells_n} | "
            f"cell_recall={cell_recall_pct:.2f}% | cell_precision={cell_precision_pct:.2f}%"
        )

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
                trace_rows
            )

        # ---------------------------------------------------------------------
        # 7. MÉTRICAS (EVENTO - ESTRICTAS)
        # ---------------------------------------------------------------------
        recall = (full_hits / total_sismos_oficiales * 100.0) if total_sismos_oficiales else 0.0
        precision = (full_hits / total_alertas * 100.0) if total_alertas else 0.0
        f1 = (
            2 * (precision / 100) * (recall / 100)
            / ((precision / 100) + (recall / 100))
            * 100
            if (precision + recall) > 0 else 0.0
        )

        # ---------------------------------------------------------------------
        # 8. GUARDAR RESUMEN (sin romper esquema)
        #     - Mantiene columnas actuales
        #     - Si existen columnas nuevas para celda-ventana, las llena; si no, solo log.
        # ---------------------------------------------------------------------
        has_cell_cols = (
            table_has_column(conn, "validation_realworld", "cell_recall_pct") and
            table_has_column(conn, "validation_realworld", "cell_precision_pct") and
            table_has_column(conn, "validation_realworld", "real_cells") and
            table_has_column(conn, "validation_realworld", "hit_real_cells") and
            table_has_column(conn, "validation_realworld", "hit_pred_cells")
        )

        insert_sql = """
            INSERT INTO validation_realworld (
                run_id, window_start, window_end,
                total_sismos, sismos_detectados, partial_hits,
                recall_pct, precision_pct, f1_pct,
                aciertos_json, partial_hits_json
        """
        if has_cell_cols:
            insert_sql += """,
                real_cells, hit_real_cells, hit_pred_cells,
                cell_recall_pct, cell_precision_pct
            """
        insert_sql += """
            ) VALUES (
                :r, :s, :e,
                :ts, :full, :partial,
                :rec, :prec, :f1,
                :json_full, :json_partial
        """
        if has_cell_cols:
            insert_sql += """,
                :rc, :hrc, :hpc,
                :crec, :cprec
            """
        insert_sql += ")"

        payload = {
            "r": run_id,
            "s": pred_start.date(),
            "e": pred_end_incl.date(),
            "ts": total_sismos_oficiales,
            "full": full_hits,
            "partial": partial_hits,
            "rec": round(recall, 4),
            "prec": round(precision, 4),
            "f1": round(f1, 4),
            "json_full": json.dumps(detalle_full),
            "json_partial": json.dumps(detalle_partial),
        }

        if has_cell_cols:
            payload.update({
                "rc": real_cells_n,
                "hrc": hit_real_cells_n,
                "hpc": hit_pred_cells_n,
                "crec": round(cell_recall_pct, 4),
                "cprec": round(cell_precision_pct, 4),
            })

        with engine.begin() as tx:
            tx.execute(text(insert_sql), payload)

        # ---------------------------------------------------------------------
        # 9. REPORTE
        # ---------------------------------------------------------------------
        no_hits = total_alertas - full_hits - partial_hits

        print("\n=== RESULTADO VALIDACIÓN (PRODUCCIÓN) ===")
        print(f"Ventana evaluada (UTC):     {pred_start} → {pred_end_incl}")
        print(f"K (Top-K):                  {EVALUAR_TOP_K}")
        print(f"Tolerancia:                 {TOLERANCIA_KM} km")
        print(f"─────────────────────────────────────")
        print(f"Sismos reales (Mw≥{UMBRAL_OFICIAL}):   {total_sismos_oficiales}")
        print(f"Sismos sub-umbral ({UMBRAL_PARCIAL}–{UMBRAL_OFICIAL}): {len(df_partial)}")
        print(f"─────────────────────────────────────")
        print(f"Alertas evaluadas:          {total_alertas}")
        print(f"  Aciertos completos:       {full_hits}")
        print(f"  Aciertos parciales:       {partial_hits}")
        print(f"  Sin match:                {no_hits}")
        print(f"─────────────────────────────────────")
        print(f"Recall    (evento, full):   {recall:.1f}%")
        print(f"Precision (evento, full):   {precision:.1f}%")
        print(f"F1-score  (evento, full):   {f1:.1f}%")
        print(f"─────────────────────────────────────")
        print(f"Cell Recall (Mw≥{UMBRAL_OFICIAL}):    {cell_recall_pct:.2f}%  (real_cells={real_cells_n}, hit={hit_real_cells_n})")
        print(f"Cell Precision (Mw≥{UMBRAL_OFICIAL}): {cell_precision_pct:.2f}% (hit_pred_cells={hit_pred_cells_n}/{total_alertas})")
        print("========================================")

        # Sanity-checks
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


if __name__ == "__main__":
    main()

# Marca para validar cambios cargados
