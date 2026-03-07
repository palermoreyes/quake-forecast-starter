# evaluate_recent.py

import pandas as pd
import math
import json
from sqlalchemy import text
from .common import get_engine, log

# =============================================================================
# CONFIGURACIÓN (PRODUCCIÓN: FIJO)
# =============================================================================
# Este script queda productivizado con una sola configuración oficial:
#   - Top-K = 25
#   - Tolerancia espacial = 100 km
#
# No se harán barridos aquí. Si luego quieres probar K o tolerancias distintas
# para análisis de tesis, eso debe ir en otro script separado.
# =============================================================================

TOLERANCIA_KM = 100
EVALUAR_TOP_K = 25

# Umbrales científicos
UMBRAL_OFICIAL = 4.0   # Evento "oficial" para métricas principales
UMBRAL_PARCIAL = 3.0   # Activación sub-umbral (complementario)

# =============================================================================
# UTILIDADES
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en km entre dos puntos."""
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
    Matching estricto 1-a-1 por evento, con dos pools separados:

      A) df_full    : Mw >= UMBRAL_OFICIAL  -> full_hit
      B) df_partial : UMBRAL_PARCIAL <= Mw < UMBRAL_OFICIAL -> partial_hit

    Reglas:
      1. Siempre intenta primero en pool full.
      2. Solo si no hay full disponible, intenta partial.
      3. Dentro del pool elige el evento de menor distancia.
      4. Matching 1-a-1 dentro de cada pool:
         - un evento full solo puede ser usado una vez
         - un evento partial solo puede ser usado una vez

    Esta métrica es "estricta" y útil para evaluar calidad fina del top-K,
    pero NO representa cobertura espacial total, porque aquí un mismo sismo
    no puede validar varias predicciones.
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
    log("=== INICIANDO VALIDACIÓN (PRODUCCIÓN: TOP-25, 100km) ===")
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
            pred_start = pd.to_datetime(run["input_max_time"]) + pd.Timedelta(days=1)
            pred_end_excl = pred_start + pd.Timedelta(days=7)
            pred_end_incl = pred_end_excl - pd.Timedelta(seconds=1)
            log("⚠️ No se encontró ventana en prediction_topk. Usando fallback de 7 días.")
        else:
            pred_start = pd.to_datetime(win["t_start"])
            pred_end_incl = pd.to_datetime(win["t_end"])
            pred_end_excl = pred_end_incl + pd.Timedelta(seconds=1)

        log(
            f"Evaluando Run #{run_id} "
            f"({pred_start} → {pred_end_incl}) | "
            f"K={EVALUAR_TOP_K} | tol={TOLERANCIA_KM}km"
        )

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
        # 4. CARGAR SISMOS REALES EN LA VENTANA EXACTA + ASIGNAR cell_id
        # ---------------------------------------------------------------------
        # Para métrica por celda–ventana, necesitamos el cell_id del evento real.
        # Usamos spatial join contra prediction_cells.geom.
        #
        # Nota:
        # - Si events_clean.geom no existe o viene null, usamos lon/lat para construir punto.
        # - Requiere PostGIS y prediction_cells.geom.
        df_real = pd.read_sql(
            text("""
                SELECT
                    e.id,
                    e.event_time_utc,
                    e.lat,
                    e.lon,
                    e.magnitude,
                    e.place,
                    c.cell_id
                FROM events_clean e
                LEFT JOIN prediction_cells c
                  ON ST_Intersects(
                        c.geom,
                        COALESCE(
                            e.geom,
                            ST_SetSRID(ST_MakePoint(e.lon, e.lat), 4326)
                        )
                     )
                WHERE e.event_time_utc >= :s
                  AND e.event_time_utc <  :e
                  AND e.magnitude >= :min_mag
            """),
            conn,
            params={"s": pred_start, "e": pred_end_excl, "min_mag": UMBRAL_PARCIAL}
        )

        # Pools separados
        df_full = df_real[df_real["magnitude"] >= UMBRAL_OFICIAL].copy()
        df_partial = df_real[
            (df_real["magnitude"] >= UMBRAL_PARCIAL) &
            (df_real["magnitude"] < UMBRAL_OFICIAL)
        ].copy()

        total_sismos_oficiales = int(len(df_full))

        # ---------------------------------------------------------------------
        # 4B. MÉTRICA POR CELDA–VENTANA (coherente con el target)
        # ---------------------------------------------------------------------
        # Definición:
        #   - official_cells: celdas distintas con >=1 sismo oficial
        #   - hit_official_cells: cuántas de esas celdas están en el Top-K (por cell_id)
        #   - cell_cover_recall_pct = hit_official_cells / official_cells
        #
        # Nota: aquí NO usamos tolerancia km. Es exacto por celda (malla del modelo).
        pred_cells_topk = set(
            df_preds["cell_id"].dropna().astype(int).tolist()
        )

        official_cells_set = set(
            df_full["cell_id"].dropna().astype(int).tolist()
        )

        official_cells = int(len(official_cells_set))
        hit_official_cells = int(len(official_cells_set.intersection(pred_cells_topk)))

        cell_cover_recall_pct = (
            hit_official_cells / official_cells * 100.0
            if official_cells else 0.0
        )

        log(
            f"[CELL-COVER] official_cells={official_cells} | "
            f"hit_official_cells={hit_official_cells} | "
            f"cell_cover_recall={cell_cover_recall_pct:.2f}%"
        )

        # ---------------------------------------------------------------------
        # 5. MATCHING ESTRICTO 1-a-1 POR EVENTO
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
                log(
                    f"  [MISS]    rank={int(pred['rank']):>2} → "
                    f"sin sismo en radio {TOLERANCIA_KM} km"
                )

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
        # 5B. MÉTRICAS DE COBERTURA CON TOLERANCIA (100 km)
        # ---------------------------------------------------------------------
        df_official = df_real[df_real["magnitude"] >= UMBRAL_OFICIAL].copy()
        official_events = int(len(df_official))

        preds_rank_cell = [
            (int(r.rank), int(r.cell_id), float(r.lat), float(r.lon))
            for r in df_preds.itertuples(index=False)
        ]

        events_xyz = [
            (int(r.id), float(r.lat), float(r.lon), float(r.magnitude), r.place)
            for r in df_official.itertuples(index=False)
        ]

        # A) Cobertura de eventos oficiales (sin 1-a-1)
        covered_event_ids = set()
        covered_events_detail = []

        for eid, elat, elon, emag, eplace in events_xyz:
            candidate_preds = []
            for prank, pcell, plat, plon in preds_rank_cell:
                dist = haversine_km(plat, plon, elat, elon)
                if dist <= TOLERANCIA_KM:
                    candidate_preds.append({
                        "rank": prank,
                        "cell_id": pcell,
                        "dist_km": round(dist, 2),
                    })

            if candidate_preds:
                covered_event_ids.add(eid)
                covered_events_detail.append({
                    "event_id": eid,
                    "evento": eplace,
                    "magnitud": emag,
                    "matching_preds": candidate_preds,
                })

        covered_official_events = int(len(covered_event_ids))

        event_cover_recall_pct = (
            covered_official_events / official_events * 100.0
            if official_events else 0.0
        )

        # B) Predicciones del Top-K que tuvieron al menos un evento oficial cerca
        preds_with_official_event = 0
        covered_pred_detail = []

        for prank, pcell, plat, plon in preds_rank_cell:
            nearby_events = []
            for eid, elat, elon, emag, eplace in events_xyz:
                dist = haversine_km(plat, plon, elat, elon)
                if dist <= TOLERANCIA_KM:
                    nearby_events.append({
                        "event_id": eid,
                        "evento": eplace,
                        "magnitud": emag,
                        "dist_km": round(dist, 2),
                    })

            if nearby_events:
                preds_with_official_event += 1
                covered_pred_detail.append({
                    "rank": prank,
                    "cell_id": pcell,
                    "nearby_events": nearby_events,
                })

        event_cover_precision_pct = (
            preds_with_official_event / total_alertas * 100.0
            if total_alertas else 0.0
        )

        log(
            f"[COVER] official_events={official_events} | "
            f"covered_official_events={covered_official_events} | "
            f"event_cover_recall={event_cover_recall_pct:.2f}% | "
            f"preds_with_official_event={preds_with_official_event}/{total_alertas} | "
            f"event_cover_precision={event_cover_precision_pct:.2f}%"
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
        # 7. MÉTRICAS ESTRICTAS (EVENTO 1-a-1)
        # ---------------------------------------------------------------------
        recall = (
            full_hits / total_sismos_oficiales * 100.0
            if total_sismos_oficiales else 0.0
        )
        precision = (
            full_hits / total_alertas * 100.0
            if total_alertas else 0.0
        )
        f1 = (
            2 * (precision / 100) * (recall / 100)
            / ((precision / 100) + (recall / 100))
            * 100
            if (precision + recall) > 0 else 0.0
        )

        # ---------------------------------------------------------------------
        # 8. GUARDAR RESUMEN
        # ---------------------------------------------------------------------
        has_cover_cols = (
            table_has_column(conn, "validation_realworld", "official_events") and
            table_has_column(conn, "validation_realworld", "covered_official_events") and
            table_has_column(conn, "validation_realworld", "preds_with_official_event") and
            table_has_column(conn, "validation_realworld", "event_cover_recall_pct") and
            table_has_column(conn, "validation_realworld", "event_cover_precision_pct")
        )

        has_cover_json_cols = (
            table_has_column(conn, "validation_realworld", "covered_event_ids_json") and
            table_has_column(conn, "validation_realworld", "covered_events_detail_json") and
            table_has_column(conn, "validation_realworld", "covered_pred_detail_json")
        )

        # NUEVO: columnas para métrica por celda
        has_cell_cols = (
            table_has_column(conn, "validation_realworld", "official_cells") and
            table_has_column(conn, "validation_realworld", "hit_official_cells") and
            table_has_column(conn, "validation_realworld", "cell_cover_recall_pct")
        )

        insert_sql = """
            INSERT INTO validation_realworld (
                run_id, window_start, window_end,
                total_sismos, sismos_detectados, partial_hits,
                recall_pct, precision_pct, f1_pct,
                aciertos_json, partial_hits_json
        """

        if has_cover_cols:
            insert_sql += """
                , official_events, covered_official_events, preds_with_official_event,
                  event_cover_recall_pct, event_cover_precision_pct
            """

        if has_cover_json_cols:
            insert_sql += """
                , covered_event_ids_json, covered_events_detail_json, covered_pred_detail_json
            """

        if has_cell_cols:
            insert_sql += """
                , official_cells, hit_official_cells, cell_cover_recall_pct
            """

        insert_sql += """
            ) VALUES (
                :r, :s, :e,
                :ts, :full, :partial,
                :rec, :prec, :f1,
                :json_full, :json_partial
        """

        if has_cover_cols:
            insert_sql += """
                , :oe, :coe, :pwoe, :er, :ep
            """

        if has_cover_json_cols:
            insert_sql += """
                , :ceids, :cedetail, :cpdetail
            """

        if has_cell_cols:
            insert_sql += """
                , :oc, :hoc, :ccr
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

        if has_cover_cols:
            payload.update({
                "oe": official_events,
                "coe": covered_official_events,
                "pwoe": preds_with_official_event,
                "er": round(event_cover_recall_pct, 4),
                "ep": round(event_cover_precision_pct, 4),
            })

        if has_cover_json_cols:
            payload.update({
                "ceids": json.dumps(sorted(list(covered_event_ids))),
                "cedetail": json.dumps(covered_events_detail),
                "cpdetail": json.dumps(covered_pred_detail),
            })

        if has_cell_cols:
            payload.update({
                "oc": official_cells,
                "hoc": hit_official_cells,
                "ccr": round(cell_cover_recall_pct, 4),
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
        print("─────────────────────────────────────")
        print(f"Sismos reales (Mw≥{UMBRAL_OFICIAL}):   {total_sismos_oficiales}")
        print(f"Sismos sub-umbral ({UMBRAL_PARCIAL}–{UMBRAL_OFICIAL}): {len(df_partial)}")
        print("─────────────────────────────────────")
        print(f"Alertas evaluadas:          {total_alertas}")
        print(f"  Aciertos completos:       {full_hits}")
        print(f"  Aciertos parciales:       {partial_hits}")
        print(f"  Sin match:                {no_hits}")
        print("─────────────────────────────────────")
        print(f"Recall estricto (full):     {recall:.1f}%")
        print(f"Precision estricta (full):  {precision:.1f}%")
        print(f"F1 estricto (full):         {f1:.1f}%")
        print("─────────────────────────────────────")
        print(f"Recall cobertura (Mw≥{UMBRAL_OFICIAL}):    {event_cover_recall_pct:.2f}% "
              f"({covered_official_events}/{official_events})")
        print(f"Precision cobertura (Top-K):            {event_cover_precision_pct:.2f}% "
              f"({preds_with_official_event}/{total_alertas})")
        print("─────────────────────────────────────")
        print(f"Recall por celda (Mw≥{UMBRAL_OFICIAL}):  {cell_cover_recall_pct:.2f}% "
              f"({hit_official_cells}/{official_cells})")
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
        if covered_official_events > official_events:
            log("[WARNING] covered_official_events > official_events. Revisar lógica.")
        if preds_with_official_event > total_alertas:
            log("[WARNING] preds_with_official_event > total_alertas. Revisar lógica.")
        if hit_official_cells > official_cells:
            log("[WARNING] hit_official_cells > official_cells. Revisar lógica.")


if __name__ == "__main__":
    main()