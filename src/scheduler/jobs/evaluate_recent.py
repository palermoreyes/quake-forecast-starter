import pandas as pd
import math
import json
from sqlalchemy import text
from .common import get_engine, log

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA
# =============================================================================

# Radio de tolerancia espacial (criterio científico)
# 100 km es consistente con:
# - propagación de ondas sísmicas
# - resolución del grid
# - literatura sismológica regional
TOLERANCIA_KM = 100

# Número de alertas evaluadas por corrida
# Coherente con frontend (Top-25) y enfoque operativo
EVALUAR_TOP_K = 25


# =============================================================================
# UTILIDADES
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Distancia Haversine entre dos puntos geográficos (km).
    """
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


# =============================================================================
# PROCESO PRINCIPAL
# =============================================================================

def main():
    log("=== INICIANDO VALIDACIÓN (HISTÓRICO) ===")
    engine = get_engine()

    with engine.connect() as conn:

        # ---------------------------------------------------------------------
        # 1. OBTENER ÚLTIMA CORRIDA DE PRONÓSTICO
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

        # Evitar evaluaciones duplicadas
        exists = conn.execute(
            text("""
                SELECT 1
                FROM validation_realworld
                WHERE run_id = :r
            """),
            {"r": run_id}
        ).first()

        if exists:
            log(f"⚠️ Run #{run_id} ya fue evaluado. Se omite.")
            return

        # ---------------------------------------------------------------------
        # 2. DEFINIR VENTANA DE VALIDACIÓN
        # ---------------------------------------------------------------------
        # Día siguiente al último dato usado → ventana de 7 días
        pred_start = pd.to_datetime(run["input_max_time"]) + pd.Timedelta(days=1)
        pred_end = pred_start + pd.Timedelta(days=7)

        log(f"Evaluando Run #{run_id} ({pred_start.date()} → {pred_end.date()})")

        # ---------------------------------------------------------------------
        # 3. CARGAR PREDICCIONES (TOP-K)
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
        # 4. CARGAR SISMOS REALES (IGP, M ≥ 4.0)
        # ---------------------------------------------------------------------
        df_real = pd.read_sql(
            text("""
                SELECT id, event_time_utc, lat, lon, magnitude, place
                FROM events_clean
                WHERE event_time_utc BETWEEN :s AND :e
                  AND magnitude >= 4.0
            """),
            conn,
            params={"s": pred_start, "e": pred_end}
        )

        total_sismos = len(df_real)
        total_alertas = len(df_preds)

        detectados = 0
        detalle = []

        # ---------------------------------------------------------------------
        # 5. MATCHING ALERTA ↔ REALIDAD (TRAZA INDIVIDUAL)
        # ---------------------------------------------------------------------
        # Cada alerta se evalúa individualmente contra los sismos reales.
        # Se registra si existe al menos un evento dentro del radio tolerado.
        with engine.begin() as tx:
            for _, pred in df_preds.iterrows():

                matched = False
                best_match = None
                best_dist = None

                for _, sismo in df_real.iterrows():
                    dist = haversine_km(
                        pred["lat"], pred["lon"],
                        sismo["lat"], sismo["lon"]
                    )
                    if dist <= TOLERANCIA_KM:
                        matched = True
                        best_match = sismo
                        best_dist = dist
                        break

                if matched:
                    detectados += 1

                # Guardar traza histórica por alerta
                tx.execute(
                    text("""
                        INSERT INTO public.prediction_trace (
                            run_id,
                            rank,
                            lat,
                            lon,
                            place,
                            predicted_window_start,
                            predicted_window_end,
                            matched_event,
                            matched_event_id,
                            event_time_utc,
                            event_magnitude,
                            event_place,
                            distance_km,
                            tolerance_km
                        )
                        VALUES (
                            :run_id,
                            :rank,
                            :lat,
                            :lon,
                            :place,
                            :ws,
                            :we,
                            :matched,
                            :eid,
                            :etime,
                            :mag,
                            :eplace,
                            :dist,
                            :tol
                        )
                    """),
                    {
                        "run_id": run_id,
                        "rank": pred["rank"],
                        "lat": pred["lat"],
                        "lon": pred["lon"],
                        "place": pred["place"],
                        "ws": pred_start.date(),
                        "we": pred_end.date(),
                        "matched": matched,
                        "eid": best_match["id"] if matched else None,
                        "etime": best_match["event_time_utc"] if matched else None,
                        "mag": best_match["magnitude"] if matched else None,
                        "eplace": best_match["place"] if matched else None,
                        "dist": round(best_dist, 2) if matched else None,
                        "tol": TOLERANCIA_KM
                    }
                )

                if matched:
                    detalle.append({
                        "rank": int(pred["rank"]),
                        "evento": best_match["place"],
                        "magnitud": float(best_match["magnitude"]),
                        "dist_km": round(best_dist, 2)
                    })

        # ---------------------------------------------------------------------
        # 6. MÉTRICAS AGREGADAS
        # ---------------------------------------------------------------------
        recall = (detectados / total_sismos * 100) if total_sismos else 0.0
        precision = (detectados / total_alertas * 100) if total_alertas else 0.0

        # ---------------------------------------------------------------------
        # 7. GUARDAR RESUMEN GLOBAL
        # ---------------------------------------------------------------------
        with engine.begin() as tx:
            tx.execute(
                text("""
                    INSERT INTO validation_realworld (
                        run_id,
                        window_start,
                        window_end,
                        total_sismos,
                        sismos_detectados,
                        recall_pct,
                        aciertos_json
                    )
                    VALUES (
                        :r,
                        :s,
                        :e,
                        :ts,
                        :d,
                        :rec,
                        :json
                    )
                """),
                {
                    "r": run_id,
                    "s": pred_start.date(),
                    "e": pred_end.date(),
                    "ts": total_sismos,
                    "d": detectados,
                    "rec": recall,
                    "json": json.dumps(detalle)
                }
            )

        # ---------------------------------------------------------------------
        # 8. REPORTE EN CONSOLA
        # ---------------------------------------------------------------------
        print("\n=== RESULTADO VALIDACIÓN REAL-WORLD ===")
        print(f"Sismos reales (M≥4.0): {total_sismos}")
        print(f"Alertas evaluadas:      {total_alertas}")
        print(f"Alertas acertadas:      {detectados}")
        print(f"Recall:                {recall:.1f}%")
        print(f"Precision:             {precision:.1f}%")
        print("=======================================")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    main()
