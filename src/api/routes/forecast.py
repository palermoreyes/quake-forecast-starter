# src/api/routes/forecast.py

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from database import get_db
from datetime import datetime

router = APIRouter(
    prefix="/forecast",
    tags=["Forecast"]
)

@router.get("/latest", summary="Obtener última predicción en GeoJSON")
async def get_latest_forecast(db: AsyncSession = Depends(get_db)):
    try:
        query = text("""
            SELECT 
                json_build_object(
                    'type', 'FeatureCollection',
                    'features', COALESCE(json_agg(
                        json_build_object(
                            'type', 'Feature',
                            'geometry', ST_AsGeoJSON(centroid)::json,
                            'properties', json_build_object(
                                'cell_id', cell_id,
                                'prob', prob,
                                'rank_pct', rank_pct,
                                'density', density,
                                'horizon', horizon_days,
                                'place', place
                            )
                        )
                    ), '[]'::json)
                )
            FROM vw_latest_grid_7d;
        """)
        result = await db.execute(query)
        return result.scalar()
    except Exception as e:
        print(f"Error DB: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

@router.get("/status", summary="Estado del modelo")
async def get_model_status(db: AsyncSession = Depends(get_db)):
    try:
        query = text("SELECT MAX(run_id) as last_run, MAX(generated_at) as run_date FROM prediction_run;")
        result = await db.execute(query)
        row = result.first()
        if not row or row[0] is None:
            return {"status": "Sin datos", "last_run": None}
        return {"status": "Operativo", "last_run_id": row[0], "last_run_date": row[1]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topk", summary="Top-K de posibles próximos sismos")
async def get_forecast_topk(horizon_days: int = 7, limit: int = 10, db: AsyncSession = Depends(get_db)):
    try:
        # 1. Datos del Run
        q_run = text("SELECT run_id, generated_at, input_max_time, horizons, mag_min FROM prediction_run ORDER BY run_id DESC LIMIT 1;")
        res_run = await db.execute(q_run)
        row_run = res_run.first()

        if not row_run:
            raise HTTPException(status_code=404, detail="No data")

        run_id, gen_at, in_max, horizons, mag_min = row_run

        # 2. Datos del Top-K (INCLUYENDO 'place')
        q_top = text("""
            SELECT rank, lat, lon, mag_pred, prob, place, t_pred_start, t_pred_end, time_conf_h, space_conf_km
            FROM prediction_topk 
            WHERE run_id = :rid AND horizon_days = :hz 
            ORDER BY rank LIMIT :lim
        """)

        res_top = await db.execute(q_top, {"rid": run_id, "hz": horizon_days, "lim": limit})

        topk_list = []
        for r in res_top.fetchall():
            topk_list.append({
                "rank": r.rank,
                "lat": float(r.lat),
                "lon": float(r.lon),
                "mag_pred": float(r.mag_pred) if r.mag_pred else float(mag_min),
                "prob": float(r.prob),
                "place": r.place if r.place else "Zona Remota",
                "t_pred_start": r.t_pred_start,
                "t_pred_end": r.t_pred_end,
                "time_conf_h": int(r.time_conf_h) if r.time_conf_h is not None else 0,
                "space_conf_km": int(r.space_conf_km) if r.space_conf_km is not None else 0
            })

        return {
            "generated_at": gen_at,
            "input_max_time": in_max,
            "topk": topk_list
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- NUEVO ENDPOINT: ÚLTIMO SISMO REAL ---
@router.get("/last-event", summary="Último sismo real reportado")
async def get_last_real_event(db: AsyncSession = Depends(get_db)):
    try:
        q = text("""
            SELECT event_time_utc, lat, lon, depth_km, magnitude, place 
            FROM events_clean 
            ORDER BY event_time_utc DESC LIMIT 1;
        """)
        res = await db.execute(q)
        row = res.first()

        if not row:
            return None

        return {
            "event_time_utc": row.event_time_utc,
            "lat": row.lat,
            "lon": row.lon,
            "depth_km": row.depth_km,
            "magnitude": row.magnitude,
            "place": row.place if row.place else "Ubicación no especificada"
        }
    except Exception as e:
        print(f"Error last-event: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

# --- NUEVO ENDPOINT: SISMOS EN VENTANA (para el dashboard) ---
@router.get("/events-window", summary="Sismos reales (Mw>=4.0) dentro de un rango [start, end)")
async def get_events_in_window(
    start: datetime = Query(..., description="Inicio (ISO). Ej: 2026-02-01T00:00:00Z"),
    end: datetime = Query(..., description="Fin (ISO, exclusivo). Ej: 2026-02-08T00:00:00Z"),
    min_mag: float = Query(4.0, description="Magnitud mínima"),
    limit: int = Query(500, ge=1, le=5000, description="Límite de eventos devueltos"),
    db: AsyncSession = Depends(get_db)
):
    try:
        if end <= start:
            raise HTTPException(status_code=400, detail="Parámetros inválidos: end debe ser mayor que start")

        q = text("""
            SELECT id, event_time_utc, lat, lon, depth_km, magnitude, place
            FROM events_clean
            WHERE event_time_utc >= :start
              AND event_time_utc <  :end
              AND magnitude >= :min_mag
              AND event_time_utc <> (SELECT max(event_time_utc) FROM events_clean)
            ORDER BY event_time_utc DESC
            LIMIT :lim;
        """)

        res = await db.execute(q, {"start": start, "end": end, "min_mag": min_mag, "lim": limit})
        rows = res.fetchall()

        return [
            {
                "id": r.id,
                "event_time_utc": r.event_time_utc,
                "lat": float(r.lat),
                "lon": float(r.lon),
                "depth_km": float(r.depth_km) if r.depth_km is not None else None,
                "magnitude": float(r.magnitude),
                "place": r.place if r.place else "Ubicación no especificada",
            }
            for r in rows
        ]

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error events-window: {e}")
        raise HTTPException(status_code=500, detail="Error interno")