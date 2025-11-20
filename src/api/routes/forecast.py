# src/api/routes/forecast.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List

# Importamos get_db desde el archivo database.py que está en la raíz de src/api
from database import get_db 

router = APIRouter(
    prefix="/forecast",
    tags=["Forecast"]
)

@router.get("/latest", summary="Obtener última predicción en GeoJSON")
async def get_latest_forecast(db: AsyncSession = Depends(get_db)):
    """
    Devuelve la grilla de predicción más reciente (7 días).
    Los datos ya vienen formateados como GeoJSON desde PostGIS.
    """
    try:
        # Usamos la Vista Materializada que creamos anteriormente.
        # ST_AsGeoJSON convierte la geometría a JSON nativamente en la BD.
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
                                'horizon', horizon_days
                            )
                        )
                    ), '[]'::json)
                )
            FROM vw_latest_grid_7d;
        """)
        
        result = await db.execute(query)
        geojson = result.scalar()
        
        return geojson

    except Exception as e:
        print(f"Error consultando DB: {e}")
        raise HTTPException(status_code=500, detail="Error interno recuperando predicciones")

@router.get("/status", summary="Estado del modelo")
async def get_model_status(db: AsyncSession = Depends(get_db)):
    """
    Metadatos sobre la última corrida del modelo.
    """
    try:
        
        query = text("""
            SELECT MAX(run_id) as last_run, MAX(generated_at) as run_date 
            FROM prediction_run;
        """)

        result = await db.execute(query)
        row = result.first()
        
        if not row or row[0] is None:
            return {"status": "Sin datos", "last_run": None}
            
        return {
            "status": "Operativo",
            "last_run_id": row[0],
            "last_run_date": row[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.get("/topk", summary="Top-K de posibles próximos sismos")
async def get_forecast_topk(
    horizon_days: int = 7,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Devuelve el Top-K de posibles próximos sismos para el último run de predicción.
    Usa la tabla prediction_topk y prediction_run.
    """

    try:
        # 1) Obtener el último run
        query_run = text("""
            SELECT run_id, generated_at, input_max_time, horizons, mag_min
            FROM prediction_run
            ORDER BY run_id DESC
            LIMIT 1;
        """)
        result_run = await db.execute(query_run)
        row_run = result_run.first()

        if not row_run:
            raise HTTPException(status_code=404, detail="No hay runs de predicción disponibles.")

        run_id, generated_at, input_max_time, horizons, mag_min = row_run

        # 2) Leer Top-K para ese run y horizonte
        query_topk = text("""
            SELECT
                run_id,
                horizon_days,
                rank,
                t_pred_start,
                t_pred_end,
                lat,
                lon,
                mag_pred,
                prob,
                time_conf_h,
                space_conf_km
            FROM prediction_topk
            WHERE run_id = :run_id
              AND horizon_days = :horizon_days
            ORDER BY rank
            LIMIT :limit;
        """)

        result_topk = await db.execute(
            query_topk,
            {
                "run_id": run_id,
                "horizon_days": horizon_days,
                "limit": limit
            }
        )

        rows = result_topk.fetchall()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No hay predicciones Top-K para horizon_days={horizon_days} en run_id={run_id}."
            )

        # Formatear respuesta
        topk_list = []
        for r in rows:
            topk_list.append({
                "run_id": r.run_id,
                "rank": r.rank,
                "horizon_days": r.horizon_days,
                "t_pred_start": r.t_pred_start,
                "t_pred_end": r.t_pred_end,
                "lat": float(r.lat),
                "lon": float(r.lon),
                "mag_pred": float(r.mag_pred) if r.mag_pred is not None else float(mag_min),
                "prob": float(r.prob),
                "time_conf_h": int(r.time_conf_h),
                "space_conf_km": int(r.space_conf_km),
            })

        return {
            "generated_at": generated_at,
            "input_max_time": input_max_time,
            "horizon_days": horizon_days,
            "mag_min": float(mag_min),
            "topk": topk_list
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en /forecast/topk: {e}")
        raise HTTPException(status_code=500, detail="Error interno recuperando Top-K")
