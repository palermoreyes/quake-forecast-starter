from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from sqlalchemy import text
from .db import SessionLocal, ping

app = FastAPI(title="Quake Forecast API", default_response_class=ORJSONResponse)

@app.get("/api/health")
def health():
    return {"status": "ok", "db_time": str(ping())}

@app.get("/api/recent")
def recent(limit: int = 50):
    with SessionLocal() as s:
        q = text("""
            SELECT t_utc, lat, lon, depth_km, mag
            FROM events_clean
            ORDER BY t_utc DESC
            LIMIT :lim
        """)
        rows = s.execute(q, {"lim": limit}).mappings().all()
        return {"events": list(rows)}

@app.get("/api/forecast")
def forecast(run: str | None = None, horizon: int = 7, mag: float = 4.5, limit: int = 10000):
    with SessionLocal() as s:
        if run is None:
            run = s.execute(text("""
                SELECT max(run_at) FROM forecasts WHERE horizon_days=:h AND mag_thr=:m
            """), {"h": horizon, "m": mag}).scalar()
        rows = s.execute(text("""
            SELECT lat_bin, lon_bin, prob
            FROM forecasts
            WHERE run_at=:r AND horizon_days=:h AND mag_thr=:m
            LIMIT :lim
        """), {"r": run, "h": horizon, "m": mag, "lim": limit}).mappings().all()
        return {"run_at": str(run), "horizon_days": horizon, "mag_thr": mag, "cells": list(rows)}
