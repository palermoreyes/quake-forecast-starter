"""
common.py
---------
Módulo central de utilidades compartidas por los jobs del scheduler.

Incluye:
- Manejo de logs (con persistencia en archivo y salida por consola).
- Funciones auxiliares para crear directorios, obtener hashes de archivos,
  realizar peticiones HTTP seguras y establecer conexiones (sync y async) 
  con la base de datos.
"""

import os
import hashlib
import requests
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Para motor asíncrono (FastAPI u otros scripts async)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine


# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------

LOG_FILE = Path("/app/artifacts/etl_logs/fetch_igp.log")


def log(msg: str) -> None:
    """Registra un mensaje tanto en consola como en archivo."""
    now = datetime.now().isoformat(timespec="seconds")
    line = f"[SCHED] [{now}] {msg}"
    print(line, flush=True)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[SCHED] Error escribiendo log: {e}", flush=True)


# -------------------------------------------------------------------
# UTILIDADES DE ARCHIVOS Y HTTP
# -------------------------------------------------------------------

def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def file_hash(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def safe_get(url: str, params: dict, timeout: int = 180) -> requests.Response:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r


# -------------------------------------------------------------------
# BASE DE DATOS
# -------------------------------------------------------------------

def get_engine() -> Engine:
    db_url = (
        os.getenv("DATABASE_URL_SYNC")    # recomendado
        or os.getenv("DATABASE_URL")      # fallback
    )
    if not db_url:
        raise RuntimeError("DATABASE_URL_SYNC o DATABASE_URL no configurada.")
    return create_engine(db_url, future=True, pool_pre_ping=True)


def get_async_engine() -> AsyncEngine:
    db_url = os.getenv("DATABASE_URL_ASYNC")
    if not db_url:
        raise RuntimeError("DATABASE_URL_ASYNC no configurada para motor async.")
    return create_async_engine(db_url, echo=False, future=True)
