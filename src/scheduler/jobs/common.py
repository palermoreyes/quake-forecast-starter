"""
common.py
---------
Módulo central de utilidades compartidas.

Mejoras V3.5:
- Logging unificado con Rotación Automática (RotatingFileHandler).
- Evita llenar el disco: Mantiene máximo 15MB de logs (5MB x 3 archivos).
- Compatible con ETL y Forecast.
"""

import os
import hashlib
import requests
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine


# -------------------------------------------------------------------
# LOGGING PROFESIONAL (Con Rotación)
# -------------------------------------------------------------------

# 1. Configuración de rutas
LOG_DIR = Path("/app/artifacts/etl_logs")
LOG_FILE = LOG_DIR / "system_unified.log"

# Asegurar que el directorio exista
os.makedirs(LOG_DIR, exist_ok=True)

def _setup_logger():
    """Configura el logger una sola vez con rotación y salida a consola."""
    logger = logging.getLogger("QuakeSystem")
    
    # Evitar duplicar handlers si se recarga el módulo
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Formato: [FECHA] [NIVEL] Mensaje
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # A. HANDLER DE ARCHIVO (ROTATIVO)
    # - maxBytes=5MB: Cuando llega a 5MB, cierra el archivo y crea uno nuevo.
    # - backupCount=2: Guarda máximo 2 archivos viejos. Borra los más antiguos automáticamente.
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # B. HANDLER DE CONSOLA (Para 'docker logs')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

# Instancia global del logger
_sys_logger = _setup_logger()


def log(msg: str) -> None:
    """
    Registra un mensaje en el sistema unificado.
    Mantiene compatibilidad con tu código existente que llama a log("msg").
    """
    # Agregamos el prefijo [SCHED] manualmente si quieres mantener el estilo visual,
    # aunque el formatter ya pone fecha y nivel.
    _sys_logger.info(f"[SCHED] {msg}")


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