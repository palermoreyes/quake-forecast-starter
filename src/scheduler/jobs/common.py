"""
common.py
---------
Módulo central de utilidades compartidas por los jobs del scheduler.

Incluye:
- Manejo de logs (con persistencia en archivo y salida por consola).
- Funciones auxiliares para crear directorios, obtener hashes de archivos, 
  realizar peticiones HTTP seguras y establecer conexión con la base de datos.
"""

import os
import hashlib
import requests
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Ruta del archivo de log persistente
LOG_FILE = Path("/app/artifacts/etl_logs/fetch_igp.log")


def log(msg: str) -> None:
    """
    Registra un mensaje tanto en consola como en archivo.

    Args:
        msg (str): Mensaje a registrar.
    """
    now = datetime.now().isoformat(timespec="seconds")
    line = f"[SCHED] [{now}] {msg}"
    print(line, flush=True)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[SCHED] Error escribiendo log: {e}", flush=True)


def ensure_dirs(path: str) -> None:
    """Crea un directorio de forma segura si no existe."""
    os.makedirs(path, exist_ok=True)


def file_hash(path: str) -> str | None:
    """Calcula el hash MD5 de un archivo para comparar cambios."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def safe_get(url: str, params: dict, timeout: int = 180) -> requests.Response:
    """
    Realiza una petición GET con control de errores.

    Args:
        url (str): URL destino.
        params (dict): Parámetros de la solicitud.
        timeout (int): Tiempo máximo de espera (segundos).
    """
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r


def get_engine() -> Engine:
    """
    Crea y devuelve un motor SQLAlchemy conectado a la base de datos.

    Returns:
        Engine: Conexión SQLAlchemy.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL no configurada.")
    return create_engine(db_url)
