"""
fetch_igp_extract.py
--------------------
Módulo encargado de la extracción de datos del Instituto Geofísico del Perú (IGP).

Descarga un archivo XLSX con los registros sísmicos desde el endpoint público del IGP,
guardando el resultado localmente solo si hay cambios (comparación por hash MD5).
"""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from .common import log, ensure_dirs, file_hash, safe_get

# Parámetros básicos de configuración
URL_BASE   = os.getenv("IGP_DOWNLOAD_URL", "https://ultimosismo.igp.gob.pe/api/ultimo-sismo/descargar-datos")
DEST_PATH  = os.getenv("IGP_HIST_LOCAL", "/app/data/igp/landing/igp-datos-sismicos.xlsx")
LOOKBACK   = int(os.getenv("IGP_LOOKBACK_DAYS", "30"))
RAW_DIR    = os.getenv("IGP_RAW_DIR", "/app/artifacts/etl_raw")

def _is_xlsx(content: bytes) -> bool:
    # XLSX es un ZIP: magic bytes 'PK\x03\x04'
    return len(content) >= 4 and content[:4] == b"PK\x03\x04"

def main() -> None:
    """Descarga el archivo XLSX del IGP y lo guarda en disco si hay cambios."""
    ensure_dirs(os.path.dirname(DEST_PATH))
    ensure_dirs(RAW_DIR)

    tz = ZoneInfo(os.getenv("IGP_TZ", "America/Lima"))
    fecha_fin = datetime.now(tz).date()
    fecha_ini = fecha_fin - timedelta(days=LOOKBACK)

    params = {
        "tipoCatalogo":      os.getenv("IGP_CATALOGO", "Instrumental"),
        "fechaInicio":       fecha_ini.isoformat(),
        "fechaFin":          fecha_fin.isoformat(),
        "minimaMagnitud":    os.getenv("IGP_MAG_MIN", "1"),
        "maximaMagnitud":    os.getenv("IGP_MAG_MAX", "9"),
        "minimaProfundidad": os.getenv("IGP_PROF_MIN", "0"),
        "maximaProfundidad": os.getenv("IGP_PROF_MAX", "900"),
        "latitudNorte":      os.getenv("IGP_LAT_N", "-1.396"),
        "latitudSur":        os.getenv("IGP_LAT_S", "-25.701"),
        "longitudEste":      os.getenv("IGP_LON_E", "-65.624"),
        "longitudOeste":     os.getenv("IGP_LON_W", "-87.382"),
    }

    log(f"Descargando datos IGP desde {fecha_ini} hasta {fecha_fin}")
    response = safe_get(URL_BASE, params)

    status = response.status_code
    ctype  = (response.headers.get("Content-Type") or "").lower()
    log(f"HTTP {status} | Content-Type: {ctype} | bytes: {len(response.content)}")

    if status != 200:
        raw_path = os.path.join(RAW_DIR, f"igp_{datetime.utcnow().isoformat().replace(':','')}_{status}.bin")
        with open(raw_path, "wb") as f:
            f.write(response.content)
        log(f"Advertencia: respuesta HTTP {status}. Guardado bruto en {raw_path}. No se reemplaza XLSX previo.")
        return

    content = response.content

    # Acepta sólo si el binario luce como XLSX por magic bytes.
    # (El Content-Type del servidor puede ser sheet, octet-stream, etc. No es confiable.)
    if not _is_xlsx(content):
        raw_path = os.path.join(RAW_DIR, f"igp_{datetime.utcnow().isoformat().replace(':','')}_not_xlsx.bin")
        with open(raw_path, "wb") as f:
            f.write(content)
        log(f"Advertencia: el contenido no parece XLSX (no ZIP). Guardado bruto en {raw_path}. No se reemplaza XLSX previo.")
        return

    # Es XLSX → escribe a archivo temporal, compara hash y reemplaza si cambió
    tmp = DEST_PATH + ".tmp"
    with open(tmp, "wb") as f:
        f.write(content)

    if file_hash(DEST_PATH) == file_hash(tmp):
        os.remove(tmp)
        log("Archivo sin cambios (hash idéntico). No se reemplaza el existente.")
        return

    os.replace(tmp, DEST_PATH)
    log(f"Archivo actualizado correctamente: {DEST_PATH} ({len(content):,} bytes)")
