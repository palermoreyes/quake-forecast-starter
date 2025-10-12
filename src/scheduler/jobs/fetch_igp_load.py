"""
fetch_igp_load.py
-----------------
Carga de datos a Postgres desde el XLSX del IGP.

- Detecta encabezado real.
- Mapea columnas por similitud (fecha/hora/lat/lon/prof/mag).
- Normaliza decimales (coma), invisibles, y parsea fecha/hora (serial Excel o texto).
- Inserta solo nuevos registros (gracias al índice único y ON CONFLICT DO NOTHING).
"""

import os
import json
import re
import math
import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime, timezone, timedelta
from dateutil import parser as dparser

from .common import log, get_engine

PATH_XLSX = os.getenv("IGP_HIST_LOCAL", "/app/data/igp/landing/igp-datos-sismicos.xlsx")

# Aliases esperados (case-insensitive, espacios normalizados)
REQ_COLS_CANON = {
    "fecha": ["fecha utc", "fecha", "fecha (utc)", "date utc", "fecha_utc"],
    "hora":  ["hora utc", "hora", "hora (utc)", "time utc", "hora_utc"],
    "lat":   ["latitud (º)", "latitud", "lat", "latitude"],
    "lon":   ["longitud (º)", "longitud", "lon", "longitude"],
    "prof":  ["profundidad (km)", "profundidad", "depth", "depth (km)"],
    "mag":   ["magnitud (m)", "magnitud", "magnitude", "mag", "ml", "mw"]
}

def _strip(s):
    if pd.isna(s): return None
    return str(s).replace("\u200b","").replace("\ufeff","").strip()

def _normalize_header(h):
    h = _strip(h) or ""
    h = h.lower()
    h = re.sub(r"\s+", " ", h)
    return h

def _map_headers(cols):
    norm = [_normalize_header(c) for c in cols]
    mapping = {}
    for key, aliases in REQ_COLS_CANON.items():
        found = None
        # match exact
        for i, c in enumerate(norm):
            if c in aliases:
                found = cols[i]; break
        # fallback contains
        if not found:
            for i, c in enumerate(norm):
                if any(a in c for a in aliases):
                    found = cols[i]; break
        if found:
            mapping[key] = found
    return mapping

def _to_float_dot(x):
    import re, numpy as np
    from pandas import isna
    if isna(x):
        return np.nan
    s = str(x).replace("\u200b","").replace("\ufeff","").strip()
    # Quita solo separadores de miles tipo espacio o apóstrofe (no toca '.')
    s = s.replace("\xa0"," ").replace("\u202f"," ")
    # Si hubiera comas, trátalas como miles y elimínalas (no convertir a '.')
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return np.nan

def _parse_date_part(fecha):
    # 1) Serial Excel
    if isinstance(fecha, (int, float)) and not pd.isna(fecha):
        base = datetime(1899, 12, 30, tzinfo=timezone.utc)
        return base + pd.to_timedelta(float(fecha), unit="D")

    s = _strip(fecha)
    if not s: return None

    s_norm = re.sub(r"[./]", "-", s)  # unifica separadores
    s_norm = re.sub(r"\s+", " ", s_norm).strip()

    # 2) ISO Y-M-D → forzar YMD sin heurísticas
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s_norm)
    if m:
        y, mth, d = map(int, m.groups())
        return datetime(y, mth, d, tzinfo=timezone.utc)

    # 3) D-M-Y o M-D-Y → decidir por límites
    m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", s_norm)
    if m:
        a, b, y = map(int, m.groups())
        if a > 12 and b <= 12:     # 31-10-2025 → D-M-Y
            d, mth = a, b
        elif b > 12 and a <= 12:   # 10-31-2025 → M-D-Y
            d, mth = b, a
        else:
            # Ambiguo (p.ej., 05-10-2025). En Perú: día/mes → D-M-Y
            d, mth = a, b
        return datetime(y, mth, d, tzinfo=timezone.utc)

    # 4) Fallback: dateutil (día/mes primero)
    dt_ = dparser.parse(s, fuzzy=True, dayfirst=True)
    if dt_.tzinfo is None: dt_ = dt_.replace(tzinfo=timezone.utc)
    return dt_.astimezone(timezone.utc)

def _parse_time_part(hora, base_dt):
    if base_dt is None: return None
    if hora is None or pd.isna(hora): 
        return base_dt
    # Excel time como fracción del día
    if isinstance(hora, (int, float)):
        frac = float(hora) % 1.0
        return base_dt + timedelta(seconds=round(frac * 86400))
    # Texto de hora
    hs = _strip(hora)
    if not hs: return base_dt
    hs = hs.replace(",", ".")
    t = dparser.parse(hs, fuzzy=True, default=base_dt.replace(hour=0, minute=0, second=0, microsecond=0))
    if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)

def _parse_event_time(fecha, hora):
    dt_date = _parse_date_part(fecha)
    if dt_date is None: return None
    return _parse_time_part(hora, dt_date)

def parse_excel(path: str) -> pd.DataFrame:
    """Lee XLSX detectando la fila de encabezados y normaliza columnas."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe XLSX: {path}")

    df_raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object, engine="openpyxl")
    # Buscar encabezado dentro de primeras 20 filas
    header_row = None
    for i in range(min(20, len(df_raw))):
        cols = df_raw.iloc[i].tolist()
        mapped = _map_headers(cols)
        if len(mapped) >= 4:  # suficiente señal
            header_row = i
            break
    if header_row is None:
        raise ValueError("No se detectó encabezado válido en el XLSX.")

    df = pd.read_excel(path, sheet_name=0, header=header_row, dtype=object, engine="openpyxl")
    df.columns = [c if c is not None else f"col_{i}" for i, c in enumerate(df.columns)]
    mapping = _map_headers(df.columns.tolist())

    if "fecha" not in mapping or "hora" not in mapping:
        raise ValueError(f"Faltan columnas de fecha/hora. Detectadas: {mapping}")

    # Renombrar a canónico si existen
    rename_dict = {mapping[k]: k for k in mapping}
    df = df.rename(columns=rename_dict)

    # Normalizaciones numéricas
    if "lat" in df:  df["lat"]  = df["lat"].apply(_to_float_dot)
    if "lon" in df:  df["lon"]  = df["lon"].apply(_to_float_dot)
    if "prof" in df: df["prof"] = df["prof"].apply(_to_float_dot)
    if "mag" in df:  df["mag"]  = df["mag"].apply(_to_float_dot)

    # Fecha/hora → UTC
    df["event_time_utc"] = df.apply(lambda r: _parse_event_time(r.get("fecha"), r.get("hora")), axis=1)
    df = df[~df["event_time_utc"].isna()].copy()

    # Sanidad geográfica
    if "lat" in df: df = df[df["lat"].between(-90, 90, inclusive="both")]
    if "lon" in df: df = df[df["lon"].between(-180, 180, inclusive="both")]

    # Selección final
    keep = ["event_time_utc"]
    for c in ("lat","lon","prof","mag"):
        if c in df: keep.append(c)
    df = df[keep].drop_duplicates()

    log(f"Archivo leído correctamente: {len(df)} filas útiles tras limpieza.")
    return df

def build_rows(df: pd.DataFrame) -> list[dict]:
    """Convierte DataFrame en dicts listos para inserción."""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "event_time_utc": r["event_time_utc"],
            "lat": float(r["lat"]) if "lat" in r and not pd.isna(r["lat"]) else None,
            "lon": float(r["lon"]) if "lon" in r and not pd.isna(r["lon"]) else None,
            "depth_km": float(r["prof"]) if "prof" in r and not pd.isna(r["prof"]) else None,
            "magnitude": float(r["mag"]) if "mag" in r and not pd.isna(r["mag"]) else None,
            "mag_type": None,
            "place": None,
            "source": "IGP",
            "raw": json.dumps({k: (None if pd.isna(v) else str(v)) for k, v in r.items()}, ensure_ascii=False)
        })
    return rows

def insert_db(rows: list[dict]) -> None:
    """Inserción con ON CONFLICT DO NOTHING (compatible con tu índice único)."""
    if not rows:
        log("No hay registros válidos para insertar.")
        return
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO public.events_clean
                (event_time_utc, lat, lon, depth_km, magnitude, mag_type, place, source, raw)
            VALUES
                (:event_time_utc, :lat, :lon, :depth_km, :magnitude, :mag_type, :place, :source, CAST(:raw AS jsonb))
            ON CONFLICT (
                event_time_utc,
                ROUND(lat::numeric, 3),
                ROUND(lon::numeric, 3),
                ROUND(COALESCE(magnitude, 0)::numeric, 1)
            ) DO NOTHING;
        """), rows)
    log(f"Insertadas {len(rows)} filas nuevas (sin duplicados).")

def main() -> None:
    log("Inicio de carga a la base de datos (IGP)")
    try:
        df = parse_excel(PATH_XLSX)
        if df.empty:
            log("Archivo leído pero sin filas útiles. Se omite inserción.")
            return
        rows = build_rows(df)
        insert_db(rows)
        log("Carga completada correctamente.")
    except Exception as e:
        log(f"Error durante la carga: {e}")
