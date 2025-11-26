"""
fetch_igp_load.py
-----------------
Carga de datos a Postgres desde el XLSX del IGP.
**MEJORADO V3:**
1. Genera nombres de lugar CIENTÍFICOS ("A 15 km al SO de...") igual que el predictor.
2. Realiza UPSERT para corregir datos existentes.
3. Mantenimiento automático de nulos.
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
import reverse_geocoder as rg 

from .common import log, get_engine

PATH_XLSX = os.getenv("IGP_HIST_LOCAL", "/app/data/igp/landing/igp-datos-sismicos.xlsx")

# Mapeo de columnas
REQ_COLS_CANON = {
    "fecha": ["fecha utc", "fecha", "fecha (utc)", "date utc", "fecha_utc"],
    "hora":  ["hora utc", "hora", "hora (utc)", "time utc", "hora_utc"],
    "lat":   ["latitud (º)", "latitud", "lat", "latitude"],
    "lon":   ["longitud (º)", "longitud", "lon", "longitude"],
    "prof":  ["profundidad (km)", "profundidad", "depth", "depth (km)"],
    "mag":   ["magnitud (m)", "magnitud", "magnitude", "mag", "ml", "mw"],
    "ref":   ["referencia", "lugar", "ubicacion", "region", "reference"]
}

# --- UTILIDADES GEOGRÁFICAS (Compartidas con Forecast) ---

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0 
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_cardinal_direction(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon1 - lon2)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dLon) * math.cos(lat1)
    x = math.cos(lat2) * math.sin(lat1) - math.sin(lat2) * math.cos(lat1) * math.cos(dLon)
    bearing = math.degrees(math.atan2(y, x))
    bearing = (bearing + 360) % 360
    dirs = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    return dirs[round(bearing / 45) % 8]

def format_location(event_lat, event_lon, geo_info):
    """Genera: 'A 15 km al SO de Mala, Lima'"""
    city_name = geo_info.get('name', 'Zona')
    region = geo_info.get('admin1', '')
    
    # Coordenadas de la ciudad de referencia (RG devuelve lat/lon como strings)
    city_lat = float(geo_info['lat'])
    city_lon = float(geo_info['lon'])
    
    dist = haversine_km(event_lat, event_lon, city_lat, city_lon)
    
    if dist < 5:
        return f"En {city_name}, {region}"
    
    direction = get_cardinal_direction(event_lat, event_lon, city_lat, city_lon)
    return f"A {int(dist)} km al {direction} de {city_name}, {region}"

# --- PARSING EXCEL (Igual que antes) ---

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
        for i, c in enumerate(norm):
            if c in aliases: found = cols[i]; break
        if not found:
            for i, c in enumerate(norm):
                if any(a in c for a in aliases): found = cols[i]; break
        if found: mapping[key] = found
    return mapping

def _to_float_dot(x):
    import re, numpy as np
    from pandas import isna
    if isna(x): return np.nan
    s = str(x).replace("\u200b","").replace("\ufeff","").strip()
    s = s.replace("\xa0"," ").replace("\u202f"," ")
    s = s.replace(",", "")
    try: return float(s)
    except: return np.nan

def _parse_date_part(fecha):
    if isinstance(fecha, (int, float)) and not pd.isna(fecha):
        base = datetime(1899, 12, 30, tzinfo=timezone.utc)
        return base + pd.to_timedelta(float(fecha), unit="D")
    s = _strip(fecha)
    if not s: return None
    s_norm = re.sub(r"[./]", "-", s).strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s_norm)
    if m:
        y, mth, d = map(int, m.groups())
        return datetime(y, mth, d, tzinfo=timezone.utc)
    dt_ = dparser.parse(s, fuzzy=True, dayfirst=True)
    if dt_.tzinfo is None: dt_ = dt_.replace(tzinfo=timezone.utc)
    return dt_.astimezone(timezone.utc)

def _parse_time_part(hora, base_dt):
    if base_dt is None: return None
    if hora is None or pd.isna(hora): return base_dt
    if isinstance(hora, (int, float)):
        frac = float(hora) % 1.0
        return base_dt + timedelta(seconds=round(frac * 86400))
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe XLSX: {path}")

    df_raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object, engine="openpyxl")
    header_row = None
    for i in range(min(20, len(df_raw))):
        cols = df_raw.iloc[i].tolist()
        mapped = _map_headers(cols)
        if len(mapped) >= 4: header_row = i; break
    if header_row is None: raise ValueError("No se detectó encabezado válido.")

    df = pd.read_excel(path, sheet_name=0, header=header_row, dtype=object, engine="openpyxl")
    df.columns = [c if c is not None else f"col_{i}" for i, c in enumerate(df.columns)]
    mapping = _map_headers(df.columns.tolist())

    if "fecha" not in mapping: raise ValueError(f"Faltan columnas clave. Detectadas: {mapping}")

    rename_dict = {mapping[k]: k for k in mapping}
    df = df.rename(columns=rename_dict)

    if "lat" in df: df["lat"] = df["lat"].apply(_to_float_dot)
    if "lon" in df: df["lon"] = df["lon"].apply(_to_float_dot)
    if "prof" in df: df["prof"] = df["prof"].apply(_to_float_dot)
    if "mag" in df: df["mag"] = df["mag"].apply(_to_float_dot)
    
    if "ref" in df:
        df["ref"] = df["ref"].astype(str).str.strip().replace('nan', None)

    df["event_time_utc"] = df.apply(lambda r: _parse_event_time(r.get("fecha"), r.get("hora")), axis=1)
    df = df[~df["event_time_utc"].isna()].copy()

    if "lat" in df: df = df[df["lat"].between(-90, 90, inclusive="both")]
    if "lon" in df: df = df[df["lon"].between(-180, 180, inclusive="both")]

    keep = ["event_time_utc"]
    for c in ("lat","lon","prof","mag", "ref"):
        if c in df: keep.append(c)
    
    df = df[keep].drop_duplicates()
    log(f"Datos leídos: {len(df)} filas.")
    return df

# --- CONSTRUCCIÓN Y CARGA ---

def build_rows(df: pd.DataFrame) -> list[dict]:
    """
    Prepara los datos y calcula las ciudades (Geocoding CIENTÍFICO).
    """
    log("Calculando ubicaciones detalladas (Reverse Geocoding + Math)...")
    coords_to_search = list(zip(df['lat'], df['lon']))
    geo_results = rg.search(coords_to_search)
    
    rows = []
    for i, r in df.iterrows():
        place_val = r.get("ref")
        
        # Si no hay referencia (o si queremos sobrescribirla para estandarizar), usamos la calculada
        if pd.isna(place_val):
            # --- AQUI ESTA EL CAMBIO V3 ---
            # Usamos la función format_location para obtener "A X km..."
            place_val = format_location(r['lat'], r['lon'], geo_results[i])

        rows.append({
            "event_time_utc": r["event_time_utc"],
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "depth_km": float(r["prof"]) if "prof" in r else None,
            "magnitude": float(r["mag"]) if "mag" in r else None,
            "mag_type": None,
            "place": place_val, 
            "source": "IGP",
            "raw": json.dumps({k: (None if pd.isna(v) else str(v)) for k, v in r.items()}, ensure_ascii=False)
        })
    return rows

def insert_db(rows: list[dict]) -> None:
    if not rows: return
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO public.events_clean
                (event_time_utc, lat, lon, depth_km, magnitude, mag_type, place, source, raw)
            VALUES
                (:event_time_utc, :lat, :lon, :depth_km, :magnitude, :mag_type, :place, :source, CAST(:raw AS jsonb))
            ON CONFLICT (event_time_utc, ROUND(lat::numeric, 3), ROUND(lon::numeric, 3), ROUND(COALESCE(magnitude, 0)::numeric, 1)) 
            DO UPDATE SET place = EXCLUDED.place; 
        """), rows)
    log(f"Procesadas {len(rows)} filas (Insertadas/Actualizadas).")

def run_maintenance_fix_nulls():
    """Repara registros antiguos con formato científico."""
    log("Mantenimiento: Buscando lugares NULL o simples...")
    eng = get_engine()
    with eng.begin() as conn:
        nulls = conn.execute(text("SELECT id, lat, lon FROM public.events_clean WHERE place IS NULL LIMIT 1000")).fetchall()
        
        if not nulls:
            log("Mantenimiento: Todo limpio.")
            return

        log(f"Reparando {len(nulls)} registros...")
        coords = [(r.lat, r.lon) for r in nulls]
        geo_results = rg.search(coords)
        
        update_batch = []
        for i, r in enumerate(nulls):
            place_name = format_location(r.lat, r.lon, geo_results[i])
            update_batch.append({"p_id": r.id, "p_place": place_name})
            
        conn.execute(text("UPDATE public.events_clean SET place = :p_place WHERE id = :p_id"), update_batch)
        log("Mantenimiento completado.")

def main() -> None:
    log("Inicio de ciclo ETL IGP V3 (Carga + Geocoding Avanzado)")
    try:
        df = parse_excel(PATH_XLSX)
        if not df.empty:
            rows = build_rows(df)
            insert_db(rows)
            log("Carga de nuevos datos completada.")
        
        run_maintenance_fix_nulls()
        log("Ciclo ETL finalizado con éxito.")
        
    except Exception as e:
        log(f"Error crítico en ETL: {e}")