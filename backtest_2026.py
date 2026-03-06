# =============================================================================
# backtest_2026.py  —  Backtesting semanal universal (V3.3.1 y V3.6)
# =============================================================================
#
# Detecta automáticamente la versión del modelo activo:
#   - V3.6  : tensor de 6 features (count, binary, log_energy,
#               max_mag, avg_depth, days_since) + norm_params
#   - Legacy: tensor de 2 features (count_norm, binary) + norm_means/norm_stds
#
# VENTANAS (lunes → domingo):
#   31/12/2025  01/01 → 04/01  predice + evalúa  (4 días, ventana especial)
#   04/01/2026  05/01 → 11/01  predice + evalúa
#   11/01/2026  12/01 → 18/01  predice + evalúa
#   ...
#   22/02/2026  23/02 → 01/03  predice + evalúa
#   01/03/2026  02/03 → 08/03  SOLO predice (semana actual incompleta)
#
# SIEMPRE top-25 exactos — min_prob_alert no filtra predicciones.
#
# USO:
#   docker cp backtest_2026.py \
#       quake_sched:/app/scheduler/jobs/backtest_2026.py
#   docker compose exec scheduler \
#       python -m scheduler.jobs.backtest_2026
# =============================================================================

import os
import json
import math
import datetime as dt
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import tensorflow as tf
import reverse_geocoder as rg
from dataclasses import dataclass
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class Config:
    db_host:       str   = os.getenv("DB_HOST",     "db")
    db_port:       int   = int(os.getenv("DB_PORT", "5432"))
    db_name:       str   = os.getenv("DB_NAME",     "quake")
    db_user:       str   = os.getenv("DB_USER",     "quake")
    db_password:   str   = os.getenv("DB_PASSWORD", "changeme")
    models_dir:    str   = os.getenv("MODELS_DIR",  "/app/artifacts/models")
    topk_k:        int   = 100
    lookback_days: int   = 90


TOLERANCIA_KM  = 100
UMBRAL_OFICIAL = 4.0
UMBRAL_PARCIAL = 3.0


def get_conn(cfg: Config):
    return psycopg2.connect(
        host=cfg.db_host, port=cfg.db_port,
        dbname=cfg.db_name, user=cfg.db_user,
        password=cfg.db_password,
    )


def get_engine(cfg: Config):
    return create_engine(
        f"postgresql+psycopg2://{cfg.db_user}:{cfg.db_password}"
        f"@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
    )


# =============================================================================
# VENTANAS LUNES → DOMINGO
# =============================================================================

def generate_windows(today: dt.date):
    """
    Ventana 1 especial: 01/01 (jueves) → 04/01 (domingo)
    Resto: siempre lunes → domingo

    Retorna:
      windows_eval   : list[(data_hasta, pred_start, pred_end)]
      window_predict : tuple(data_hasta, pred_start, pred_end) o None
    """
    windows_eval   = []
    window_predict = None

    # Ventana 1 especial (01/01 jueves → 04/01 domingo)
    w1_start = dt.date(2026, 1, 1)
    w1_end   = dt.date(2026, 1, 4)
    windows_eval.append((w1_start - dt.timedelta(days=1), w1_start, w1_end))

    # Ventanas siguientes: lunes → domingo
    pred_start = w1_end + dt.timedelta(days=1)  # 05/01 lunes

    while True:
        pred_end   = pred_start + dt.timedelta(days=6)   # domingo
        data_hasta = pred_start - dt.timedelta(days=1)   # domingo anterior

        if pred_end < today:
            windows_eval.append((data_hasta, pred_start, pred_end))
        elif pred_start <= today:
            window_predict = (data_hasta, pred_start, pred_end)
            break
        else:
            break

        pred_start = pred_end + dt.timedelta(days=1)

    return windows_eval, window_predict


# =============================================================================
# UTILIDADES GEOGRÁFICAS
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dLon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_cardinal(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon1 - lon2)
    la1, la2 = math.radians(lat1), math.radians(lat2)
    y = math.sin(dLon) * math.cos(la1)
    x = (math.cos(la2) * math.sin(la1)
         - math.sin(la2) * math.cos(la1) * math.cos(dLon))
    b = (math.degrees(math.atan2(y, x)) + 360) % 360
    return ["N", "NE", "E", "SE", "S", "SO", "O", "NO"][round(b / 45) % 8]


def format_location(lat, lon, geo):
    city, region = geo.get("name", "Zona"), geo.get("admin1", "")
    clat, clon   = float(geo["lat"]), float(geo["lon"])
    dist = haversine_km(lat, lon, clat, clon)
    if dist < 5:
        return f"En {city}, {region}".strip(", ")
    return (f"A {int(dist)} km al {get_cardinal(lat,lon,clat,clon)} "
            f"de {city}, {region}").strip(", ")


# =============================================================================
# DAYS_SINCE — idéntico a forecast_lstm.py (NO modificar)
# =============================================================================

def _compute_days_since_last(binary_arr: np.ndarray) -> np.ndarray:
    n_days, n_cells = binary_arr.shape
    result  = np.zeros((n_days, n_cells), dtype=np.float32)
    counter = np.full(n_cells, 365.0, dtype=np.float32)
    for i in range(n_days):
        active  = binary_arr[i] > 0
        counter = np.where(active, 0.0, np.minimum(counter + 1.0, 365.0))
        result[i] = counter
    return result


# =============================================================================
# TENSOR V3.6 — 6 features hasta data_hasta INCLUSIVE
# =============================================================================

def build_tensor_v36(cfg: Config, mag_min: float, window_days: int,
                     cell_ids: list, norm_params: dict,
                     data_hasta: dt.date) -> np.ndarray:
    cutoff = pd.Timestamp(data_hasta) + pd.Timedelta(days=1)  # exclusive
    start  = cutoff - pd.Timedelta(days=cfg.lookback_days)

    query = """
        SELECT
            c.cell_id,
            date_trunc('day', e.event_time_utc)::date  AS date,
            COUNT(*)                                    AS event_count,
            MAX(e.magnitude)                            AS max_mag,
            AVG(e.depth_km)                             AS avg_depth,
            SUM(1.5 * e.magnitude)                      AS log_energy
        FROM public.prediction_cells c
        LEFT JOIN public.events_clean e
            ON ST_Intersects(c.geom, e.geom)
           AND e.magnitude      >= %(mag_min)s
           AND e.event_time_utc >= %(start)s
           AND e.event_time_utc <  %(cutoff)s
        WHERE c.cell_id = ANY(%(cell_ids)s)
        GROUP BY c.cell_id, date
    """
    with get_conn(cfg) as conn:
        df = pd.read_sql_query(query, conn, params={
            "mag_min": mag_min, "start": start.date(),
            "cutoff": cutoff.date(), "cell_ids": cell_ids,
        })

    df["date"] = pd.to_datetime(df["date"])
    for col in ["event_count", "max_mag", "avg_depth", "log_energy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    idx = pd.date_range(start=start.date(), end=data_hasta, freq="D")

    def _pivot(col, fill=0.0):
        if df.empty:
            return pd.DataFrame(fill, index=idx,
                                columns=cell_ids, dtype=np.float32)
        w = (df.pivot(index="date", columns="cell_id", values=col)
               .reindex(index=idx, columns=cell_ids, fill_value=fill)
               .astype(np.float32))
        return w.replace([np.inf, -np.inf], fill).fillna(fill)

    wide_count      = _pivot("event_count")
    wide_maxmag     = _pivot("max_mag")
    wide_log_energy = _pivot("log_energy")

    wd_raw = (df.pivot(index="date", columns="cell_id", values="avg_depth")
                .reindex(index=idx, columns=cell_ids).astype(np.float32)
                .replace([np.inf, -np.inf], np.nan)
              ) if not df.empty else pd.DataFrame(
                  np.nan, index=idx, columns=cell_ids, dtype=np.float32)
    wide_depth = wd_raw.T.fillna(wd_raw.median(axis=0)).T.fillna(50.0).astype(np.float32)

    binary_arr      = (wide_count.values > 0).astype(np.float32)
    wide_days_since = pd.DataFrame(
        _compute_days_since_last(binary_arr), index=idx, columns=cell_ids
    ).astype(np.float32)

    def _lw(mat): return mat.iloc[-window_days:].values.T  # [C, W]

    arr_count      = _lw(wide_count)
    arr_maxmag     = _lw(wide_maxmag)
    arr_log_energy = _lw(wide_log_energy)
    arr_depth      = _lw(wide_depth)
    arr_days_since = _lw(wide_days_since)
    arr_binary     = (arr_count > 0).astype(np.float32)

    def _norm(arr, name):
        p = norm_params[name]
        m = np.array(p["mean"], dtype=np.float32)
        s = np.array(p["std"],  dtype=np.float32)
        s = np.where((s > 0) & np.isfinite(s), s, 1.0)
        return np.clip(((arr - m[:, None]) / s[:, None]), -8.0, 8.0).astype(np.float32)

    return np.stack([
        _norm(arr_count,      "count"),
        arr_binary,
        _norm(arr_log_energy, "log_energy"),
        _norm(arr_maxmag,     "max_mag"),
        _norm(arr_depth,      "avg_depth"),
        _norm(arr_days_since, "days_since"),
    ], axis=-1).astype(np.float32)


# =============================================================================
# TENSOR LEGACY — 2 features para V3.3.1 y anteriores
# =============================================================================

def build_tensor_legacy(cfg: Config, mag_min: float, window_days: int,
                         cell_ids: list, norm_means: np.ndarray,
                         norm_stds: np.ndarray, data_hasta: dt.date) -> np.ndarray:
    cutoff = pd.Timestamp(data_hasta) + pd.Timedelta(days=1)
    start  = cutoff - pd.Timedelta(days=cfg.lookback_days)

    query = """
        SELECT c.cell_id,
               date_trunc('day', e.event_time_utc)::date AS date,
               COUNT(e.id) AS count
        FROM public.prediction_cells c
        LEFT JOIN public.events_clean e
               ON ST_Intersects(c.geom, e.geom)
              AND e.magnitude      >= %(mag_min)s
              AND e.event_time_utc >= %(start)s
              AND e.event_time_utc <  %(cutoff)s
        GROUP BY c.cell_id, date
    """
    with get_conn(cfg) as conn:
        df = pd.read_sql_query(query, conn, params={
            "mag_min": mag_min, "start": start.date(), "cutoff": cutoff.date(),
        })

    df["date"] = pd.to_datetime(df["date"])
    idx   = pd.date_range(start=start.date(), end=data_hasta, freq="D")
    pivot = (
        df.pivot(index="date", columns="cell_id", values="count")
        .reindex(index=idx, columns=cell_ids, fill_value=0)
        .astype(np.float32)
    )

    recent    = pivot.iloc[-window_days:].values.T  # [C, W]
    feat_norm = np.clip((recent - norm_means[:, None]) / norm_stds[:, None],
                        -8.0, 8.0).astype(np.float32)
    feat_bin  = (recent > 0).astype(np.float32)
    return np.stack([feat_norm, feat_bin], axis=-1).astype(np.float32)


# =============================================================================
# PREDICCIÓN — guarda en BD, retorna run_id
# =============================================================================

def run_prediction(cfg: Config, model, model_id: int, horizons: list,
                   mag_min: float, cell_ids: list, params: dict,
                   window_days: int, version: str,
                   data_hasta: dt.date,
                   pred_start: dt.date, pred_end: dt.date) -> int:

    # Construir tensor según versión
    if version == "V3.6":
        X = build_tensor_v36(
            cfg, mag_min, window_days, cell_ids,
            params["norm_params"], data_hasta
        )
    else:
        norm_means = np.array(params["norm_means"], dtype=np.float32)
        norm_stds  = np.array(params["norm_stds"],  dtype=np.float32)
        X = build_tensor_legacy(
            cfg, mag_min, window_days, cell_ids,
            norm_means, norm_stds, data_hasta
        )

    probs = model.predict(X, batch_size=2048, verbose=0).flatten()

    horizon_days  = int(horizons[0])
    last_input_ts = pd.Timestamp(data_hasta).to_pydatetime()
    pred_start_dt = pd.Timestamp(pred_start).to_pydatetime()
    pred_end_dt   = pd.Timestamp(pred_end).replace(hour=23, minute=59, second=59)

    df_res             = pd.DataFrame({"cell_id": cell_ids, "prob": probs})
    df_res["rank_pct"] = df_res["prob"].rank(pct=True)

    # SIEMPRE top-25 exactos — sin filtro de umbral previo
    topk = df_res.nlargest(cfg.topk_k, "prob")

    with get_conn(cfg) as conn:
        ids_list = [int(c) for c in topk["cell_id"].tolist()]
        df_geo   = pd.read_sql_query(
            "SELECT cell_id, ST_Y(centroid) AS lat, ST_X(centroid) AS lon "
            "FROM prediction_cells WHERE cell_id = ANY(%s)",
            conn, params=(ids_list,)
        )
        coords    = list(zip(df_geo["lat"], df_geo["lon"]))
        geo_res   = rg.search(coords)
        place_map = {
            int(r.cell_id): format_location(r.lat, r.lon, geo_res[i])
            for i, r in enumerate(df_geo.itertuples(index=False))
        }

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO prediction_run
                   (generated_at, model_id, input_max_time,
                    horizons, mag_min, cell_km, topk_k)
                   VALUES (now(), %s, %s, %s, %s, 10, %s)
                   RETURNING run_id""",
                (model_id, last_input_ts, list(horizons), mag_min, cfg.topk_k)
            )
            run_id = cur.fetchone()[0]

            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO prediction_cell_prob
                   (run_id, cell_id, horizon_days, prob, density, rank_pct)
                   VALUES %s""",
                [(run_id, int(r.cell_id), horizon_days,
                  float(r.prob), float(r.prob), float(r.rank_pct))
                 for r in df_res.itertuples()]
            )

            for rank_idx, row in enumerate(topk.itertuples(), 1):
                cid = int(row.cell_id)
                cur.execute(
                    """INSERT INTO prediction_topk
                       (run_id, horizon_days, rank,
                        t_pred_start, t_pred_end,
                        lat, lon, mag_pred, prob, place, cell_id)
                       SELECT %s,%s,%s,%s,%s,
                              ST_Y(centroid), ST_X(centroid),
                              %s,%s,%s,%s
                       FROM prediction_cells WHERE cell_id=%s""",
                    (run_id, horizon_days, rank_idx,
                     pred_start_dt, pred_end_dt,
                     mag_min + 0.5, float(row.prob),
                     place_map.get(cid, "Zona remota"),
                     cid, cid)
                )
            conn.commit()

    return run_id


# =============================================================================
# EVALUACIÓN — misma lógica que evaluate_recent.py
# =============================================================================

def find_best_match(pred_lat, pred_lon, df_full, df_partial,
                    used_full_ids, used_partial_ids):
    full_cands = [
        (haversine_km(pred_lat, pred_lon, s["lat"], s["lon"]), s)
        for _, s in df_full.iterrows()
        if s["id"] not in used_full_ids
        and haversine_km(pred_lat, pred_lon, s["lat"], s["lon"]) <= TOLERANCIA_KM
    ]
    if full_cands:
        full_cands.sort(key=lambda x: x[0])
        return full_cands[0][1], full_cands[0][0], "full_hit"

    partial_cands = [
        (haversine_km(pred_lat, pred_lon, s["lat"], s["lon"]), s)
        for _, s in df_partial.iterrows()
        if s["id"] not in used_partial_ids
        and haversine_km(pred_lat, pred_lon, s["lat"], s["lon"]) <= TOLERANCIA_KM
    ]
    if partial_cands:
        partial_cands.sort(key=lambda x: x[0])
        return partial_cands[0][1], partial_cands[0][0], "partial_hit"

    return None, None, "no_hit"


def run_evaluation(engine, run_id: int,
                   pred_start: dt.date, pred_end: dt.date) -> dict:
    ps = pd.Timestamp(pred_start)
    pe = pd.Timestamp(pred_end).replace(hour=23, minute=59, second=59)

    with engine.connect() as conn:
        df_preds = pd.read_sql(
            text("SELECT rank, lat, lon, place FROM prediction_topk "
                 "WHERE run_id=:r ORDER BY rank ASC"),
            conn, params={"r": run_id}
        )
        df_real = pd.read_sql(
            text("SELECT id, event_time_utc, lat, lon, magnitude, place "
                 "FROM events_clean "
                 "WHERE event_time_utc BETWEEN :s AND :e AND magnitude >= :m"),
            conn, params={"s": ps, "e": pe, "m": UMBRAL_PARCIAL}
        )

    df_full    = df_real[df_real["magnitude"] >= UMBRAL_OFICIAL].copy()
    df_partial = df_real[(df_real["magnitude"] >= UMBRAL_PARCIAL) &
                          (df_real["magnitude"] <  UMBRAL_OFICIAL)].copy()

    full_hits = partial_hits = 0
    detalle_full    = []
    detalle_partial = []
    used_full_ids    = set()
    used_partial_ids = set()
    trace_rows       = []

    for _, pred in df_preds.iterrows():
        best, dist, hit = find_best_match(
            pred["lat"], pred["lon"],
            df_full, df_partial,
            used_full_ids, used_partial_ids
        )

        if best is not None:
            if hit == "full_hit":
                used_full_ids.add(best["id"])
                full_hits += 1
                detalle_full.append({
                    "rank"    : int(pred["rank"]),
                    "evento"  : best["place"],
                    "magnitud": float(best["magnitude"]),
                    "dist_km" : round(dist, 2),
                })
            else:
                used_partial_ids.add(best["id"])
                partial_hits += 1
                detalle_partial.append({
                    "rank"    : int(pred["rank"]),
                    "evento"  : best["place"],
                    "magnitud": float(best["magnitude"]),
                    "dist_km" : round(dist, 2),
                })

        trace_rows.append({
            "run_id"  : run_id,
            "rank"    : int(pred["rank"]),
            "lat"     : pred["lat"],
            "lon"     : pred["lon"],
            "place"   : pred["place"],
            "ws"      : pred_start,
            "we"      : pred_end,
            "matched" : hit != "no_hit",
            "eid"     : int(best["id"])          if best is not None else None,
            "etime"   : best["event_time_utc"]   if best is not None else None,
            "mag"     : float(best["magnitude"]) if best is not None else None,
            "eplace"  : best["place"]            if best is not None else None,
            "dist"    : round(dist, 2)           if dist is not None else None,
            "tol"     : TOLERANCIA_KM,
            "hit_type": hit,
        })

    total_oficiales = len(df_full)
    total_alertas   = len(df_preds)
    recall    = full_hits / total_oficiales * 100 if total_oficiales else 0.0
    precision = full_hits / total_alertas   * 100 if total_alertas   else 0.0
    f1 = (2 * (precision / 100) * (recall / 100)
          / ((precision / 100) + (recall / 100)) * 100
          if (precision + recall) > 0 else 0.0)

    with engine.begin() as tx:
        if trace_rows:
            tx.execute(text("""
                INSERT INTO prediction_trace (
                    run_id, rank, lat, lon, place,
                    predicted_window_start, predicted_window_end,
                    matched_event, matched_event_id,
                    event_time_utc, event_magnitude, event_place,
                    distance_km, tolerance_km, hit_type
                ) VALUES (
                    :run_id, :rank, :lat, :lon, :place,
                    :ws, :we, :matched, :eid,
                    :etime, :mag, :eplace,
                    :dist, :tol, :hit_type
                )"""), trace_rows)

        tx.execute(text("""
            INSERT INTO validation_realworld (
                run_id, window_start, window_end,
                total_sismos, sismos_detectados, partial_hits,
                recall_pct, precision_pct, f1_pct,
                aciertos_json, partial_hits_json
            ) VALUES (
                :r, :s, :e, :ts, :full, :partial,
                :rec, :prec, :f1, :jf, :jp
            )"""), {
            "r"      : run_id,      "s": pred_start,  "e": pred_end,
            "ts"     : total_oficiales,
            "full"   : full_hits,   "partial": partial_hits,
            "rec"    : round(recall, 4),
            "prec"   : round(precision, 4),
            "f1"     : round(f1, 4),
            "jf"     : json.dumps(detalle_full),
            "jp"     : json.dumps(detalle_partial),
        })

    return {
        "run_id"         : run_id,
        "pred_start"     : pred_start,
        "pred_end"       : pred_end,
        "total_oficiales": total_oficiales,
        "full_hits"      : full_hits,
        "partial_hits"   : partial_hits,
        "recall"         : recall,
        "precision"      : precision,
        "f1"             : f1,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg    = Config()
    today  = dt.datetime.utcnow().date()
    engine = get_engine(cfg)

    print("=" * 65)
    print("BACKTEST UNIVERSAL  —  INICIO 2026 HASTA HOY")
    print(f"Fecha actual (UTC): {today}")
    print("=" * 65)

    # ── Cargar modelo activo ─────────────────────────────────────────────────
    with get_conn(cfg) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT model_id, params_json, horizons, mag_min
                FROM public.model_registry
                WHERE is_active = TRUE
                ORDER BY created_at DESC LIMIT 1
            """)
            row = cur.fetchone()

    if not row:
        print("[ERROR] No hay modelo activo en model_registry.")
        return

    params      = row["params_json"]
    if isinstance(params, str):
        params  = json.loads(params)

    version     = params.get("version", "legacy")
    model_id    = row["model_id"]
    horizons    = row["horizons"]
    mag_min     = float(row["mag_min"])
    cell_ids    = list(params["cell_ids"])
    window_days = int(params.get("window_days", 30))
    model_path  = os.path.join(cfg.models_dir, params["model_path"])

    # Validar parámetros de normalización según versión
    if version == "V3.6":
        if "norm_params" not in params:
            print("[ERROR] V3.6 requiere 'norm_params' en params_json.")
            return
    else:
        if "norm_means" not in params or "norm_stds" not in params:
            print("[ERROR] Modelo legacy requiere 'norm_means' y 'norm_stds'.")
            return

    print(f"Modelo     : ID={model_id} | {params['model_path']}")
    print(f"Versión    : {version}")
    print(f"Features   : {params.get('n_features', 2)} | ventana: {window_days}d")
    print(f"mag_min    : {mag_min}")
    print(f"Top-K      : siempre exactamente {cfg.topk_k}\n")

    print("[BACKTEST] Cargando modelo Keras...")
    model = tf.keras.models.load_model(model_path)
    print("[BACKTEST] Modelo listo.\n")

    # ── Generar ventanas ─────────────────────────────────────────────────────
    windows_eval, window_predict = generate_windows(today)

    print(f"Ventanas evaluables : {len(windows_eval)}")
    print(f"Ventana actual      : "
          f"{window_predict[1] if window_predict else 'ninguna'}\n")

    print(f"{'#':>2}  {'Data hasta':12} {'Pred inicio':12} "
          f"{'Pred fin':12} {'Días':>4}  Acción")
    print("─" * 65)
    for i, (ed, ps, pe) in enumerate(windows_eval, 1):
        dias = (pe - ps).days + 1
        print(f"{i:>2}  {str(ed):12} {str(ps):12} {str(pe):12} "
              f"{dias:>4}d  predecir+evaluar")
    if window_predict:
        ed, ps, pe = window_predict
        dias = (pe - ps).days + 1
        print(f" →  {str(ed):12} {str(ps):12} {str(pe):12} "
              f"{dias:>4}d  solo predecir")
    print()

    # ── Loop predecir + evaluar ───────────────────────────────────────────────
    resultados = []

    for i, (data_hasta, pred_start, pred_end) in enumerate(windows_eval, 1):
        print(f"{'─'*65}")
        print(f"[{i:>2}/{len(windows_eval)}] "
              f"Data hasta {data_hasta} | {pred_start} → {pred_end}")

        run_id = run_prediction(
            cfg, model, model_id, horizons, mag_min,
            cell_ids, params, window_days, version,
            data_hasta, pred_start, pred_end
        )
        print(f"  ✓ Predicción guardada | run_id={run_id}")

        m = run_evaluation(engine, run_id, pred_start, pred_end)
        resultados.append(m)

        print(f"  Sismos Mw≥4.0 : {m['total_oficiales']}")
        print(f"  Full hits     : {m['full_hits']}  "
              f"Partial: {m['partial_hits']}")
        print(f"  Recall={m['recall']:.1f}%  "
              f"Precision={m['precision']:.1f}%  "
              f"F1={m['f1']:.1f}%")

    # ── Semana actual: solo predecir ─────────────────────────────────────────
    if window_predict:
        data_hasta, pred_start, pred_end = window_predict
        print(f"\n{'─'*65}")
        print(f"[SEMANA ACTUAL] Data hasta {data_hasta} | "
              f"{pred_start} → {pred_end}  (sin evaluar)")
        run_id = run_prediction(
            cfg, model, model_id, horizons, mag_min,
            cell_ids, params, window_days, version,
            data_hasta, pred_start, pred_end
        )
        print(f"  ✓ Predicción guardada | run_id={run_id}")
        print(f"  Evaluar el {pred_end + dt.timedelta(days=1)} "
              f"con evaluate_recent.py")

    # ── Resumen acumulado ─────────────────────────────────────────────────────
    if not resultados:
        print("\nNo hay resultados para resumir.")
        return

    print(f"\n{'='*65}")
    print(f"RESUMEN BACKTESTING {version}  —  2026")
    print(f"{'='*65}")
    print(f"{'Ventana':<24} {'Días':>4} {'Sis':>4} {'Hit':>4} "
          f"{'Recall':>8} {'Prec':>8} {'F1':>8}")
    print("─" * 65)

    total_sis = total_hits = total_alertas = 0

    for m in resultados:
        dias    = (m["pred_end"] - m["pred_start"]).days + 1
        ventana = f"{m['pred_start']} → {m['pred_end']}"
        print(f"{ventana:<24} {dias:>4} "
              f"{m['total_oficiales']:>4} {m['full_hits']:>4} "
              f"{m['recall']:>7.1f}% "
              f"{m['precision']:>7.1f}% "
              f"{m['f1']:>7.1f}%")
        total_sis     += m["total_oficiales"]
        total_hits    += m["full_hits"]
        total_alertas += cfg.topk_k

    print("─" * 65)
    recall_g = total_hits / total_sis     * 100 if total_sis     else 0.0
    prec_g   = total_hits / total_alertas * 100 if total_alertas else 0.0
    f1_g     = (2 * (prec_g / 100) * (recall_g / 100)
                / ((prec_g / 100) + (recall_g / 100)) * 100
                if (prec_g + recall_g) > 0 else 0.0)

    print(f"{'TOTAL ACUMULADO':<24} {'':>4} "
          f"{total_sis:>4} {total_hits:>4} "
          f"{recall_g:>7.1f}% {prec_g:>7.1f}% {f1_g:>7.1f}%")

    print(f"\n  Sismos detectados : {total_hits} / {total_sis}")
    print(f"  Recall global     : {recall_g:.1f}%")

    if version == "V3.6":
        baseline_label = "Baseline V3.3.1 (medición anterior)"
        baseline_val   = 70.0
    else:
        baseline_label = "Baseline anterior (con leakage)"
        baseline_val   = 70.0

    print(f"  {baseline_label} : {baseline_val:.1f}%")

    if recall_g > baseline_val:
        print(f"  ✅ {version} SUPERA al baseline  (+{recall_g - baseline_val:.1f}%)")
    elif recall_g >= baseline_val - 1:
        print(f"  ✅ {version} IGUALA al baseline")
    else:
        diff = baseline_val - recall_g
        print(f"  ⚠️  {version} por debajo del baseline  (-{diff:.1f}%)")

    print("=" * 65)


if __name__ == "__main__":
    main()