CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS events_clean (
  id BIGSERIAL PRIMARY KEY,
  t_utc TIMESTAMPTZ NOT NULL,
  lat DOUBLE PRECISION NOT NULL,
  lon DOUBLE PRECISION NOT NULL,
  depth_km REAL,
  mag REAL,
  src TEXT,
  geom GEOGRAPHY(POINT) GENERATED ALWAYS AS (
    ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography
  ) STORED
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events_clean (t_utc);
CREATE INDEX IF NOT EXISTS idx_events_geom ON events_clean USING GIST ((geom::geometry));
CREATE INDEX IF NOT EXISTS idx_events_mag  ON events_clean (mag);

CREATE TABLE IF NOT EXISTS grid_cells (
  lat_bin INT,
  lon_bin INT,
  bbox geometry(POLYGON,4326) NOT NULL,
  PRIMARY KEY (lat_bin, lon_bin)
);
CREATE INDEX IF NOT EXISTS idx_grid_bbox ON grid_cells USING GIST (bbox);

CREATE TABLE IF NOT EXISTS features_daily (
  date DATE NOT NULL,
  lat_bin INT NOT NULL,
  lon_bin INT NOT NULL,
  count_7d REAL, count_30d REAL, count_90d REAL,
  rate_smooth REAL, a_value REAL, b_value REAL,
  time_since_m4 REAL,
  PRIMARY KEY (date, lat_bin, lon_bin)
);

CREATE TABLE IF NOT EXISTS forecasts (
  run_at TIMESTAMPTZ NOT NULL,
  horizon_days INT NOT NULL,
  mag_thr REAL NOT NULL,
  lat_bin INT NOT NULL,
  lon_bin INT NOT NULL,
  prob REAL NOT NULL,
  PRIMARY KEY (run_at, horizon_days, mag_thr, lat_bin, lon_bin)
);
CREATE INDEX IF NOT EXISTS idx_fcst_meta ON forecasts (run_at, horizon_days, mag_thr);
