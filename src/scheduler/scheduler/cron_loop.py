import os, time
from sqlalchemy import create_engine, text

DBURL = os.getenv("DATABASE_URL")
engine = create_engine(DBURL, pool_pre_ping=True)

def ingest_recent_dummy():
    with engine.begin() as con:
        con.execute(text("""
            INSERT INTO events_clean (t_utc, lat, lon, depth_km, mag, src)
            SELECT NOW(), -12.05, -77.05, 45, 4.2, 'demo'
            WHERE NOT EXISTS (SELECT 1 FROM events_clean WHERE t_utc > NOW() - INTERVAL '5 minutes');
        """))

def write_dummy_forecast():
    with engine.begin() as con:
        con.execute(text("""
            DELETE FROM forecasts WHERE run_at < NOW() - INTERVAL '7 days';
        """))
        con.execute(text("""
            INSERT INTO forecasts (run_at, horizon_days, mag_thr, lat_bin, lon_bin, prob)
            VALUES (NOW(), 7, 4.5, 100, 120, 0.12)
            ON CONFLICT DO NOTHING;
        """))

def main():
    while True:
        try:
            ingest_recent_dummy()
            write_dummy_forecast()
            print("Scheduler tick OK")
        except Exception as e:
            print("Scheduler error:", e)
        time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    main()
