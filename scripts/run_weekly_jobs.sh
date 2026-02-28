#!/bin/bash

# Log explícito para cron
LOG="/home/reyestineopalermo/quake-forecast-starter/artifacts/etl_logs/system_unified.log"

echo "========== CRON START $(date) ==========" >> $LOG

cd /home/reyestineopalermo/quake-forecast-starter || exit 1

# Evaluación histórica
/usr/bin/docker compose exec -T scheduler \
  python -m scheduler.jobs.evaluate_recent >> $LOG 2>&1

# Pronóstico semanal
/usr/bin/docker compose exec -T scheduler \
  python -m scheduler.jobs.forecast_predict >> $LOG 2>&1

echo "========== CRON END $(date) ==========" >> $LOG
