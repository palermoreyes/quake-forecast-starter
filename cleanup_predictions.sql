-- =============================================================
-- 1_cleanup_predictions.sql
-- SOLO ejecutar en entorno LOCAL
-- =============================================================
-- docker compose exec db psql -U quake -d quake \
--   -c "$(cat 1_cleanup_predictions.sql)"
-- =============================================================

BEGIN;

TRUNCATE TABLE validation_realworld  CASCADE;
TRUNCATE TABLE prediction_trace      CASCADE;
TRUNCATE TABLE prediction_topk       CASCADE;
TRUNCATE TABLE prediction_cell_prob  CASCADE;
TRUNCATE TABLE prediction_run        CASCADE;

ALTER SEQUENCE prediction_run_run_id_seq RESTART WITH 1;

SELECT setval(
    'model_registry_model_id_seq',
    (SELECT MAX(model_id) FROM public.model_registry)
);

COMMIT;

SELECT
    'Limpieza completada' AS status,
    (SELECT COUNT(*) FROM prediction_run)       AS prediction_run,
    (SELECT COUNT(*) FROM prediction_topk)      AS prediction_topk,
    (SELECT COUNT(*) FROM validation_realworld) AS validation_realworld;