-- Query para alimentar la tabla de prediction_cells

/* La siguiente sentencia SQL genera un grid nacional regular de celdas cuadradas de 10 km × 10 km aproximadamente, que se utilizarán para la predicción sísmica. Cada celda se almacena en la tabla prediction_cells con:

Su geometría en WGS84 (EPSG:4326)

Su centroide

El tamaño de la celda en kilómetros

Este grid es esencial para el modelo LSTM, ya que cada celda representa una unidad espacial independiente para el cálculo de probabilidades.*/


WITH bounds AS (
    SELECT 
        ST_Transform(
            ST_MakeEnvelope(-82.0, -19.0, -68.0, 0.0, 4326),
            3857
        ) AS geom_3857,
        10000::numeric AS cell_m
),
grid AS (
    SELECT
        ST_Transform(
            ST_MakeEnvelope(x, y, x + cell_m, y + cell_m, 3857),
            4326
        ) AS geom_4326,
        (cell_m / 1000.0) AS cell_km
    FROM bounds,
         generate_series(
             floor(ST_XMin(geom_3857))::bigint,
             floor(ST_XMax(geom_3857))::bigint - cell_m::bigint,
             cell_m::bigint
         ) AS x,
         generate_series(
             floor(ST_YMin(geom_3857))::bigint,
             floor(ST_YMax(geom_3857))::bigint - cell_m::bigint,
             cell_m::bigint
         ) AS y
)
INSERT INTO public.prediction_cells (geom, centroid, cell_km)
SELECT 
    geom_4326,
    ST_PointOnSurface(geom_4326),
    cell_km
FROM grid;