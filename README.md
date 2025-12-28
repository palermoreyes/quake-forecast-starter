Este repositorio contiene el desarrollo completo end-to-end del proyecto de tesis:

“MODELO DE REDES NEURONALES PARA PRONOSTICAR EVENTOS SÍSMICOS EN EL PERÚ, 2026”

El sistema integra procesamiento de datos históricos del IGP, modelos de Deep Learning (LSTM), validación científica, API REST y visualización web, con énfasis en rigor académico, automatización y evaluación realista de predicciones sísmicas.

📌 Objetivo General

Desarrollar y validar un modelo basado en redes neuronales recurrentes (LSTM) que permita pronosticar la probabilidad de ocurrencia de eventos sísmicos en el territorio peruano, utilizando datos históricos y un enfoque cuantitativo–experimental.

🎯 Alcance del Proyecto

Pronóstico probabilístico, no determinista.

Ventana temporal: 7 días futuros.

Umbral sísmico analizado: magnitud ≥ 4.0 Mw.

Cobertura espacial: todo el Perú, mediante una grilla regular de celdas geográficas.

Enfoque: aplicado, predictivo, cuantitativo y experimental.

🧠 Arquitectura General
ETL IGP → PostgreSQL + PostGIS → LSTM → Inferencia Batch
                                ↓
                       Validación Real (Backtesting)
                                ↓
                       API REST (FastAPI)
                                ↓
                     Portal Web (Mapa + Ranking)

🧱 Componentes del Sistema
1️⃣ ETL de Datos Sísmicos

Fuente: Instituto Geofísico del Perú (IGP).

Descarga automática cada hora.

Limpieza, validación e inserción incremental.

Datos desde 1960 hasta la actualidad.

Tabla principal: events_clean

2️⃣ Base de Datos Geoespacial

PostgreSQL 15 + PostGIS 3.4

Grilla nacional de celdas (~10x10 km).

Operaciones espaciales con precisión geográfica real (Haversine / Geography).

3️⃣ Modelo de Deep Learning

Framework: TensorFlow / Keras

Tipo: Bi-LSTM

Ventana de entrada: 30 días

Horizonte de predicción: 7 días

Salida: Probabilidad de ocurrencia sísmica por celda.

📌 Modelo final seleccionado:
LSTM_V3.3.1_Hybrid
(Elegido por su mejor desempeño real en validación histórica y operativa).

4️⃣ Evaluación Científica

Se implementaron múltiples estrategias de validación:

Backtesting sobre dataset de prueba.

Matriz de confusión, Precision, Recall y F1-Score.

ROC-AUC.

Validación en mundo real:

Comparación semanal de predicciones vs sismos reales.

Tolerancia espacial configurable (ej. 100 km).

Registro histórico acumulativo.

Tabla clave: validation_realworld

5️⃣ API REST (FastAPI)

Endpoints principales:

GET /forecast/status → Estado del último pronóstico.

GET /forecast/latest → GeoJSON con probabilidades.

GET /forecast/topk → Ranking Top-K de celdas críticas.

6️⃣ Portal Web

Stack: HTML + CSS + JavaScript + Leaflet

Funciones:

Mapa interactivo de probabilidades.

Ranking dinámico Top-K.

Visualización pública y académica.

Acceso exclusivamente por HTTPS.

🌐 Producción: https://proximosismo.org

⏱️ Automatización

El sistema opera de forma autónoma mediante cron jobs:

Proceso	Frecuencia
ETL IGP	Cada hora (minuto 05)
Pronóstico semanal	Lunes 00:15 (hora Perú)
Auditoría de predicción	Lunes 00:10
🔐 Seguridad

HTTPS forzado (TLS / Let’s Encrypt).

Base de datos no expuesta públicamente.

Acceso analítico vía túneles SSH.

Cumplimiento de buenas prácticas de despliegue.

📊 Resultados Relevantes

Accuracy elevada (no usada como métrica principal).

Optimización enfocada en Recall y Precision.

Validación real histórica con trazabilidad.

Evidencia cuantitativa para defensa de tesis.

Los resultados consolidados se encuentran en:

Resultados_del_modelo.xlsx

INFORME TÉCNICO CONSOLIDADO.docx

🧪 Metodología de Investigación

Tipo: Aplicada

Nivel: Predictivo

Enfoque: Cuantitativo

Diseño: Experimental

📂 Estructura del Repositorio
.
├── src/
│   ├── api/                # FastAPI
│   └── scheduler/          # ETL, entrenamiento, inferencia, evaluación
├── docker/
│   ├── nginx.conf
│   └── postgres/
├── artifacts/
│   ├── models/             # Modelos .keras
│   └── etl_logs/
├── data/
├── docker-compose.yml
├── README.md

🚀 Reproducibilidad

El proyecto puede ejecutarse íntegramente mediante:

docker compose up -d --build

📚 Consideraciones Académicas

No se afirma predicción determinista de sismos.

Se trabaja con probabilidades y validación empírica.

Se evita data leakage.

Se reportan limitaciones y alcances reales.


👤 Autores

Palermo Reyes, Paulo Arce
Proyecto de Tesis – Ingeniería de Sistemas
Universidad Privada del Norte
Perú 🇵🇪