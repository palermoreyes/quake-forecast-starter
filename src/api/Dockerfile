# Acepta arg del compose; usa 3.11-slim por defecto si no llega
ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# deps del sistema para psqsl/geo si hicieras queries directas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

# uvicorn lo ejecuta el compose (command:)
