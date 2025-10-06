FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/app /app/app
EXPOSE 8000
