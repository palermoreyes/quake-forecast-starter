# src/scheduler/Dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TZ=America/Lima

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copia todo el paquete scheduler (cron + jobs)
COPY . /app/scheduler

CMD ["python", "-m", "scheduler.cron_loop"]