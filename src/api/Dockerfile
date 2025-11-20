# src/api/Dockerfile

ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

WORKDIR /app

# Instalamos dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el código de src/api a /app
COPY . .

# El comando se sobreescribe en el docker-compose, pero este es el default
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]