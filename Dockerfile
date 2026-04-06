# Dockerfile — only change is adding server/ copy
FROM python:3.11-slim

LABEL maintainer="ml-debugger-env"
LABEL description="OpenEnv ML Pipeline Debugger Environment"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY models.py        .
COPY environment.py   .
COPY app.py           .
COPY server/          ./server/
COPY tasks/           ./tasks/
COPY data/            ./data/
COPY openenv.yaml     .
COPY pyproject.toml   .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" \
    || exit 1

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]
