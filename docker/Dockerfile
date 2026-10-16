# One image for the API, the scheduler and one-off training jobs.
FROM python:3.12.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/* && git config --global --add safe.directory /app
WORKDIR /app

# requirements.txt is compiled for this image (Linux, CPU-only torch from the PyTorch CPU index): `make lock`.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install --no-deps -e .

EXPOSE 8000
CMD ["bl", "serve", "--port", "8000"]
