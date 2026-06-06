FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first to leverage Docker layer caching
COPY requirements.txt requirements-api.txt requirements-app.txt ./

RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt -r requirements-app.txt

# Copy project source
COPY src/ src/
COPY app/ app/
COPY configs/ configs/

# Create directories — dataset and models are mounted via docker-compose volumes
RUN mkdir -p data/raw data/interim data/processed models reports/figures

# Pre-train if no model exists; otherwise use the cached model
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/app/src

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
