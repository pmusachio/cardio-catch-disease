#!/bin/bash
set -e

COMMAND=${1:-api}

# Train model if it doesn't exist yet
if [ ! -f /app/models/model.joblib ]; then
    echo "No model found — training now..."
    PYTHONPATH=/app/src python -m cardio_catch_disease.cli train
fi

if [ "$COMMAND" = "api" ]; then
    exec uvicorn cardio_catch_disease.api:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 2

elif [ "$COMMAND" = "streamlit" ]; then
    exec streamlit run /app/app/streamlit_app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true

else
    echo "Unknown command: $COMMAND"
    echo "Usage: docker run <image> [api|streamlit]"
    exit 1
fi
