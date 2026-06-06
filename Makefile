.PHONY: install profile train analyze test api app docker-build docker-up docker-down docker-logs all

# ── Local development ───────────────────────────────────────────────────────

install:
	python -m pip install -r requirements.txt

install-app:
	python -m pip install -r requirements-app.txt

profile:
	PYTHONPATH=src python -m cardio_catch_disease.cli profile

train:
	PYTHONPATH=src python -m cardio_catch_disease.cli train

analyze:
	PYTHONPATH=src python -m cardio_catch_disease.cli analyze

test:
	python -m pytest

api:
	PYTHONPATH=src uvicorn cardio_catch_disease.api:app --reload --host 0.0.0.0 --port 8000

app:
	streamlit run app/streamlit_app.py --server.port 8501

# Run the full pipeline: profile → train → test
all: profile train test

# ── Docker ──────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d
	@echo ""
	@echo "  API:       http://localhost:8000"
	@echo "  API docs:  http://localhost:8000/docs"
	@echo "  Streamlit: http://localhost:8501"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
