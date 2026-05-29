.PHONY: install migrate migrate-up init-db init-weaviate start worker start-all stop frontend-install frontend-dev frontend-build

install:
	pip install -e ".[dev]"

migrate:
	alembic revision --autogenerate -m "$(m)"

migrate-up:
	alembic upgrade head

init-db:
	python scripts/init_db.py

init-weaviate:
	python scripts/init_weaviate.py

start:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	celery -A app.workers.celery_app worker --loglevel=info

start-all:
	@echo "Starting API and worker..."
	@make start &
	@make worker

stop:
	pkill -f uvicorn || true
	pkill -f celery || true

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build
