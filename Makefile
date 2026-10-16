# Everything runs in Docker; only `make` and Docker Desktop (>= 8 GB memory) are needed on the host.
COMPOSE := docker compose
RUN     := $(COMPOSE) run --rm trainer
API     := http://localhost:8000
AS_OF   ?=

lock: ## re-pin requirements.txt for the Linux image (CPU-only torch)
	docker run --rm -v "$$PWD":/w -w /w python:3.12.13-slim sh -c "pip install -q pip-tools==7.6.2 && pip-compile -q --strip-extras --extra-index-url https://download.pytorch.org/whl/cpu requirements.in -o requirements.txt"

.PHONY: lock up down logs build load-data train-test train predict rollback promote status load-test test lint demo

up: ## start MLflow (localhost:5001), the API (localhost:8000) and the weekly scheduler
	$(COMPOSE) up -d --build --wait mlflow api scheduler

down: ## stop the stack (volumes are kept; `docker compose down -v` wipes them)
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api scheduler

build:
	$(COMPOSE) build

load-data: ## data/bl_full_data.csv -> training table
	$(RUN) bl load-data

train-test: ## researcher train/test evaluation -> MLflow metrics + log
	$(RUN) bl train --mode train_test

train: ## production training -> register -> gate -> promote -> API hot reload (AS_OF=YYYY-MM-DD optional)
	$(RUN) bl train --mode production $(if $(AS_OF),--as-of $(AS_OF),)

predict: ## rank brands for the example user
	@curl -s -i -X POST $(API)/v1/predict -H "content-type: application/json" -d @load_test/example.json \
	  | grep -iE "^(HTTP|x-model|x-payout|x-degraded|x-cache|x-empty)|^\{"

rollback: ## move @champion back to the previous version (API reloads without restart)
	$(RUN) bl rollback

promote: ## promote the current @challenger
	$(RUN) bl promote

status:
	$(RUN) bl status
	@curl -s $(API)/version; echo

load-test: ## realistic load against the running API (backend as served), results -> docs/measurements
	$(RUN) python -m load_test.run_scenario --host http://api:8000 --name "docker · realistic (5 users, 1–3 s think time)" --users 5 --wait 1,3 --run-time 60s

test: ## unit + integration tests in the image (no real data, no API credits)
	$(RUN) pytest -q -m "not data and not api"

lint:
	$(RUN) ruff check .

demo: ## the whole story end to end: train -> serve -> retrain with gate -> rollback
	@./scripts/demo.sh
