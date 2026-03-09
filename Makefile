PYTHON ?= python3
UVICORN ?= uvicorn
STREAMLIT ?= streamlit
NPM ?= npm
PYTHONPATH ?= src
VERSION ?=
HOST ?= 0.0.0.0
PORT ?= 8000
IMAGE ?= cost-estimation-network:latest
FRONTEND_IMAGE ?= cost-estimation-frontend:latest
BASE_URL ?= http://127.0.0.1:8000

.PHONY: install install-dev train train-fast api app test test-cov lint versions activate recent drift alert docker-build docker-build-frontend preflight smoke frontend-dev frontend-build frontend-start frontend-lint release-check

install:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pip install .

install-dev:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pip install -e ".[dev]"

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m laptop_price.train

train-fast:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m laptop_price.train --no-tuning

api:
	PYTHONPATH=$(PYTHONPATH) $(UVICORN) laptop_price.api:app --host $(HOST) --port $(PORT)

app:
	PYTHONPATH=$(PYTHONPATH) $(STREAMLIT) run app.py

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

test-cov:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q --cov=laptop_price --cov-report=term-missing

lint:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ruff check src/ tests/

versions:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m laptop_price.train --list-versions

activate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m laptop_price.train --activate-version $(VERSION)

recent:
	tail -n 20 logs/predictions.jsonl

drift:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m laptop_price.drift

alert:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m laptop_price.alerts

docker-build:
	docker build -t $(IMAGE) .

docker-build-frontend:
	docker build -t $(FRONTEND_IMAGE) frontend/

preflight:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m laptop_price.ops preflight

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m laptop_price.ops smoke-test --base-url $(BASE_URL)

frontend-dev:
	cd frontend && $(NPM) run dev

frontend-build:
	cd frontend && $(NPM) run build

frontend-start:
	cd frontend && $(NPM) run start

frontend-lint:
	cd frontend && $(NPM) run lint -- .

release-check:
	./scripts/release_check.sh