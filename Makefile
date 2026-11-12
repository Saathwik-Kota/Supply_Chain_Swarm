.PHONY: install install-dev run api test cov lint format typecheck migrate benchmark docker-build docker-up clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps --no-build-isolation -e .

install-dev: install
	$(PYTHON) -m pip install mypy ruff black isort pre-commit pytest pytest-cov

run:
	$(PYTHON) -m streamlit run app.py

api:
	$(PYTHON) -m uvicorn services.supplier_api:supplier_app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest tests/ -v

cov:
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check agents database evaluation llm ml observability rag services swarm tests app.py config.py
	$(PYTHON) -m black --check agents database evaluation llm ml observability rag services swarm app.py config.py
	$(PYTHON) -m isort --check-only agents database evaluation llm ml observability rag services swarm app.py config.py

format:
	$(PYTHON) -m isort agents database evaluation llm ml observability rag services swarm tests app.py config.py
	$(PYTHON) -m black agents database evaluation llm ml observability rag services swarm tests app.py config.py
	$(PYTHON) -m ruff check --fix agents database evaluation llm ml observability rag services swarm tests app.py config.py

typecheck:
	$(PYTHON) -m mypy config.py database evaluation llm ml observability rag services swarm agents

migrate:
	$(PYTHON) -m alembic upgrade head

benchmark:
	$(PYTHON) -m evaluation.benchmark

docker-build:
	docker compose build

docker-up:
	docker compose up -d

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

