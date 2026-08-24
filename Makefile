.PHONY: install run test lint format type-check clean

install:
	pip install -e ".[embed,store,dev]"

run:
	uvicorn vector_service.main:app --reload --host $$(grep VS_HOST .env | cut -d= -f2) --port $$(grep VS_PORT .env | cut -d= -f2)

test:
	pytest -q

test-unit:
	pytest -q tests/unit

test-contract:
	pytest -q tests/contract

test-integration:
	pytest -q tests/integration -m "not slow"

test-slow:
	pytest -q tests/integration -m slow

lint:
	ruff check src tests

format:
	ruff format src tests

type-check:
	mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
