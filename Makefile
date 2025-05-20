.PHONY: install test coverage rust wasm demo serve check docker clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

coverage:
	pytest --cov=triton_gateway --cov-branch --cov-report=term-missing --cov-report=html

rust:
	cargo test

wasm:
	wasm-pack build --target web --release

demo:
	python -m triton_gateway --demo

serve:
	uvicorn app.main:app --reload --port 8000

check: coverage rust

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .coverage htmlcov coverage.xml build dist *.egg-info pkg target
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
