.PHONY: \
	install test lint format format-check typecheck \
	notebooks notebooks-check smoke experiment check

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev,notebooks,parquet]"

test:
	$(PYTHON) -m pytest --cov=rllab --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src/rllab

notebooks:
	$(PYTHON) scripts/build_notebooks.py

notebooks-check:
	$(PYTHON) scripts/build_notebooks.py --check

smoke:
	$(PYTHON) experiments/stochastic_maze/run_stochasticity_sweep.py --quick

experiment:
	$(PYTHON) -m rllab.cli run configs/stochasticity_sweep.yaml

check: format-check lint typecheck notebooks-check test
