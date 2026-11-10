.DEFAULT_GOAL := help
PY ?= python3
VENV := venv
BIN := $(VENV)/bin

.PHONY: help setup test lint fmt check clean-cache figures sim index tree nbstrip-on nbstrip-off nbsize

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install the package with all extras
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[all]"

test:  ## Run the test suite
	$(BIN)/python -m pytest

lint:  ## Check style and catch unused/undefined names
	$(BIN)/ruff check src scripts tests

fmt:  ## Auto-fix what ruff can, including import order
	$(BIN)/ruff check --fix src scripts tests
	$(BIN)/ruff format src scripts tests

check: lint test  ## Lint and test

figures:  ## Regenerate the experimental figures
	$(BIN)/python scripts/figures/plot_all_evolution.py
	$(BIN)/python scripts/figures/plot_single_rep_evolution.py
	$(BIN)/python scripts/figures/plot_coarsening_pattern.py
	$(BIN)/python scripts/figures/plot_coarsening_size.py

sim:  ## Run the simulation demos
	$(BIN)/python scripts/simulations/run_simulation_ho.py
	$(BIN)/python scripts/simulations/run_sodium_domains.py

index:  ## Build the SQLite shot index for the coarsening dataset
	$(BIN)/python scripts/analysis/index_shots.py --build coarsening

nbsize:  ## Show how much notebook output would go into git history
	@du -h notebooks/*.ipynb | sort -h

nbstrip-on:  ## Strip notebook OUTPUTS on commit (source kept, outputs stay local)
	$(BIN)/pip install nbstripout
	$(BIN)/nbstripout --install --attributes .gitattributes
	@echo "Enabled. Notebooks now commit without embedded output images."

nbstrip-off:  ## Stop stripping notebook outputs on commit
	$(BIN)/nbstripout --uninstall --attributes .gitattributes

clean-cache:  ## Remove bytecode and tool caches (never touches data/ or results/)
	find . -name __pycache__ -type d -not -path './venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov

tree:  ## Show the layout, without data or the venv
	find . -not -path './.git/*' -not -path './venv/*' -not -path './data/*' \
		-not -path './results/*' -not -path '*/__pycache__/*' \
		-not -name '.DS_Store' | sort
