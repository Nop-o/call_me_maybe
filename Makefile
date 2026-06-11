PYTHON = python3

.SILENT:

help:
	@echo "Available commands:"
	@echo "  make install					- Create virtual environment and install dependencies"
	@echo "  make run					- Execute the main script"
	@echo "  make debug					- Run the main script in debug mode (pdb)"
	@echo "  make clean					- Remove all temporary files and caches"
	@echo "  make lint					- Run flake8 and mypy with standard checks"
	@echo "  make lint-strict				- Run flake8 and mypy with strict mode"
	@echo "  make help					- Show this help message"


install:
	uv sync

run:
	uv run $(PYTHON) -m src

debug:
	uv run $(PYTHON) -m src -m pdb -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

lint:
	uv run flake8 . --exclude=.venv
	uv run src --warn-unused-ignores \
	        --warn-return-any \
	        --ignore-missing-imports \
	        --disallow-untyped-defs \
	        --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=$(VENV)
	uv run mypy	--strict src

.PHONY: install run debug clean lint lint-strict help


.DEFAULT_GOAL := help