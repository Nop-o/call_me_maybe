PYTHON := python3
SRC_DIR := src
VENV := .venv

FUNCTIONS := data/input/functions_definition.json
PROMPTS   := data/input/function_calling_tests.json
OUTPUT    := data/output/function_calls.json

.DEFAULT_GOAL := help

.SILENT:

help:
	@echo "Available commands:"
	@echo "  make install					- Create virtual environment and install dependencies (uses uv)"
	@echo "  make run					- Execute the main script"
	@echo "  make debug					- Run the main script in debug mode (pdb)"
	@echo "  make clean					- Remove all temporary files and caches"
	@echo "  make lint					- Run flake8 and mypy with standard checks"
	@echo "  make lint-strict				- Run flake8 and mypy with strict mode"
	@echo "  make help					- Show this help message"


install:
	uv sync

run: install
	uv run $(PYTHON) -m $(SRC_DIR) \
		--functions_definition $(FUNCTIONS) \
		--input $(PROMPTS) \
		--output $(OUTPUT)

debug:
	uv run $(PYTHON) -m pdb -m $(SRC_DIR) \
		--functions_definition $(FUNCTIONS) \
		--input $(PROMPTS) \
		--output $(OUTPUT)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name $(VENV) -exec rm -rf {} +
	rm -rf $(OUTPUT)

lint:
	uv run flake8 $(SRC_DIR) --exclude=$(VENV)
	uv run mypy $(SRC_DIR) \
			--warn-unused-ignores \
	        --warn-return-any \
	        --ignore-missing-imports \
	        --disallow-untyped-defs \
	        --check-untyped-defs

lint-strict:
	uv run flake8 $(SRC_DIR) --exclude=$(VENV)
	uv run mypy	--strict $(SRC_DIR)

.PHONY: install run debug clean lint lint-strict help
