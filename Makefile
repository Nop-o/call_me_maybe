PYTHON = python3
VENV = call_me_maybe_venv

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
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo ""
	@echo "source $(VENV)/bin/activate"

run:
	$(VENV)/bin/python -m src.call_me_maybe $(INPUT)

debug:
	$(VENV)/bin/python -m pdb -m src.call_me_maybe $(INPUT)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name $(VENV) -exec rm -rf {} +

lint:
	$(PYTHON) -m flake8 src --exclude=$(VENV)
	mypy src --warn-unused-ignores \
	        --warn-return-any \
	        --ignore-missing-imports \
	        --disallow-untyped-defs \
	        --check-untyped-defs

lint-strict:
	$(PYTHON) -m flake8 src --exclude=$(VENV)
	mypy	--strict src

.PHONY: install run debug clean lint lint-strict help


.DEFAULT_GOAL := help