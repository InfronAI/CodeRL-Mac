PYTHON ?= python3.11
VENV := .venv

.PHONY: setup doctor test tiny experiment report demo
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -e '.[dev]'
doctor:
	$(VENV)/bin/coderl doctor
test:
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/python -m pytest -q
tiny:
	$(VENV)/bin/coderl run --config configs/tiny.json --output experiments/tiny-local
experiment:
	$(VENV)/bin/coderl run --config configs/m1.json --output experiments/m1-local
report:
	$(VENV)/bin/coderl report $(RUN)
demo:
	$(VENV)/bin/python scripts/demo.py --run $(RUN)
