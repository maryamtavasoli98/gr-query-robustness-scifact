PY ?= .venv/bin/python
.PHONY: venv data eval figures readme all test lint
venv:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"
data:            ## download and prepare the dataset (never committed)
	$(PY) scripts/prepare_data.py
eval:            ## all experiments (about 9 minutes on an M3 Pro; writes results/metrics/*.json)
	$(PY) scripts/run_eval.py
figures:
	$(PY) scripts/make_figures.py
readme:          ## render tables from results/metrics into README.md and report/report.md
	$(PY) scripts/update_readme.py
all: data eval figures readme
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
