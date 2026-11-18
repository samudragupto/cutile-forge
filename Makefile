.PHONY: all build test bench lint clean docs

PYTHON ?= python
PYTEST ?= pytest -v --tb=short
MODEL  ?= D:/models/mistral_7b_4bit/
RESULTS ?= benchmarks/results/

all: build test

build:
	@echo "=== Building CUDA baselines ==="
	$(PYTHON) -m cutile_forge.baselines.build
	@echo "=== Installing package ==="
	$(PYTHON) -m pip install -e ".[dev]" --no-build-isolation --no-deps

test:
	@echo "=== Running unit tests ==="
	$(PYTEST) tests/ --timeout=120
	@echo "=== Running correctness tests ==="
	$(PYTEST) benchmarks/correctness/ --timeout=600
	@echo "All tests passed"

bench:
	@echo "=== Running micro-benchmark suite ==="
	$(PYTHON) benchmarks/run_all.py \
		--model $(MODEL) \
		--output $(RESULTS) \
		--micro-only \
		--warmup 5 --iters 200 \
		--thermal-logging
	@echo "=== Generating report ==="
	$(PYTHON) cutile_forge/bench/report.py \
		--input $(RESULTS) \
		--output docs/RESULTS.md
	@echo "Benchmarks complete. Results in $(RESULTS)"

lint:
	@echo "=== Linting ==="
	ruff check cutile_forge/ benchmarks/ tests/
	black --check cutile_forge/ benchmarks/ tests/
	mypy cutile_forge/ --ignore-missing-imports
	@echo "Lint passed"

clean:
	rm -rf cutile_forge/baselines/build/
	rm -rf cutile_forge/baselines/*.so cutile_forge/baselines/*.pyd
	rm -rf build/ dist/ *.egg-info
	rm -rf $(RESULTS)/*.csv $(RESULTS)/*.png $(RESULTS)/*.svg
	find . -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Cleaned"

docs:
	@echo "=== Generating docs ==="
	$(PYTHON) examples/visualize_tfg.py --layer 0 --output docs/tfg_layer0.svg
	$(PYTHON) examples/visualize_tfg.py --layer 1 --output docs/tfg_layer1.svg
	@echo "Docs generated"
