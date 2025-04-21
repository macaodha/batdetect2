# Variables
PYTHON_DIRS = batdetect2 tests
DOCS_SOURCE = docs/source
DOCS_BUILD = docs/build
HTML_COVERAGE_DIR = htmlcov

# Default target (optional, often 'help' or 'all')
.DEFAULT_GOAL := help

# Phony targets (targets that don't produce a file with the same name)
.PHONY: help test coverage coverage-report coverage-html docs docs-serve format format-check lint lint-fix typecheck check clean clean-pyc clean-test clean-docs clean-build

help:
	@echo "Makefile Targets:"
	@echo "  help             Show this help message."
	@echo "  test             Run tests using pytest."
	@echo "  coverage         Run tests and generate coverage data (.coverage, coverage.xml)."
	@echo "  coverage-report  Show coverage report in the terminal."
	@echo "  coverage-html    Generate an HTML coverage report in htmlcov/."
	@echo "  docs             Build documentation using Sphinx."
	@echo "  docs-serve       Serve documentation with live reload using sphinx-autobuild."
	@echo "  format           Format code using ruff."
	@echo "  format-check     Check code formatting using ruff."
	@echo "  lint             Lint code using ruff."
	@echo "  lint-fix         Lint code using ruff and apply automatic fixes."
	@echo "  typecheck        Type check code using pyright."
	@echo "  check            Run all checks (format-check, lint, typecheck)."
	@echo "  clean            Remove all build, test, documentation, and Python artifacts."
	@echo "  clean-pyc        Remove Python bytecode and cache."
	@echo "  clean-test       Remove test and coverage artifacts."
	@echo "  clean-docs       Remove built documentation."
	@echo "  clean-build      Remove package build artifacts."

# Testing & Coverage
test:
	pytest tests

coverage:
	pytest --cov=batdetect2 --cov-report=term-missing --cov-report=xml tests

coverage-report: coverage
	@echo "Generating HTML coverage report..."
	coverage html -d $(HTML_COVERAGE_DIR)
	@echo "HTML coverage report generated in $(HTML_COVERAGE_DIR)/"
	@echo "Serving report at http://localhost:8000/ ..."
	python -m http.server --directory $(HTML_COVERAGE_DIR)

# Documentation
docs:
	sphinx-build -b html $(DOCS_SOURCE) $(DOCS_BUILD)

docs-serve:
	sphinx-autobuild $(DOCS_SOURCE) $(DOCS_BUILD) --watch $(PYTHON_DIRS)

# Formatting & Linting
format:
	ruff format $(PYTHON_DIRS)

format-check:
	ruff format --check $(PYTHON_DIRS)

lint:
	ruff check $(PYTHON_DIRS)

lint-fix:
	ruff check --fix $(PYTHON_DIRS)

# Type Checking
typecheck:
	pyright $(PYTHON_DIRS)

# Combined Checks
check: format-check lint typecheck test

# Cleaning tasks
clean-pyc:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-test:
	rm -f .coverage coverage.xml
	rm -rf .pytest_cache htmlcov/

clean-docs:
	rm -rf $(DOCS_BUILD)

clean-build:
	rm -rf build/ dist/ *.egg-info/

clean: clean-build clean-pyc clean-test clean-docs
