# Default command, runs if no recipe is specified.
default:
    just --list

# Variables
SOURCE_DIR      := "src"
TESTS_DIR       := "tests"
PYTHON_DIRS     := "src tests"
DOCS_SOURCE     := "docs/source"
DOCS_BUILD      := "docs/build"
HTML_COVERAGE_DIR := "htmlcov"

# Show available commands
help:
    @just --list

# Testing & Coverage
# Run tests using pytest.
test:
    pytest {{TESTS_DIR}}

# Run tests and generate coverage data.
coverage:
    pytest --cov=batdetect2 --cov-report=term-missing --cov-report=xml {{TESTS_DIR}}

# Generate an HTML coverage report.
coverage-html: coverage
    @echo "Generating HTML coverage report..."
    coverage html -d {{HTML_COVERAGE_DIR}}
    @echo "HTML coverage report generated in {{HTML_COVERAGE_DIR}}/"

# Serve the HTML coverage report locally.
coverage-serve: coverage-html
    @echo "Serving report at http://localhost:8000/ ..."
    python -m http.server --directory {{HTML_COVERAGE_DIR}} 8000

# Documentation
# Build documentation using Sphinx.
docs:
    sphinx-build -b html {{DOCS_SOURCE}} {{DOCS_BUILD}}

# Serve documentation with live reload.
docs-serve:
    sphinx-autobuild {{DOCS_SOURCE}} {{DOCS_BUILD}} --watch {{SOURCE_DIR}} --open-browser

# Formatting & Linting
# Format code using ruff.
format:
    ruff format {{PYTHON_DIRS}}

# Check code formatting using ruff.
format-check:
    ruff format --check {{PYTHON_DIRS}}

# Lint code using ruff.
lint:
    ruff check {{PYTHON_DIRS}}

# Lint code using ruff and apply automatic fixes.
lint-fix:
    ruff check --fix {{PYTHON_DIRS}}

# Type Checking
# Type check code using pyright.
typecheck:
    pyright {{PYTHON_DIRS}}

# Combined Checks
# Run all checks (format-check, lint, typecheck).
check: format-check lint typecheck test

# Cleaning tasks
# Remove Python bytecode and cache.
clean-pyc:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove test and coverage artifacts.
clean-test:
    rm -f .coverage coverage.xml
    rm -rf .pytest_cache htmlcov/

# Remove built documentation.
clean-docs:
    rm -rf {{DOCS_BUILD}}

# Remove package build artifacts.
clean-build:
    rm -rf build/ dist/ *.egg-info/

# Remove all build, test, documentation, and Python artifacts.
clean: clean-build clean-pyc clean-test clean-docs

# Examples
# Preprocess example data.
example-preprocess OPTIONS:
    batdetect2 preprocess \
        --base-dir . \
        --dataset-field datasets.train \
        --config example_data/config.yaml \
        {{OPTIONS}} \
        example_data/datasets.yaml example_data/preprocessed

# Train on example data.
example-train:
    batdetect2 train \
        --train-examples example_data/preprocessed \
        --train-config config.yaml \
        --train-config-field train \
        --preprocess-config config.yaml \
        --preprocess-config-field preprocessing \
        --target-config config.yaml \
        --target-config-field targets \
        --postprocess-config config.yaml \
        --postprocess-config-field postprocessing \
        --model-config config.yaml \
        --model-config-field model
