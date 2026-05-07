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

install:
    uv sync

# Install full development dependencies for CI and docs builds.
install-dev:
    uv sync --all-extras --dev

# Testing & Coverage
# Run tests using pytest.
test:
    uv run pytest {{TESTS_DIR}}

# Run the fast subset of tests (excludes @pytest.mark.slow).
test-quick:
    uv run pytest --durations=10 -m "not slow" {{TESTS_DIR}}

# Run only long-running tests marked with @pytest.mark.slow.
test-slow:
    uv run pytest -m "slow" {{TESTS_DIR}}

# Run tests and generate coverage data.
coverage:
    uv run pytest --cov=batdetect2 --cov-report=term-missing --cov-report=xml {{TESTS_DIR}}

# Generate an HTML coverage report.
coverage-html: coverage
    @echo "Generating HTML coverage report..."
    uv run coverage html -d {{HTML_COVERAGE_DIR}}
    @echo "HTML coverage report generated in {{HTML_COVERAGE_DIR}}/"

# Serve the HTML coverage report locally.
coverage-serve: coverage-html
    @echo "Serving report at http://localhost:8000/ ..."
    uv run python -m http.server --directory {{HTML_COVERAGE_DIR}} 8000

# Documentation
# Build documentation using Sphinx.
docs:
    uv run sphinx-build -b html {{DOCS_SOURCE}} {{DOCS_BUILD}}

# Check that documentation builds successfully.
check-docs: docs

# Serve documentation with live reload.
docs-serve:
    uv run sphinx-autobuild {{DOCS_SOURCE}} {{DOCS_BUILD}} --watch {{SOURCE_DIR}} --open-browser

# Formatting & Linting
# Format code using ruff.
fix-format:
    uv run ruff format {{PYTHON_DIRS}}

# Lint code using ruff and apply automatic fixes.
fix-lint:
    uv run ruff check --fix {{PYTHON_DIRS}}

# Combined Formatting & Linting
fix: fix-format fix-lint

# Checking tasks
# Check code formatting using ruff.
check-format:
    uv run ruff format --check {{PYTHON_DIRS}}

# Lint code using ruff.
check-lint:
    uv run ruff check {{PYTHON_DIRS}}

# Type Checking
# Type check code using ty.
check-types:
    uv run ty check {{PYTHON_DIRS}}

# Combined Checks
# Run all checks (format-check, lint, typecheck).
check: check-format check-lint check-types

# Run the standard CI validation sequence.
ci: check test

# Build source and wheel distributions.
build-dist:
    uv run --with build python -m build

# Bump the patch version, commit, and tag.
bump-patch:
    uvx bump2version patch

# Bump the minor version, commit, and tag.
bump-minor:
    uvx bump2version minor

# Bump the major version, commit, and tag.
bump-major:
    uvx bump2version major

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

# Train on example data.
example-train OPTIONS="":
    uv run batdetect2 train \
        --val-dataset example_data/dataset.yaml \
        --base-dir . \
        --targets example_data/targets.yaml \
        --model-config example_data/configs/model.yaml \
        --training-config example_data/configs/training.yaml \
        --audio-config example_data/configs/audio.yaml \
        --evaluation-config example_data/configs/evaluation.yaml \
        --logging-config example_data/configs/logging.yaml \
        {{OPTIONS}} \
        example_data/dataset.yaml
