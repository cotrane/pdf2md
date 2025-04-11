#!/usr/bin/make -f

.PHONY: help check test

help: ## Show this help text
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install:
	@echo "Installing dependencies..."
	@uv sync --all-groups

check: check-lint check-format check-types
	@echo "All checks passed"

check-lint:
	@echo "Checking lint..."
	@uv run python -m pylint src/ tests/

check-format:
	@echo "Checking format..."
	@uv run python -m black src/ tests/

format:
	@echo "Formatting..."
	@uv run python -m black src/ tests/

check-types:
	@echo "Checking types..."
	@uv run python -m mypy src/ tests/ --ignore-missing-imports  --explicit-package-bases

test:
	@echo "Running tests..."
	@uv run pytest src/ tests/ --cov=src --cov-report=term-missing