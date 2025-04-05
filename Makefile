# =============================================================================
# Makefile for Managing the QuadSci Challenge Project
# =============================================================================


.PHONY: format test up

# Format and lint the codebase
format: 
	@echo "Starting code formatting and linting..."
	bash scripts/lint.sh

test: 
	@echo "Running tests..."
	uv run --extra "format" pytest

up:
	@echo "Starting the application..."
	bash scripts/up.sh