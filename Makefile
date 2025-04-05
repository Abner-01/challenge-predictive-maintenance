# =============================================================================
# Makefile for Managing the QuadSci Challenge Project
# =============================================================================


.PHONY: format 

# Format and lint the codebase
format: 
	@echo "Starting code formatting and linting..."
	bash scripts/lint.sh

