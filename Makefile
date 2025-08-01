# Run all tests
test:
	pytest

# Run fast tests only
test-fast:
	pytest -m "not slow"

# Run tests with coverage
test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# Run tests in watch mode (requires pytest-watch)
test-watch:
	ptw --runner "pytest -x"

# Run tests in parallel
test-parallel:
	pytest -n auto

# Clean test artifacts
clean-test:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +