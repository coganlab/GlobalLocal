#!/bin/bash

# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run specific test file
pytest tests/analysis/utils/test_labeled_array_utils.py

# Run specific test class
pytest tests/analysis/utils/test_labeled_array_utils.py::TestLabeledArrayCreation

# Run specific test
pytest tests/analysis/utils/test_labeled_array_utils.py::TestLabeledArrayCreation::test_create_subject_labeled_array_from_dict

# Run tests in parallel (install pytest-xdist first)
pytest -n auto

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x