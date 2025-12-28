---
name: test
description: Run pytest test suite. Use to verify functionality after code changes or before commits.
---

# Run Tests

Execute pytest test suite.

## Usage

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_matmul.py -v

# Run with coverage
python -m pytest tests/ -v --cov=src/pygpukit

# Run only fast tests (skip slow GPU tests)
python -m pytest tests/ -v -m "not slow"
```

## Instructions

1. Run the pytest command
2. Report test results:
   - Number of passed/failed/skipped tests
   - Any failure details with tracebacks
   - Suggestions for fixing failures

## Test Categories

- `tests/` - Main test directory
- Unit tests for core functionality
- Integration tests for GPU operations
- Some tests require GPU and may be slow

## Notes

- Run after lint and typecheck
- GPU tests require CUDA device
- Use `-v` for verbose output
- Use `-x` to stop on first failure
