---
name: check-all
description: Run all checks including lint, typecheck, and tests. Use before creating PRs or for comprehensive validation.
---

# Run All Checks

Complete validation including lint, types, and tests.

## Commands

```bash
# 1. Lint
git ls-files "*.py" | xargs python -m ruff check

# 2. Mypy
python -m mypy src/ --ignore-missing-imports --disable-error-code=union-attr --disable-error-code=no-redef --disable-error-code=no-any-return --disable-error-code=attr-defined --disable-error-code=assignment --disable-error-code=arg-type --disable-error-code=index --disable-error-code=misc

# 3. Tests
python -m pytest tests/ -v

# 4. Benchmark (optional)
python scripts/benchmark.py --quick
```

## Instructions

1. Run lint check (no auto-fix for PR verification)
2. Run mypy type check
3. Run pytest
4. Optionally run quick benchmark
5. Report comprehensive results:
   - Lint: PASS/FAIL (N issues)
   - Types: PASS/FAIL (N errors)
   - Tests: PASS/FAIL (N passed, M failed)
   - Benchmark: Performance summary (if run)

## PR Checklist

- [ ] Lint passes (no `--fix`)
- [ ] Mypy passes
- [ ] Tests pass
- [ ] Benchmark runs (optional)

## Notes

- Use this before `gh pr create`
- DO NOT create PR until all checks pass locally
- This is the full validation suite
