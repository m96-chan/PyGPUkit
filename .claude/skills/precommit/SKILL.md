---
name: precommit
description: Run all pre-commit checks (lint + typecheck). Use before every git commit to ensure code quality.
---

# Pre-Commit Checks

Run all required checks before committing.

## Commands

```bash
# 1. Ruff lint check (auto-fix and format)
git ls-files "*.py" | xargs python -m ruff check --fix
git ls-files "*.py" | xargs python -m ruff format

# 2. Mypy type check
python -m mypy src/ --ignore-missing-imports --disable-error-code=union-attr --disable-error-code=no-redef --disable-error-code=no-any-return --disable-error-code=attr-defined --disable-error-code=assignment --disable-error-code=arg-type --disable-error-code=index --disable-error-code=misc
```

## Instructions

1. Run lint check with auto-fix
2. Run format
3. Run mypy type check
4. Report results:
   - PASS if both succeed
   - FAIL with details if either fails
5. Stage any auto-fixed files if requested

## Checklist

- [ ] Ruff lint passes
- [ ] Ruff format applied
- [ ] Mypy type check passes

## Notes

- NEVER commit without passing ALL checks
- CI will reject PRs with lint/type errors
- Run this skill before every commit
