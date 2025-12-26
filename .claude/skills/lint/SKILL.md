---
name: lint
description: Run Ruff linter and formatter on Python code. Use before commits or when checking code style and quality issues.
---

# Lint Python Code

Run Ruff linter with auto-fix and formatting.

## Usage

```bash
# Check and auto-fix
git ls-files "*.py" | xargs python -m ruff check --fix

# Format code
git ls-files "*.py" | xargs python -m ruff format
```

## Instructions

1. Run the lint check command
2. Run the format command
3. Report any remaining issues that could not be auto-fixed
4. If there are unfixable issues, suggest manual fixes

## Common Issues Fixed by Ruff

- Unused imports
- Missing trailing newlines
- Incorrect indentation
- Line length violations
- Import sorting

## Notes

- Always run lint before committing
- CI will reject PRs with lint errors
- Use `--fix` to auto-fix safe issues
