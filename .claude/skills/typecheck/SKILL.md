---
name: typecheck
description: Run Mypy type checker on Python code. Use to verify type annotations and catch type errors before commits.
---

# Type Check Python Code

Run Mypy with project-specific settings.

## Usage

```bash
python -m mypy src/ --ignore-missing-imports --disable-error-code=union-attr --disable-error-code=no-redef --disable-error-code=no-any-return --disable-error-code=attr-defined --disable-error-code=assignment --disable-error-code=arg-type --disable-error-code=index --disable-error-code=misc
```

## Instructions

1. Run the mypy command with all specified flags
2. Report any type errors found
3. For each error, explain:
   - What the type mismatch is
   - Suggested fix
4. Do not modify code unless explicitly asked

## Disabled Error Codes

These are disabled for compatibility with the native module and dynamic types:

- `union-attr`: Union type attribute access
- `no-redef`: Function redefinition
- `no-any-return`: Return Any type
- `attr-defined`: Dynamic attributes
- `assignment`: Dynamic assignment types
- `arg-type`: Argument type mismatches
- `index`: Index type errors
- `misc`: Miscellaneous errors

## Notes

- Run after lint check
- CI will reject PRs with type errors
- Only check files in `src/` directory
