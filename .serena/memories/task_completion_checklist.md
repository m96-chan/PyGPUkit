# Task Completion Checklist

## Before Every Commit (MANDATORY)
1. Ruff lint check:
   ```bash
   git ls-files "*.py" | xargs python -m ruff check --fix
   git ls-files "*.py" | xargs python -m ruff format
   ```

2. Mypy type check:
   ```bash
   python -m mypy src/ --ignore-missing-imports --disable-error-code=union-attr --disable-error-code=no-redef --disable-error-code=no-any-return --disable-error-code=attr-defined --disable-error-code=assignment --disable-error-code=arg-type --disable-error-code=index --disable-error-code=misc
   ```

## Before Python Implementation
1. Read `api_reference` memory first
2. Check if similar API already exists
3. Follow existing naming conventions and patterns
4. Avoid duplicating functionality

## Kernel Development
1. Edit -> Build -> Validate -> Benchmark -> Commit
2. Always commit after benchmark (even if no improvement)
3. Include benchmark results in commit message
4. Never overwrite working kernel without committing first

## Before PR
1. All lint/typecheck passes
2. Tests pass: `python -m pytest tests/ -v`
3. Benchmark (optional): `python benchmark.py --quick`
