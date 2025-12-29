# Suggested Commands

## Build (Git Bash)
```bash
./build.sh 120a     # RTX 5090 (default)
./build.sh 86       # RTX 3090 Ti
./build.sh 89       # RTX 4090
./build.sh 90       # H100
```
Build logs: `.claude/logs/build/`

## Lint & Format
```bash
git ls-files "*.py" | xargs python -m ruff check --fix
git ls-files "*.py" | xargs python -m ruff format
```

## Type Check
```bash
python -m mypy src/ --ignore-missing-imports --disable-error-code=union-attr --disable-error-code=no-redef --disable-error-code=no-any-return --disable-error-code=attr-defined --disable-error-code=assignment --disable-error-code=arg-type --disable-error-code=index --disable-error-code=misc
```

## Test
```bash
python -m pytest tests/ -v
```

## Benchmark
```bash
python benchmark.py          # Full benchmark
python benchmark.py --quick  # Quick mode
```

## LLM Chat Test
```bash
python examples/chat_cli.py F:/LLM/Qwen2.5-7B-Instruct
```

## Git (Windows/Git Bash)
Standard git commands work in Git Bash.
