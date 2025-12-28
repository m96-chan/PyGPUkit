---
name: build-log
description: View and analyze build logs. Use when user wants to see build errors, check previous build output, or debug build failures.
---

# Build Log Viewer

View and analyze PyGPUkit build logs.

## Log Location

Build logs are stored in `.claude/logs/build/`

**Format:** `build_sm{SM}_cuda{VERSION}_{TIMESTAMP}.log`

## Usage

```bash
# List recent logs
ls -lt .claude/logs/build/

# View latest log
cat .claude/logs/build/$(ls -t .claude/logs/build/ | head -1)

# View specific log
cat .claude/logs/build/build_sm120a_cuda13.1_20241227_120000.log

# Search for errors
grep -i error .claude/logs/build/$(ls -t .claude/logs/build/ | head -1)

# Search for warnings
grep -i warning .claude/logs/build/$(ls -t .claude/logs/build/ | head -1)

# Show last 50 lines of latest log
tail -50 .claude/logs/build/$(ls -t .claude/logs/build/ | head -1)
```

## Common Error Patterns

| Pattern | Meaning |
|---------|---------|
| `nvcc fatal` | CUDA compilation error |
| `error: no operator` | C++ type mismatch |
| `undefined reference` | Linker error (missing symbol) |
| `CMake Error` | Build configuration issue |
| `fatal error C1083` | Missing header file |

## Instructions

1. When build fails, first list recent logs
2. Read the latest log or the specific failed build log
3. Search for `error:` or `fatal:` patterns
4. Report the specific error message and context
5. Suggest fixes based on the error type

## Cleanup

Logs are automatically cleaned up (last 10 kept). To manually clean:

```bash
# Remove all logs older than 7 days
find .claude/logs/build/ -mtime +7 -delete

# Keep only last 5 logs
ls -t .claude/logs/build/*.log | tail -n +6 | xargs rm -f
```
