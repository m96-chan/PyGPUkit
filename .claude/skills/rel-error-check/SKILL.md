# Relative Error Check

Measure GEMV kernel relative error vs FP32 reference.

## Formula

```
Rel.Err = ||C_test - C_fp32|| / ||C_fp32||
```

Where:
- `C_fp32` = FP32 matmul reference (numpy)
- `C_test` = Kernel output
- `||x||` = L2 norm (np.linalg.norm)

## Usage

```bash
python tests/check_rel_error.py [--kernel KERNEL] [--sizes K,N]
```

## Options

- `--kernel`: Kernel to test (bf16, w8a16, w8a8, w4a16, w4a4, int4, all)
- `--sizes`: Comma-separated K,N sizes (default: 4096,4096)

## Example

```bash
# Check all kernels
python tests/check_rel_error.py --kernel all

# Check specific kernel
python tests/check_rel_error.py --kernel w8a8 --sizes 4096,14336
```

## Expected Results

| Kernel | Rel. Err (vs FP32) |
|--------|-------------------|
| BF16   | ~0.6%             |
| W8A16  | ~12%              |
| W8A8   | ~9%               |
| W4A16  | ~15%              |
| W4A4   | ~20%              |
| Int4   | ~15%              |
