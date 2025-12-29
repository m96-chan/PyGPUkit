# Current Development Status

## Branch
`feature/v0.2.16`

## Work in Progress (v0.2.16)
- #110 MoE - partial (chat_cli_moe.py exists, but not complete)
- #118 FP8 model loading - partial (FP8 GEMV, LinearFP8 exists, but not complete)

## Recently Added
- FP8 GEMV kernel with online dequantization
- LinearFP8 layer
- matmul directory restructure
- Build log saving
- Serena MCP integration

## Pending Issues
- #116 Triton Backend MVP
- #107 CUTLASS SM120 FP8 GEMM alignment fix
- #91 SM120 (RTX 5090) support
