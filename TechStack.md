```
PyGPUkit
│
├── python/        → Python API（NumPy互換）
│
├── core/
│   ├── C++ (CUDA Runtime API)
│   └── Rust backend（opt-in）
│
├── memory/
│   ├── Rust（LRU, pool allocator）
│   └── Python shim
│
├── scheduler/
│   ├── Rust（状態管理）
│   └── C++（kernel launch wrappers）
│
└── jit/
    ├── C++（NVRTC）
    └── Python wrappers
```
