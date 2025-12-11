#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void init_core_bindings(py::module_& m);
void init_jit_bindings(py::module_& m);
void init_ops_bindings(py::module_& m);

PYBIND11_MODULE(_pygpukit_native, m) {
    m.doc() = "PyGPUkit native backend";

    // Core module (device, memory, stream)
    init_core_bindings(m);

    // JIT module (NVRTC compiler)
    init_jit_bindings(m);

    // Ops module (basic operations)
    init_ops_bindings(m);
}
