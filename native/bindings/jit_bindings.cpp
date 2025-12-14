#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../jit/compiler.hpp"
#include "../jit/kernel.hpp"

namespace py = pybind11;
using namespace pygpukit;

void init_jit_bindings(py::module_& m) {
    // CompiledPTX struct
    py::class_<CompiledPTX>(m, "CompiledPTX")
        .def_readonly("ptx", &CompiledPTX::ptx)
        .def_readonly("log", &CompiledPTX::log);

    // is_nvrtc_available function
    m.def("is_nvrtc_available", &is_nvrtc_available,
          "Check if NVRTC JIT compiler is available.\n\n"
          "NVRTC enables runtime compilation of custom CUDA kernels.\n"
          "Pre-compiled GPU operations work without NVRTC.\n\n"
          "Returns:\n"
          "    bool: True if NVRTC is functional, False otherwise.");

    // compile_to_ptx function
    m.def("compile_to_ptx", &compile_to_ptx,
          py::arg("source"),
          py::arg("name") = "kernel.cu",
          py::arg("options") = std::vector<std::string>{},
          "Compile CUDA source to PTX.\n\n"
          "Requires NVRTC. Use is_nvrtc_available() to check.\n\n"
          "Args:\n"
          "    source: CUDA C++ source code\n"
          "    name: Kernel filename (default: kernel.cu)\n"
          "    options: Compiler options\n\n"
          "Returns:\n"
          "    CompiledPTX with ptx and log attributes\n\n"
          "Raises:\n"
          "    RuntimeError: If NVRTC is not available or compilation fails.");

    // get_nvrtc_version function
    m.def("get_nvrtc_version", []() {
        int major, minor;
        get_nvrtc_version(&major, &minor);
        return py::make_tuple(major, minor);
    }, "Get NVRTC version as (major, minor).\n\n"
       "Requires NVRTC. Use is_nvrtc_available() to check.\n\n"
       "Returns:\n"
       "    tuple: (major, minor) version numbers\n\n"
       "Raises:\n"
       "    RuntimeError: If NVRTC is not available.");

    // get_nvrtc_library_path function
    m.def("get_nvrtc_library_path", &get_nvrtc_library_path,
          "Get the path to the loaded NVRTC library.\n\n"
          "Returns:\n"
          "    str: Path to NVRTC DLL/SO if loaded, empty string otherwise.");

    // JITKernel class
    py::class_<JITKernel>(m, "JITKernel")
        .def(py::init<const std::string&, const std::string&, const std::vector<std::string>&>(),
             py::arg("source"),
             py::arg("func_name"),
             py::arg("options") = std::vector<std::string>{})
        .def_property_readonly("name", &JITKernel::name)
        .def_property_readonly("ptx", &JITKernel::ptx)
        .def_property_readonly("is_compiled", &JITKernel::is_compiled)
        .def("get_suggested_block_size", &JITKernel::get_suggested_block_size,
             py::arg("dynamic_smem") = 0)
        .def("__repr__", [](const JITKernel& self) {
            return "JITKernel(name=" + self.name() + ", compiled=" +
                   (self.is_compiled() ? "true" : "false") + ")";
        });
}
