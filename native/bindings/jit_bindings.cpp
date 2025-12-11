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

    // compile_to_ptx function
    m.def("compile_to_ptx", &compile_to_ptx,
          py::arg("source"),
          py::arg("name") = "kernel.cu",
          py::arg("options") = std::vector<std::string>{},
          "Compile CUDA source to PTX");

    // get_nvrtc_version function
    m.def("get_nvrtc_version", []() {
        int major, minor;
        get_nvrtc_version(&major, &minor);
        return py::make_tuple(major, minor);
    }, "Get NVRTC version as (major, minor)");

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
