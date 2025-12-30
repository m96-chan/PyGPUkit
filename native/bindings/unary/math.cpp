/**
 * Unary math operations: exp, log, sqrt, rsqrt, abs, neg
 */
#include "../bindings_common.hpp"

void init_unary_math(py::module_& m) {
    // Exp
    m.def("exp", py::overload_cast<const GPUArray&>(&ops::exp),
          py::arg("a"),
          "Element-wise exponential (float32/float64 only)");

    m.def("exp_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::exp),
          py::arg("a"), py::arg("out"),
          "Element-wise exponential with output array");

    // Log
    m.def("log", py::overload_cast<const GPUArray&>(&ops::log),
          py::arg("a"),
          "Element-wise natural logarithm (float32/float64 only)");

    m.def("log_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::log),
          py::arg("a"), py::arg("out"),
          "Element-wise natural logarithm with output array");

    // Sqrt
    m.def("sqrt", py::overload_cast<const GPUArray&>(&ops::sqrt),
          py::arg("a"),
          "Element-wise square root");

    m.def("sqrt_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::sqrt),
          py::arg("a"), py::arg("out"),
          "Element-wise square root with output array");

    // Rsqrt
    m.def("rsqrt", py::overload_cast<const GPUArray&>(&ops::rsqrt),
          py::arg("a"),
          "Element-wise reciprocal square root: 1/sqrt(x)");

    m.def("rsqrt_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::rsqrt),
          py::arg("a"), py::arg("out"),
          "Element-wise reciprocal square root with output array");

    // Abs
    m.def("abs", py::overload_cast<const GPUArray&>(&ops::abs),
          py::arg("a"),
          "Element-wise absolute value");

    m.def("abs_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::abs),
          py::arg("a"), py::arg("out"),
          "Element-wise absolute value with output array");

    // Neg
    m.def("neg", py::overload_cast<const GPUArray&>(&ops::neg),
          py::arg("a"),
          "Element-wise negation: -x");

    m.def("neg_", py::overload_cast<const GPUArray&, GPUArray&>(&ops::neg),
          py::arg("a"), py::arg("out"),
          "Element-wise negation with output array");
}
