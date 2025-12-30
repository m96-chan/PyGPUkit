/**
 * Sampling seed operations
 */
#include "../bindings_common.hpp"

void init_sampling_seed(py::module_& m) {
    m.def("set_sampling_seed", &ops::set_sampling_seed,
          py::arg("seed"),
          "Set random seed for reproducible GPU sampling.");
}
