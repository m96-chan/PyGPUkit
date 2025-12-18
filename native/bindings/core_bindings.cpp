#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../core/device.hpp"
#include "../core/memory.hpp"
#include "../core/stream.hpp"
#include "../core/cuda_graph.hpp"

namespace py = pybind11;
using namespace pygpukit;

void init_core_bindings(py::module_& m) {
    // DataType enum
    py::enum_<DataType>(m, "DataType")
        .value("Float32", DataType::Float32)
        .value("Float64", DataType::Float64)
        .value("Float16", DataType::Float16)
        .value("BFloat16", DataType::BFloat16)
        .value("Int32", DataType::Int32)
        .value("Int64", DataType::Int64)
        .value("Int8", DataType::Int8)
        .value("UInt8", DataType::UInt8)
        .value("Int4", DataType::Int4)
        .export_values();

    // StreamPriority enum
    py::enum_<StreamPriority>(m, "StreamPriority")
        .value("High", StreamPriority::High)
        .value("Low", StreamPriority::Low)
        .export_values();

    // DeviceProperties struct
    py::class_<DeviceProperties>(m, "DeviceProperties")
        .def_readonly("name", &DeviceProperties::name)
        .def_readonly("total_memory", &DeviceProperties::total_memory)
        .def_readonly("compute_capability_major", &DeviceProperties::compute_capability_major)
        .def_readonly("compute_capability_minor", &DeviceProperties::compute_capability_minor)
        .def_readonly("multiprocessor_count", &DeviceProperties::multiprocessor_count)
        .def_readonly("max_threads_per_block", &DeviceProperties::max_threads_per_block)
        .def_readonly("warp_size", &DeviceProperties::warp_size);

    // Device functions
    m.def("is_cuda_available", &is_cuda_available, "Check if CUDA is available");
    m.def("get_driver_version", &get_driver_version, "Get CUDA driver version");
    m.def("get_runtime_version", &get_runtime_version, "Get CUDA runtime version");
    m.def("get_device_count", &get_device_count, "Get number of CUDA devices");
    m.def("get_device_properties", &get_device_properties,
          py::arg("device_id") = 0, "Get properties of a CUDA device");
    m.def("set_device", &set_device, py::arg("device_id"), "Set current device");
    m.def("get_current_device", &get_current_device, "Get current device");
    m.def("device_synchronize", &device_synchronize, "Synchronize current device");
    m.def("get_sm_version", &get_sm_version, py::arg("device_id") = 0,
          "Get SM version as integer (e.g., 86 for SM 8.6)");
    m.def("validate_compute_capability", &validate_compute_capability,
          py::arg("device_id") = 0,
          "Validate device compute capability (requires SM >= 80)");
    m.def("get_recommended_arch", &get_recommended_arch, py::arg("device_id") = 0,
          "Get recommended -arch option for JIT compilation (e.g., 'sm_86')");
    m.def("get_fallback_archs", &get_fallback_archs, py::arg("device_id") = 0,
          "Get fallback -arch options for older drivers (in order of preference)");
    m.def("is_arch_supported", &is_arch_supported, py::arg("arch"),
          "Check if driver supports a given PTX architecture");

    // GPUArray class
    py::class_<GPUArray>(m, "GPUArray")
        .def(py::init<const std::vector<size_t>&, DataType>(),
             py::arg("shape"), py::arg("dtype"))
        .def_property_readonly("shape", &GPUArray::shape)
        .def_property_readonly("dtype", &GPUArray::dtype)
        .def_property_readonly("ndim", &GPUArray::ndim)
        .def_property_readonly("size", &GPUArray::size)
        .def_property_readonly("nbytes", &GPUArray::nbytes)
        .def_property_readonly("itemsize", &GPUArray::itemsize)
        .def("fill_zeros", &GPUArray::fill_zeros)
        .def("copy_from_numpy", [](GPUArray& self, py::array arr) {
            // Ensure contiguous
            arr = py::array::ensure(arr, py::array::c_style);
            self.copy_from_host(arr.data());
        })
        .def("to_numpy", [](const GPUArray& self) {
            // Create numpy array with appropriate dtype
            std::vector<py::ssize_t> py_shape(self.shape().begin(), self.shape().end());
            py::array result;

            switch (self.dtype()) {
                case DataType::Float32:
                    result = py::array_t<float>(py_shape);
                    break;
                case DataType::Float64:
                    result = py::array_t<double>(py_shape);
                    break;
                case DataType::Float16:
                    // NumPy has native float16 support
                    result = py::array(py::dtype("float16"), py_shape);
                    break;
                case DataType::BFloat16:
                    // NumPy doesn't have native bfloat16, use uint16 as storage
                    // Users can convert using ml_dtypes or similar libraries
                    result = py::array(py::dtype("uint16"), py_shape);
                    break;
                case DataType::Int32:
                    result = py::array_t<int32_t>(py_shape);
                    break;
                case DataType::Int64:
                    result = py::array_t<int64_t>(py_shape);
                    break;
                case DataType::Int8:
                    result = py::array_t<int8_t>(py_shape);
                    break;
                case DataType::UInt8:
                    result = py::array_t<uint8_t>(py_shape);
                    break;
                case DataType::Int4:
                    // Int4 packs 2 values per byte, use uint8 for storage
                    result = py::array_t<uint8_t>(py_shape);
                    break;
            }

            self.copy_to_host(result.mutable_data());
            return result;
        })
        .def("__repr__", [](const GPUArray& self) {
            std::string shape_str = "(";
            for (size_t i = 0; i < self.shape().size(); ++i) {
                if (i > 0) shape_str += ", ";
                shape_str += std::to_string(self.shape()[i]);
            }
            shape_str += ")";
            return "GPUArray(shape=" + shape_str + ", dtype=" + dtype_name(self.dtype()) + ")";
        })
        .def_property_readonly("owns_memory", &GPUArray::owns_memory,
            "Whether this array owns its memory (False for views)")
        .def_static("narrow", &GPUArray::narrow,
            py::arg("source"), py::arg("offset_elements"), py::arg("new_shape"),
            "Create a zero-copy view into source array.\n\n"
            "Args:\n"
            "    source: Source GPUArray to view into\n"
            "    offset_elements: Offset from start in number of elements\n"
            "    new_shape: Shape of the view\n\n"
            "Returns:\n"
            "    Non-owning GPUArray pointing to source memory + offset\n\n"
            "Note: The returned view does not own memory - source must outlive the view.");

    // Factory functions
    m.def("zeros", &zeros, py::arg("shape"), py::arg("dtype"),
          "Create a GPUArray filled with zeros");
    m.def("ones", &ones, py::arg("shape"), py::arg("dtype"),
          "Create a GPUArray filled with ones");
    m.def("empty", &empty, py::arg("shape"), py::arg("dtype"),
          "Create an uninitialized GPUArray");

    m.def("from_numpy", [](py::array arr) {
        // Ensure contiguous
        arr = py::array::ensure(arr, py::array::c_style);

        // Determine dtype based on numpy dtype
        DataType dtype;
        py::dtype np_dtype = arr.dtype();
        char kind = np_dtype.kind();
        size_t itemsize = np_dtype.itemsize();

        if (kind == 'f') {
            // Floating point types
            if (itemsize == 4) {
                dtype = DataType::Float32;
            } else if (itemsize == 8) {
                dtype = DataType::Float64;
            } else if (itemsize == 2) {
                dtype = DataType::Float16;
            } else {
                throw std::runtime_error("Unsupported float dtype size: " + std::to_string(itemsize));
            }
        } else if (kind == 'i') {
            // Signed integer types
            if (itemsize == 4) {
                dtype = DataType::Int32;
            } else if (itemsize == 8) {
                dtype = DataType::Int64;
            } else {
                throw std::runtime_error("Unsupported int dtype size: " + std::to_string(itemsize));
            }
        } else if (kind == 'u' && itemsize == 2) {
            // uint16 can be used for bfloat16 storage
            dtype = DataType::BFloat16;
        } else {
            throw std::runtime_error("Unsupported numpy dtype");
        }

        // Get shape
        std::vector<size_t> shape(arr.ndim());
        for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
            shape[i] = arr.shape(i);
        }

        return from_host(arr.data(), shape, dtype);
    }, py::arg("array"), "Create a GPUArray from a numpy array");

    // Stream class
    py::class_<Stream>(m, "Stream")
        .def(py::init<StreamPriority>(), py::arg("priority") = StreamPriority::Low)
        .def("synchronize", &Stream::synchronize)
        .def_property_readonly("priority", &Stream::priority)
        .def("__repr__", [](const Stream& self) {
            return std::string("Stream(priority=") +
                   (self.priority() == StreamPriority::High ? "High" : "Low") + ")";
        });

    // CudaGraph class for optimized decode
    py::class_<CudaGraph>(m, "CudaGraph")
        .def(py::init<>(),
             "Create a CUDA Graph for capturing and replaying operations.\n\n"
             "CUDA Graphs reduce kernel launch overhead by capturing a sequence of\n"
             "operations and replaying them with minimal CPU involvement.\n\n"
             "Usage:\n"
             "  graph = CudaGraph()\n"
             "  graph.begin_capture()\n"
             "  # ... execute operations to capture ...\n"
             "  graph.end_capture()\n"
             "  graph.replay()  # Fast execution")
        .def("begin_capture", &CudaGraph::begin_capture,
             "Begin capturing CUDA operations.\n"
             "All subsequent CUDA operations will be recorded into the graph.")
        .def("end_capture", &CudaGraph::end_capture,
             "End capturing and create an executable graph.\n"
             "After this call, the graph can be replayed.")
        .def("replay", &CudaGraph::replay,
             "Replay the captured graph (asynchronous).\n"
             "Executes all captured operations with minimal CPU overhead.\n"
             "Call synchronize() after replay to wait for completion.")
        .def("synchronize", &CudaGraph::synchronize,
             "Synchronize the graph's internal stream.\n"
             "Call this after replay() to wait for the graph execution to complete.")
        .def("reset", &CudaGraph::reset,
             "Reset the graph, freeing all resources.\n"
             "After reset, begin_capture() can be called again.")
        .def("is_ready", &CudaGraph::is_ready,
             "Check if the graph has been captured and is ready for replay.")
        .def("is_capturing", &CudaGraph::is_capturing,
             "Check if the graph is currently capturing operations.")
        .def_property_readonly("num_nodes", &CudaGraph::num_nodes,
             "Get the number of nodes in the captured graph.")
        .def("__repr__", [](const CudaGraph& self) {
            if (self.is_ready()) {
                return "CudaGraph(ready, nodes=" + std::to_string(self.num_nodes()) + ")";
            } else {
                return std::string("CudaGraph(not ready)");
            }
        });
}
