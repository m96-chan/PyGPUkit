#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../core/device.hpp"
#include "../core/memory.hpp"
#include "../core/stream.hpp"

namespace py = pybind11;
using namespace pygpukit;

void init_core_bindings(py::module_& m) {
    // DataType enum
    py::enum_<DataType>(m, "DataType")
        .value("Float32", DataType::Float32)
        .value("Float64", DataType::Float64)
        .value("Int32", DataType::Int32)
        .value("Int64", DataType::Int64)
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
                case DataType::Int32:
                    result = py::array_t<int32_t>(py_shape);
                    break;
                case DataType::Int64:
                    result = py::array_t<int64_t>(py_shape);
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
        });

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

        // Determine dtype
        DataType dtype;
        if (py::isinstance<py::array_t<float>>(arr)) {
            dtype = DataType::Float32;
        } else if (py::isinstance<py::array_t<double>>(arr)) {
            dtype = DataType::Float64;
        } else if (py::isinstance<py::array_t<int32_t>>(arr)) {
            dtype = DataType::Int32;
        } else if (py::isinstance<py::array_t<int64_t>>(arr)) {
            dtype = DataType::Int64;
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
}
