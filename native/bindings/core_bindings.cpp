#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cuda.h>

#include "../core/device.hpp"
#include "../core/memory.hpp"
#include "../core/stream.hpp"
#include "../core/event.hpp"
#include "../core/cuda_graph.hpp"

namespace py = pybind11;
using namespace pygpukit;

void init_core_bindings(py::module_& m) {
    // DataType enum
    py::enum_<DataType>(m, "DataType")
        .value("Float64", DataType::Float64)
        .value("Float32", DataType::Float32)
        .value("Float16", DataType::Float16)
        .value("BFloat16", DataType::BFloat16)
        .value("Int64", DataType::Int64)
        .value("Int32", DataType::Int32)
        .value("Int16", DataType::Int16)
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
                case DataType::Float64:
                    result = py::array_t<double>(py_shape);
                    break;
                case DataType::Float32:
                    result = py::array_t<float>(py_shape);
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
                case DataType::Int64:
                    result = py::array_t<int64_t>(py_shape);
                    break;
                case DataType::Int32:
                    result = py::array_t<int32_t>(py_shape);
                    break;
                case DataType::Int16:
                    result = py::array_t<int16_t>(py_shape);
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
        .def("data_ptr", [](const GPUArray& self) {
            return reinterpret_cast<uintptr_t>(self.data());
        }, "Get the raw device pointer as an integer")
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
            if (itemsize == 8) {
                dtype = DataType::Int64;
            } else if (itemsize == 4) {
                dtype = DataType::Int32;
            } else if (itemsize == 2) {
                dtype = DataType::Int16;
            } else if (itemsize == 1) {
                dtype = DataType::Int8;
            } else {
                throw std::runtime_error("Unsupported int dtype size: " + std::to_string(itemsize));
            }
        } else if (kind == 'u') {
            // Unsigned integer types
            if (itemsize == 1) {
                dtype = DataType::UInt8;
            } else if (itemsize == 2) {
                // uint16 can be used for bfloat16 storage
                dtype = DataType::BFloat16;
            } else {
                throw std::runtime_error("Unsupported uint dtype size: " + std::to_string(itemsize));
            }
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

    // CudaEvent class for GPU-side timing
    py::class_<CudaEvent>(m, "CudaEvent")
        .def(py::init<bool>(), py::arg("blocking_sync") = false,
             "Create a CUDA event for GPU-side timing.\n\n"
             "Args:\n"
             "    blocking_sync: If True, synchronize() will block CPU. Default False.\n\n"
             "Usage for timing:\n"
             "    start = CudaEvent()\n"
             "    stop = CudaEvent()\n"
             "    start.record()\n"
             "    # ... GPU operations ...\n"
             "    stop.record()\n"
             "    stop.synchronize()\n"
             "    elapsed_ms = event_elapsed_ms(start, stop)")
        .def("record", py::overload_cast<const Stream&>(&CudaEvent::record),
             py::arg("stream"),
             "Record event in the specified stream.")
        .def("record", py::overload_cast<>(&CudaEvent::record),
             "Record event in the default stream.")
        .def("synchronize", &CudaEvent::synchronize,
             "Wait for the event to complete.")
        .def("query", &CudaEvent::query,
             "Check if the event has completed (non-blocking).")
        .def("__repr__", [](const CudaEvent& self) {
            return std::string("CudaEvent()");
        });

    // Event timing functions
    m.def("event_elapsed_ms", &event_elapsed_ms,
          py::arg("start"), py::arg("stop"),
          "Get elapsed time between two events in milliseconds.\n"
          "Both events must have been recorded and stop must be synchronized.");
    m.def("event_elapsed_us", &event_elapsed_us,
          py::arg("start"), py::arg("stop"),
          "Get elapsed time between two events in microseconds.\n"
          "Both events must have been recorded and stop must be synchronized.");

    // Async memory transfer from host pointer to GPUArray
    m.def("memcpy_to_device_async", [](GPUArray& dst, py::buffer src, const Stream& stream) {
        py::buffer_info info = src.request();
        if (static_cast<size_t>(info.size * info.itemsize) != dst.nbytes()) {
            throw std::runtime_error("Buffer size mismatch");
        }
        memcpy_host_to_device_async(dst.data(), info.ptr, dst.nbytes(), stream.handle());
    }, py::arg("dst"), py::arg("src"), py::arg("stream"),
    "Async copy from host buffer to GPUArray. src must be pinned memory for true async.");

    // Async memcpy from raw pointer (integer address) to GPUArray
    m.def("memcpy_ptr_to_device_async",
        [](GPUArray& dst, uintptr_t src_ptr, size_t size_bytes, const Stream& stream) {
            if (size_bytes > dst.nbytes()) {
                throw std::runtime_error("Size exceeds destination capacity");
            }
            memcpy_host_to_device_async(dst.data(), reinterpret_cast<const void*>(src_ptr),
                                        size_bytes, stream.handle());
        },
        py::arg("dst"), py::arg("src_ptr"), py::arg("size_bytes"), py::arg("stream"),
        "Async copy from raw host pointer to GPUArray.\n"
        "Note: For true async behavior, src_ptr should point to pinned memory.");

    // Async memcpy using raw stream handle (for CUDA Graph stream)
    m.def("memcpy_ptr_to_device_async_raw_stream",
        [](GPUArray& dst, uintptr_t src_ptr, size_t size_bytes, uintptr_t stream_handle) {
            if (size_bytes > dst.nbytes()) {
                throw std::runtime_error("Size exceeds destination capacity");
            }
            CUstream stream = reinterpret_cast<CUstream>(stream_handle);
            memcpy_host_to_device_async(dst.data(), reinterpret_cast<const void*>(src_ptr),
                                        size_bytes, stream);
        },
        py::arg("dst"), py::arg("src_ptr"), py::arg("size_bytes"), py::arg("stream_handle"),
        "Async copy from raw host pointer to GPUArray using raw stream handle.\n"
        "Used for CUDA Graph's internal stream.");

    // Sync memcpy from raw pointer (for mmap'd data)
    m.def("memcpy_ptr_to_device",
        [](GPUArray& dst, uintptr_t src_ptr, size_t size_bytes) {
            if (size_bytes > dst.nbytes()) {
                throw std::runtime_error("Size exceeds destination capacity");
            }
            memcpy_host_to_device(dst.data(), reinterpret_cast<const void*>(src_ptr), size_bytes);
        },
        py::arg("dst"), py::arg("src_ptr"), py::arg("size_bytes"),
        "Copy from raw host pointer (e.g., mmap'd memory) to GPUArray.");

    // Device-to-device async
    m.def("memcpy_device_to_device_async",
        [](GPUArray& dst, const GPUArray& src, const Stream& stream) {
            if (dst.nbytes() != src.nbytes()) {
                throw std::runtime_error("Array size mismatch");
            }
            memcpy_device_to_device_async(dst.data(), src.data(), src.nbytes(), stream.handle());
        },
        py::arg("dst"), py::arg("src"), py::arg("stream"),
        "Async copy between GPUArrays on the same device.");

    // Device-to-device with offset (for stacking arrays)
    m.def("memcpy_device_to_device_offset",
        [](const GPUArray& src, GPUArray& dst, size_t src_offset, size_t dst_offset, size_t size_bytes) {
            if (src_offset + size_bytes > src.nbytes()) {
                throw std::runtime_error("Source offset + size exceeds source array bounds");
            }
            if (dst_offset + size_bytes > dst.nbytes()) {
                throw std::runtime_error("Destination offset + size exceeds destination array bounds");
            }
            CUdeviceptr src_ptr = reinterpret_cast<CUdeviceptr>(src.data()) + src_offset;
            CUdeviceptr dst_ptr = reinterpret_cast<CUdeviceptr>(dst.data()) + dst_offset;
            CUresult err = cuMemcpy(dst_ptr, src_ptr, size_bytes);
            if (err != CUDA_SUCCESS) {
                const char* error_str = nullptr;
                cuGetErrorString(err, &error_str);
                throw std::runtime_error(std::string("cuMemcpy failed: ") + (error_str ? error_str : "unknown"));
            }
        },
        py::arg("src"), py::arg("dst"), py::arg("src_offset"), py::arg("dst_offset"), py::arg("size_bytes"),
        "Copy from src[src_offset:] to dst[dst_offset:] on device.");

    // Synchronize a raw stream handle (using Driver API)
    m.def("stream_synchronize_raw",
        [](uintptr_t stream_handle) {
            CUstream stream = reinterpret_cast<CUstream>(stream_handle);
            CUresult err = cuStreamSynchronize(stream);
            if (err != CUDA_SUCCESS) {
                const char* error_str = nullptr;
                cuGetErrorString(err, &error_str);
                throw std::runtime_error(std::string("Stream synchronize failed: ") +
                                         (error_str ? error_str : "unknown error"));
            }
        },
        py::arg("stream_handle"),
        "Synchronize a stream using its raw handle.");

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
        .def("get_stream_handle", [](const CudaGraph& self) {
            return reinterpret_cast<uintptr_t>(self.get_stream_handle());
        }, "Get the internal stream handle as an integer for async operations.")
        .def("__repr__", [](const CudaGraph& self) {
            if (self.is_ready()) {
                return "CudaGraph(ready, nodes=" + std::to_string(self.num_nodes()) + ")";
            } else {
                return std::string("CudaGraph(not ready)");
            }
        });
}
