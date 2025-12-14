//! PyO3 bindings for device capabilities and kernel types

use pyo3::prelude::*;
use pygpukit_core::device::{DeviceCapabilities, KernelType};

/// Python-exposed kernel type enum
#[pyclass(name = "KernelType")]
#[derive(Clone)]
pub struct PyKernelType {
    inner: KernelType,
}

#[pymethods]
impl PyKernelType {
    /// FP32 FMA kernel
    #[classattr]
    fn FP32_FMA() -> Self {
        Self { inner: KernelType::Fp32Fma }
    }

    /// TF32 MMA (TensorCore) kernel
    #[classattr]
    fn TF32_MMA() -> Self {
        Self { inner: KernelType::Tf32Mma }
    }

    /// FP16 MMA (TensorCore) kernel
    #[classattr]
    fn FP16_MMA() -> Self {
        Self { inner: KernelType::Fp16Mma }
    }

    /// BF16 MMA (TensorCore) kernel
    #[classattr]
    fn BF16_MMA() -> Self {
        Self { inner: KernelType::Bf16Mma }
    }

    /// L2-optimized naive kernel
    #[classattr]
    fn L2_NAIVE() -> Self {
        Self { inner: KernelType::L2Naive }
    }

    /// Tiled shared memory kernel
    #[classattr]
    fn TILED_SMEM() -> Self {
        Self { inner: KernelType::TiledSmem }
    }

    /// Check if this kernel type uses Tensor Cores
    fn uses_tensor_cores(&self) -> bool {
        self.inner.uses_tensor_cores()
    }

    /// Get the minimum SM version required
    fn min_sm_version(&self) -> u32 {
        self.inner.min_sm_version()
    }

    /// Get human-readable name
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        format!("KernelType.{}", match self.inner {
            KernelType::Fp32Fma => "FP32_FMA",
            KernelType::Tf32Mma => "TF32_MMA",
            KernelType::Fp16Mma => "FP16_MMA",
            KernelType::Bf16Mma => "BF16_MMA",
            KernelType::L2Naive => "L2_NAIVE",
            KernelType::TiledSmem => "TILED_SMEM",
        })
    }

    fn __str__(&self) -> String {
        self.inner.name().to_string()
    }

    fn __eq__(&self, other: &PyKernelType) -> bool {
        self.inner == other.inner
    }
}

impl From<KernelType> for PyKernelType {
    fn from(inner: KernelType) -> Self {
        Self { inner }
    }
}

impl From<&PyKernelType> for KernelType {
    fn from(py_type: &PyKernelType) -> Self {
        py_type.inner
    }
}

/// Python-exposed device capabilities
#[pyclass(name = "DeviceCapabilities")]
#[derive(Clone)]
pub struct PyDeviceCapabilities {
    inner: DeviceCapabilities,
}

#[pymethods]
impl PyDeviceCapabilities {
    /// Create capabilities from SM version
    #[new]
    #[pyo3(signature = (sm_version=86))]
    fn new(sm_version: u32) -> Self {
        Self {
            inner: DeviceCapabilities::from_sm_version(sm_version),
        }
    }

    /// Create Ampere device capabilities
    #[staticmethod]
    fn ampere() -> Self {
        Self {
            inner: DeviceCapabilities::ampere(),
        }
    }

    /// Create Ada device capabilities
    #[staticmethod]
    fn ada() -> Self {
        Self {
            inner: DeviceCapabilities::ada(),
        }
    }

    /// Create Hopper device capabilities
    #[staticmethod]
    fn hopper() -> Self {
        Self {
            inner: DeviceCapabilities::hopper(),
        }
    }

    /// Device ID
    #[getter]
    fn device_id(&self) -> u32 {
        self.inner.device_id
    }

    /// Device name
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// SM version (e.g., 86 for SM 8.6)
    #[getter]
    fn sm_version(&self) -> u32 {
        self.inner.sm_version
    }

    /// Compute capability (alias for sm_version)
    #[getter]
    fn compute_capability(&self) -> u32 {
        self.inner.sm_version
    }

    /// Compute major version
    #[getter]
    fn compute_major(&self) -> u32 {
        self.inner.compute_major
    }

    /// Compute minor version
    #[getter]
    fn compute_minor(&self) -> u32 {
        self.inner.compute_minor
    }

    /// Whether TF32 Tensor Cores are available (SM >= 80)
    #[getter]
    fn tensorcore(&self) -> bool {
        self.inner.tensorcore
    }

    /// Whether FP16 Tensor Cores are available (SM >= 70)
    #[getter]
    fn tensorcore_fp16(&self) -> bool {
        self.inner.tensorcore_fp16
    }

    /// Whether BF16 Tensor Cores are available (SM >= 80)
    #[getter]
    fn tensorcore_bf16(&self) -> bool {
        self.inner.tensorcore_bf16
    }

    /// Total global memory in bytes
    #[getter]
    fn total_memory(&self) -> u64 {
        self.inner.total_memory
    }

    /// L2 cache size in bytes
    #[getter]
    fn l2_cache_size(&self) -> u32 {
        self.inner.l2_cache_size
    }

    /// Shared memory per block in bytes
    #[getter]
    fn shared_mem_per_block(&self) -> u32 {
        self.inner.shared_mem_per_block
    }

    /// Maximum threads per block
    #[getter]
    fn max_threads_per_block(&self) -> u32 {
        self.inner.max_threads_per_block
    }

    /// Number of SMs
    #[getter]
    fn sm_count(&self) -> u32 {
        self.inner.sm_count
    }

    /// Warp size
    #[getter]
    fn warp_size(&self) -> u32 {
        self.inner.warp_size
    }

    /// Whether async copy (cp.async) is supported
    #[getter]
    fn async_copy(&self) -> bool {
        self.inner.async_copy
    }

    /// Check if a kernel type is supported
    fn supports_kernel(&self, kernel_type: &PyKernelType) -> bool {
        self.inner.supports_kernel(kernel_type.inner)
    }

    /// Get the best matmul kernel type
    #[pyo3(signature = (use_tf32=false, dtype_is_fp32=true, large_matrix=true))]
    fn best_matmul_kernel(&self, use_tf32: bool, dtype_is_fp32: bool, large_matrix: bool) -> PyKernelType {
        self.inner.best_matmul_kernel(use_tf32, dtype_is_fp32, large_matrix).into()
    }

    /// Check if this is Ampere or newer
    fn is_ampere_or_newer(&self) -> bool {
        self.inner.is_ampere_or_newer()
    }

    fn __repr__(&self) -> String {
        format!(
            "DeviceCapabilities(sm_version={}, tensorcore={}, name='{}')",
            self.inner.sm_version,
            self.inner.tensorcore,
            self.inner.name
        )
    }
}

impl From<DeviceCapabilities> for PyDeviceCapabilities {
    fn from(inner: DeviceCapabilities) -> Self {
        Self { inner }
    }
}

impl From<&PyDeviceCapabilities> for DeviceCapabilities {
    fn from(py_caps: &PyDeviceCapabilities) -> Self {
        py_caps.inner.clone()
    }
}

/// Register device module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKernelType>()?;
    m.add_class::<PyDeviceCapabilities>()?;
    Ok(())
}
