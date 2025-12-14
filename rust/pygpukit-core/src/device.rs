//! Device capabilities and kernel type definitions
//!
//! Provides GPU device capability detection and kernel type enumeration
//! for selecting appropriate kernel implementations.

/// GPU kernel type enumeration
///
/// Represents different kernel implementations available for GPU operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    /// FP32 FMA (Fused Multiply-Add) kernel
    /// Standard FP32 precision, used for maximum accuracy
    Fp32Fma,

    /// TF32 MMA (Matrix Multiply-Accumulate) kernel using Tensor Cores
    /// Uses TF32 precision (19-bit mantissa) for faster computation
    /// Only available on Ampere (SM80+) and newer GPUs
    Tf32Mma,

    /// FP16 MMA kernel using Tensor Cores
    /// Uses FP16 precision for maximum throughput
    Fp16Mma,

    /// BF16 MMA kernel using Tensor Cores
    /// Uses BF16 precision (8-bit exponent, 7-bit mantissa)
    Bf16Mma,

    /// L2-optimized naive kernel
    /// Simple kernel optimized for L2 cache locality
    L2Naive,

    /// Tiled shared memory kernel
    /// Uses shared memory tiling for memory bandwidth optimization
    TiledSmem,
}

impl KernelType {
    /// Check if this kernel type uses Tensor Cores
    pub fn uses_tensor_cores(&self) -> bool {
        matches!(self, KernelType::Tf32Mma | KernelType::Fp16Mma | KernelType::Bf16Mma)
    }

    /// Get the minimum SM version required for this kernel type
    pub fn min_sm_version(&self) -> u32 {
        match self {
            KernelType::Fp32Fma => 60,      // Pascal
            KernelType::L2Naive => 60,      // Pascal
            KernelType::TiledSmem => 60,    // Pascal
            KernelType::Tf32Mma => 80,      // Ampere
            KernelType::Fp16Mma => 70,      // Volta (but better on Ampere)
            KernelType::Bf16Mma => 80,      // Ampere
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            KernelType::Fp32Fma => "FP32 FMA",
            KernelType::Tf32Mma => "TF32 MMA (TensorCore)",
            KernelType::Fp16Mma => "FP16 MMA (TensorCore)",
            KernelType::Bf16Mma => "BF16 MMA (TensorCore)",
            KernelType::L2Naive => "L2 Naive",
            KernelType::TiledSmem => "Tiled Shared Memory",
        }
    }
}

impl std::fmt::Display for KernelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// GPU Device Capabilities
///
/// Contains information about GPU hardware capabilities
/// used for kernel selection and optimization decisions.
#[derive(Debug, Clone, Default)]
pub struct DeviceCapabilities {
    /// Device index
    pub device_id: u32,

    /// Device name (e.g., "NVIDIA GeForce RTX 3090 Ti")
    pub name: String,

    /// SM (Streaming Multiprocessor) version
    /// Computed as major * 10 + minor (e.g., SM 8.6 = 86)
    pub sm_version: u32,

    /// Compute capability major version
    pub compute_major: u32,

    /// Compute capability minor version
    pub compute_minor: u32,

    /// Whether TF32 Tensor Cores are available (SM >= 80)
    pub tensorcore: bool,

    /// Whether FP16 Tensor Cores are available (SM >= 70)
    pub tensorcore_fp16: bool,

    /// Whether BF16 Tensor Cores are available (SM >= 80)
    pub tensorcore_bf16: bool,

    /// Total global memory in bytes
    pub total_memory: u64,

    /// L2 cache size in bytes
    pub l2_cache_size: u32,

    /// Shared memory per block in bytes
    pub shared_mem_per_block: u32,

    /// Maximum threads per block
    pub max_threads_per_block: u32,

    /// Number of SMs
    pub sm_count: u32,

    /// Warp size
    pub warp_size: u32,

    /// Whether async copy (cp.async) is supported (SM >= 80)
    pub async_copy: bool,
}

impl DeviceCapabilities {
    /// Create capabilities for a specific SM version
    ///
    /// This is useful for testing or when actual device info is not available.
    pub fn from_sm_version(sm_version: u32) -> Self {
        let compute_major = sm_version / 10;
        let compute_minor = sm_version % 10;

        Self {
            device_id: 0,
            name: format!("SM {}.{}", compute_major, compute_minor),
            sm_version,
            compute_major,
            compute_minor,
            tensorcore: sm_version >= 80,
            tensorcore_fp16: sm_version >= 70,
            tensorcore_bf16: sm_version >= 80,
            total_memory: 0,
            l2_cache_size: 0,
            shared_mem_per_block: 49152, // 48KB default
            max_threads_per_block: 1024,
            sm_count: 0,
            warp_size: 32,
            async_copy: sm_version >= 80,
        }
    }

    /// Create an Ampere (RTX 30xx / A100) device
    pub fn ampere() -> Self {
        Self::from_sm_version(86)
    }

    /// Create an Ada (RTX 40xx) device
    pub fn ada() -> Self {
        Self::from_sm_version(89)
    }

    /// Create a Hopper (H100) device
    pub fn hopper() -> Self {
        Self::from_sm_version(90)
    }

    /// Check if a kernel type is supported
    pub fn supports_kernel(&self, kernel_type: KernelType) -> bool {
        self.sm_version >= kernel_type.min_sm_version()
    }

    /// Get the best kernel type for matmul
    ///
    /// # Arguments
    /// * `use_tf32` - Whether TF32 is allowed
    /// * `dtype_is_fp32` - Whether the data type is FP32
    /// * `large_matrix` - Whether the matrix is large enough for optimized kernels
    pub fn best_matmul_kernel(&self, use_tf32: bool, dtype_is_fp32: bool, large_matrix: bool) -> KernelType {
        if use_tf32 && dtype_is_fp32 && self.tensorcore && large_matrix {
            KernelType::Tf32Mma
        } else if dtype_is_fp32 && large_matrix && self.sm_version >= 80 {
            // Ampere-optimized FP32 FMA
            KernelType::Fp32Fma
        } else if large_matrix {
            KernelType::TiledSmem
        } else {
            KernelType::L2Naive
        }
    }

    /// Check if this is an Ampere or newer GPU
    pub fn is_ampere_or_newer(&self) -> bool {
        self.sm_version >= 80
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_type_tensor_cores() {
        assert!(!KernelType::Fp32Fma.uses_tensor_cores());
        assert!(KernelType::Tf32Mma.uses_tensor_cores());
        assert!(KernelType::Fp16Mma.uses_tensor_cores());
        assert!(KernelType::Bf16Mma.uses_tensor_cores());
        assert!(!KernelType::L2Naive.uses_tensor_cores());
    }

    #[test]
    fn test_kernel_type_min_sm() {
        assert_eq!(KernelType::Fp32Fma.min_sm_version(), 60);
        assert_eq!(KernelType::Tf32Mma.min_sm_version(), 80);
        assert_eq!(KernelType::Fp16Mma.min_sm_version(), 70);
        assert_eq!(KernelType::Bf16Mma.min_sm_version(), 80);
    }

    #[test]
    fn test_device_capabilities_from_sm() {
        let caps = DeviceCapabilities::from_sm_version(86);
        assert_eq!(caps.sm_version, 86);
        assert_eq!(caps.compute_major, 8);
        assert_eq!(caps.compute_minor, 6);
        assert!(caps.tensorcore);
        assert!(caps.tensorcore_fp16);
        assert!(caps.tensorcore_bf16);
        assert!(caps.async_copy);
    }

    #[test]
    fn test_device_capabilities_old_gpu() {
        let caps = DeviceCapabilities::from_sm_version(75);
        assert_eq!(caps.sm_version, 75);
        assert!(!caps.tensorcore);  // TF32 requires SM80
        assert!(caps.tensorcore_fp16);  // FP16 tensor cores on Turing
        assert!(!caps.tensorcore_bf16);  // BF16 requires SM80
        assert!(!caps.async_copy);
    }

    #[test]
    fn test_supports_kernel() {
        let ampere = DeviceCapabilities::ampere();
        assert!(ampere.supports_kernel(KernelType::Fp32Fma));
        assert!(ampere.supports_kernel(KernelType::Tf32Mma));
        assert!(ampere.supports_kernel(KernelType::Fp16Mma));

        let turing = DeviceCapabilities::from_sm_version(75);
        assert!(turing.supports_kernel(KernelType::Fp32Fma));
        assert!(!turing.supports_kernel(KernelType::Tf32Mma));
        assert!(turing.supports_kernel(KernelType::Fp16Mma));
    }

    #[test]
    fn test_best_matmul_kernel_tf32() {
        let ampere = DeviceCapabilities::ampere();

        // TF32 enabled, FP32 dtype, large matrix
        let kernel = ampere.best_matmul_kernel(true, true, true);
        assert_eq!(kernel, KernelType::Tf32Mma);

        // TF32 disabled
        let kernel = ampere.best_matmul_kernel(false, true, true);
        assert_eq!(kernel, KernelType::Fp32Fma);
    }

    #[test]
    fn test_best_matmul_kernel_small_matrix() {
        let ampere = DeviceCapabilities::ampere();

        // Small matrix should use L2 naive
        let kernel = ampere.best_matmul_kernel(true, true, false);
        assert_eq!(kernel, KernelType::L2Naive);
    }

    #[test]
    fn test_is_ampere_or_newer() {
        assert!(DeviceCapabilities::ampere().is_ampere_or_newer());
        assert!(DeviceCapabilities::ada().is_ampere_or_newer());
        assert!(DeviceCapabilities::hopper().is_ampere_or_newer());
        assert!(!DeviceCapabilities::from_sm_version(75).is_ampere_or_newer());
    }
}
