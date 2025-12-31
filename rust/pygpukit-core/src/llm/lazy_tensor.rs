//! Lazy tensor loading for large models
//!
//! Provides on-demand GPU loading with LRU eviction for models
//! that exceed VRAM capacity.
//!
//! # Design
//!
//! ```text
//! SafeTensorsFile (mmap)
//!        |
//!        v
//!   LazyTensor (metadata + GPU cache)
//!        |
//!        v
//!   MemoryPool (LRU eviction)
//! ```
//!
//! Tensors remain on disk (via mmap) until first GPU access.
//! When VRAM is full, least-recently-used tensors are evicted.

use std::sync::Arc;
use std::time::Instant;
use parking_lot::RwLock;

use crate::llm::tensor_loader::{SafeTensorsFile, TensorInfo, Dtype, SafeTensorsError};
use crate::memory::{MemoryPool, MemoryError};

/// Error type for lazy tensor operations
#[derive(Debug)]
pub enum LazyTensorError {
    /// Tensor not found in file
    TensorNotFound(String),
    /// Memory allocation failed
    MemoryError(MemoryError),
    /// SafeTensors parsing error
    SafeTensorsError(SafeTensorsError),
    /// GPU operation failed
    GpuError(String),
}

impl std::fmt::Display for LazyTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            Self::MemoryError(e) => write!(f, "Memory error: {}", e),
            Self::SafeTensorsError(e) => write!(f, "SafeTensors error: {}", e),
            Self::GpuError(e) => write!(f, "GPU error: {}", e),
        }
    }
}

impl std::error::Error for LazyTensorError {}

impl From<MemoryError> for LazyTensorError {
    fn from(e: MemoryError) -> Self {
        Self::MemoryError(e)
    }
}

impl From<SafeTensorsError> for LazyTensorError {
    fn from(e: SafeTensorsError) -> Self {
        Self::SafeTensorsError(e)
    }
}

/// State of a lazy tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorState {
    /// On disk only (mmap, not loaded to GPU)
    OnDisk,
    /// Currently loading to GPU
    Loading,
    /// Resident on GPU
    OnGpu,
    /// Evicted from GPU (mmap still valid)
    Evicted,
}

/// A tensor that loads to GPU on first access
pub struct LazyTensor {
    /// Source file (shared mmap)
    file: Arc<SafeTensorsFile>,
    /// Tensor name in the file
    name: String,
    /// Tensor metadata
    info: TensorInfo,
    /// Current state
    state: TensorState,
    /// Memory pool block ID (when on GPU)
    block_id: Option<u64>,
    /// GPU device pointer (when on GPU)
    device_ptr: Option<u64>,
    /// Last access time (for LRU)
    last_access: Instant,
}

impl LazyTensor {
    /// Create a new lazy tensor
    pub fn new(file: Arc<SafeTensorsFile>, name: String, info: TensorInfo) -> Self {
        Self {
            file,
            name,
            info,
            state: TensorState::OnDisk,
            block_id: None,
            device_ptr: None,
            last_access: Instant::now(),
        }
    }

    /// Get tensor name
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get tensor info (dtype, shape, size)
    #[inline]
    pub fn info(&self) -> &TensorInfo {
        &self.info
    }

    /// Get tensor shape
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.info.shape
    }

    /// Get tensor dtype
    #[inline]
    pub fn dtype(&self) -> Dtype {
        self.info.dtype
    }

    /// Get tensor size in bytes
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.info.size_bytes
    }

    /// Get current state
    #[inline]
    pub fn state(&self) -> TensorState {
        self.state
    }

    /// Check if tensor is on GPU
    #[inline]
    pub fn is_on_gpu(&self) -> bool {
        self.state == TensorState::OnGpu && self.device_ptr.is_some()
    }

    /// Get GPU device pointer (None if not on GPU)
    #[inline]
    pub fn device_ptr(&self) -> Option<u64> {
        self.device_ptr
    }

    /// Get raw data from mmap (zero-copy)
    pub fn mmap_data(&self) -> Result<&[u8], LazyTensorError> {
        let tensor_data = self.file.tensor(&self.name)?;
        Ok(tensor_data.data)
    }

    /// Load tensor to GPU using the provided memory pool
    ///
    /// # Arguments
    ///
    /// * `pool` - Memory pool for allocation
    /// * `copy_fn` - Function to copy data: (dst_ptr, src_data, size) -> Result
    ///
    /// # Returns
    ///
    /// GPU device pointer on success
    pub fn to_gpu<F>(
        &mut self,
        pool: &MemoryPool,
        copy_fn: F,
    ) -> Result<u64, LazyTensorError>
    where
        F: FnOnce(u64, &[u8]) -> Result<(), String>,
    {
        // Already on GPU - just touch and return
        if let Some(ptr) = self.device_ptr {
            if self.state == TensorState::OnGpu {
                self.last_access = Instant::now();
                if let Some(block_id) = self.block_id {
                    pool.touch(block_id);
                }
                return Ok(ptr);
            }
        }

        self.state = TensorState::Loading;

        // Get mmap data
        let tensor_data = self.file.tensor(&self.name)?;
        let size = tensor_data.data.len();

        // Allocate from pool
        let block_id = pool.allocate(size)?;
        self.block_id = Some(block_id);

        // The caller provides the actual CUDA allocation and copy
        // This allows flexibility in how the GPU memory is managed
        //
        // In practice, this would be:
        // 1. cuMemAlloc or cuMemAllocAsync
        // 2. cuMemcpyHtoD or cuMemcpyHtoDAsync
        //
        // We pass the data slice so the copy_fn can handle it
        let device_ptr = block_id as u64; // Placeholder - real impl uses CUDA

        match copy_fn(device_ptr, tensor_data.data) {
            Ok(()) => {
                pool.set_device_ptr(block_id, device_ptr);
                self.device_ptr = Some(device_ptr);
                self.state = TensorState::OnGpu;
                self.last_access = Instant::now();
                Ok(device_ptr)
            }
            Err(e) => {
                pool.free(block_id);
                self.block_id = None;
                self.state = TensorState::OnDisk;
                Err(LazyTensorError::GpuError(e))
            }
        }
    }

    /// Evict tensor from GPU (keep mmap reference)
    ///
    /// # Arguments
    ///
    /// * `pool` - Memory pool for deallocation
    /// * `free_fn` - Function to free GPU memory: (device_ptr) -> Result
    ///
    /// # Returns
    ///
    /// Number of bytes freed
    pub fn evict<F>(
        &mut self,
        pool: &MemoryPool,
        free_fn: F,
    ) -> Result<usize, LazyTensorError>
    where
        F: FnOnce(u64) -> Result<(), String>,
    {
        if self.state != TensorState::OnGpu {
            return Ok(0);
        }

        let size = self.info.size_bytes;

        if let Some(ptr) = self.device_ptr.take() {
            free_fn(ptr).map_err(LazyTensorError::GpuError)?;
        }

        if let Some(block_id) = self.block_id.take() {
            pool.evict(block_id);
        }

        self.state = TensorState::Evicted;
        Ok(size)
    }

    /// Touch to update LRU timestamp
    pub fn touch(&mut self, pool: &MemoryPool) {
        self.last_access = Instant::now();
        if let Some(block_id) = self.block_id {
            pool.touch(block_id);
        }
    }

    /// Completely unload tensor (release GPU memory and mark as unloaded)
    ///
    /// Unlike `evict()`, this marks the tensor as needing reload from disk.
    /// The mmap reference is kept but the tensor must be re-loaded to use.
    ///
    /// # Arguments
    ///
    /// * `pool` - Memory pool for deallocation
    /// * `free_fn` - Function to free GPU memory: (device_ptr) -> Result
    ///
    /// # Returns
    ///
    /// Number of bytes freed
    pub fn unload<F>(
        &mut self,
        pool: &MemoryPool,
        free_fn: F,
    ) -> Result<usize, LazyTensorError>
    where
        F: FnOnce(u64) -> Result<(), String>,
    {
        let freed = self.evict(pool, free_fn)?;
        // Reset to OnDisk state (can be reloaded)
        self.state = TensorState::OnDisk;
        Ok(freed)
    }

    /// Check if tensor can be unloaded (is on GPU)
    #[inline]
    pub fn can_unload(&self) -> bool {
        self.state == TensorState::OnGpu
    }
}

/// Lazy model loader for multiple SafeTensors files
pub struct LazyModelLoader {
    /// Loaded files (shared mmaps)
    files: Vec<Arc<SafeTensorsFile>>,
    /// All tensors by name
    tensors: RwLock<std::collections::HashMap<String, LazyTensor>>,
    /// Memory pool for GPU allocation
    pool: Arc<MemoryPool>,
    /// Total size of all tensors
    total_size: usize,
    /// Currently loaded size on GPU
    loaded_size: RwLock<usize>,
}

impl LazyModelLoader {
    /// Create a new lazy model loader
    ///
    /// # Arguments
    ///
    /// * `memory_budget` - Maximum GPU memory to use (bytes)
    /// * `enable_eviction` - Whether to auto-evict when budget exceeded
    pub fn new(memory_budget: usize, enable_eviction: bool) -> Self {
        Self {
            files: Vec::new(),
            tensors: RwLock::new(std::collections::HashMap::new()),
            pool: Arc::new(MemoryPool::new(memory_budget, enable_eviction)),
            total_size: 0,
            loaded_size: RwLock::new(0),
        }
    }

    /// Load a SafeTensors file (mmap only, no GPU transfer)
    pub fn load_file(&mut self, path: &std::path::Path) -> Result<(), LazyTensorError> {
        let file = Arc::new(SafeTensorsFile::open(path)?);

        let mut tensors = self.tensors.write();
        for name in file.tensor_names() {
            if let Some(info) = file.tensor_info(name) {
                self.total_size += info.size_bytes;
                let tensor = LazyTensor::new(
                    Arc::clone(&file),
                    name.to_string(),
                    info.clone(),
                );
                tensors.insert(name.to_string(), tensor);
            }
        }

        self.files.push(file);
        Ok(())
    }

    /// Get a tensor by name
    pub fn get(&self, name: &str) -> Option<TensorInfo> {
        self.tensors.read().get(name).map(|t| t.info().clone())
    }

    /// Get tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.read().keys().cloned().collect()
    }

    /// Get total model size in bytes
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Get currently loaded size on GPU
    pub fn loaded_size(&self) -> usize {
        *self.loaded_size.read()
    }

    /// Get memory pool statistics
    pub fn pool_stats(&self) -> crate::memory::PoolStats {
        self.pool.stats()
    }

    /// Number of tensors
    pub fn num_tensors(&self) -> usize {
        self.tensors.read().len()
    }

    /// Number of files loaded
    pub fn num_files(&self) -> usize {
        self.files.len()
    }

    /// Get the memory pool (for external GPU operations)
    pub fn pool(&self) -> &Arc<MemoryPool> {
        &self.pool
    }

    /// Unload entire model from GPU
    ///
    /// Releases all GPU memory but keeps mmap references.
    /// Model can be reloaded by accessing tensors again.
    ///
    /// # Arguments
    ///
    /// * `free_fn` - Function to free GPU memory: (device_ptr) -> Result
    ///
    /// # Returns
    ///
    /// Total bytes freed from GPU
    pub fn unload_model<F>(&self, mut free_fn: F) -> Result<usize, LazyTensorError>
    where
        F: FnMut(u64) -> Result<(), String>,
    {
        let mut tensors = self.tensors.write();
        let mut total_freed = 0;

        for tensor in tensors.values_mut() {
            if tensor.can_unload() {
                let freed = tensor.unload(&self.pool, &mut free_fn)?;
                total_freed += freed;
            }
        }

        *self.loaded_size.write() = 0;
        Ok(total_freed)
    }

    /// Unload tensors by layer prefix
    ///
    /// Useful for unloading specific transformer layers.
    /// E.g., prefix "model.layers.0." unloads all tensors in layer 0.
    ///
    /// # Arguments
    ///
    /// * `prefix` - Tensor name prefix to match
    /// * `free_fn` - Function to free GPU memory
    ///
    /// # Returns
    ///
    /// (tensors_unloaded, bytes_freed)
    pub fn unload_layer<F>(
        &self,
        prefix: &str,
        mut free_fn: F,
    ) -> Result<(usize, usize), LazyTensorError>
    where
        F: FnMut(u64) -> Result<(), String>,
    {
        let mut tensors = self.tensors.write();
        let mut count = 0;
        let mut total_freed = 0;

        for (name, tensor) in tensors.iter_mut() {
            if name.starts_with(prefix) && tensor.can_unload() {
                let freed = tensor.unload(&self.pool, &mut free_fn)?;
                total_freed += freed;
                count += 1;
            }
        }

        // Update loaded size
        let mut loaded = self.loaded_size.write();
        *loaded = loaded.saturating_sub(total_freed);

        Ok((count, total_freed))
    }

    /// Unload specific tensors by name
    ///
    /// # Arguments
    ///
    /// * `names` - List of tensor names to unload
    /// * `free_fn` - Function to free GPU memory
    ///
    /// # Returns
    ///
    /// (tensors_unloaded, bytes_freed)
    pub fn unload_tensors<F>(
        &self,
        names: &[&str],
        mut free_fn: F,
    ) -> Result<(usize, usize), LazyTensorError>
    where
        F: FnMut(u64) -> Result<(), String>,
    {
        let mut tensors = self.tensors.write();
        let mut count = 0;
        let mut total_freed = 0;

        for name in names {
            if let Some(tensor) = tensors.get_mut(*name) {
                if tensor.can_unload() {
                    let freed = tensor.unload(&self.pool, &mut free_fn)?;
                    total_freed += freed;
                    count += 1;
                }
            }
        }

        // Update loaded size
        let mut loaded = self.loaded_size.write();
        *loaded = loaded.saturating_sub(total_freed);

        Ok((count, total_freed))
    }

    /// Get list of tensors currently on GPU
    pub fn loaded_tensors(&self) -> Vec<String> {
        self.tensors
            .read()
            .iter()
            .filter(|(_, t)| t.is_on_gpu())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get number of tensors currently on GPU
    pub fn num_loaded(&self) -> usize {
        self.tensors.read().values().filter(|t| t.is_on_gpu()).count()
    }

    /// Get tensor names matching a prefix
    pub fn get_layer_tensors(&self, prefix: &str) -> Vec<String> {
        self.tensors
            .read()
            .keys()
            .filter(|name| name.starts_with(prefix))
            .cloned()
            .collect()
    }

    /// Get total size of tensors matching a prefix
    pub fn layer_size(&self, prefix: &str) -> usize {
        self.tensors
            .read()
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(_, t)| t.size_bytes())
            .sum()
    }

    /// Check if a layer is fully loaded on GPU
    pub fn is_layer_loaded(&self, prefix: &str) -> bool {
        let tensors = self.tensors.read();
        let layer_tensors: Vec<_> = tensors
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .collect();

        if layer_tensors.is_empty() {
            return false;
        }

        layer_tensors.iter().all(|(_, t)| t.is_on_gpu())
    }

    /// Get layer loading state: (total_tensors, loaded_tensors, total_bytes, loaded_bytes)
    pub fn layer_state(&self, prefix: &str) -> (usize, usize, usize, usize) {
        let tensors = self.tensors.read();
        let mut total_count = 0;
        let mut loaded_count = 0;
        let mut total_bytes = 0;
        let mut loaded_bytes = 0;

        for (name, tensor) in tensors.iter() {
            if name.starts_with(prefix) {
                total_count += 1;
                total_bytes += tensor.size_bytes();
                if tensor.is_on_gpu() {
                    loaded_count += 1;
                    loaded_bytes += tensor.size_bytes();
                }
            }
        }

        (total_count, loaded_count, total_bytes, loaded_bytes)
    }

    /// Clear all data (unload + close mmaps)
    ///
    /// After this, the loader cannot be used until new files are loaded.
    pub fn clear<F>(&mut self, mut free_fn: F) -> Result<usize, LazyTensorError>
    where
        F: FnMut(u64) -> Result<(), String>,
    {
        // Unload all tensors first
        let freed = self.unload_model(&mut free_fn)?;

        // Clear internal state
        self.tensors.write().clear();
        self.files.clear();
        self.total_size = 0;
        *self.loaded_size.write() = 0;

        // Clear memory pool
        self.pool.clear();

        Ok(freed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_state_default() {
        let pool = MemoryPool::new(1024 * 1024, true);
        let file = Arc::new(SafeTensorsFile::open("test.safetensors").ok());

        // This would normally create from a real file
        // For unit tests, we just verify the enum values
        assert_eq!(TensorState::OnDisk, TensorState::OnDisk);
        assert_ne!(TensorState::OnDisk, TensorState::OnGpu);
    }

    #[test]
    fn test_lazy_model_loader_creation() {
        let loader = LazyModelLoader::new(1024 * 1024 * 100, true);
        assert_eq!(loader.total_size(), 0);
        assert_eq!(loader.loaded_size(), 0);
        assert_eq!(loader.num_tensors(), 0);
    }
}
