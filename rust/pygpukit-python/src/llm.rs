//! Python bindings for LLM support (safetensors loader, tokenizer, lazy loading)

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError, PyRuntimeError};
use pygpukit_core::llm::{
    SafeTensorsFile, Dtype, SafeTensorsError, Tokenizer, TokenizerError,
    LazyModelLoader, LazyTensorError, TensorState,
};
use pygpukit_core::memory::PoolStats;
use std::sync::Arc;
use parking_lot::RwLock;

/// Convert SafeTensorsError to PyErr
fn to_py_err(e: SafeTensorsError) -> PyErr {
    match e {
        SafeTensorsError::IoError(e) => PyIOError::new_err(e.to_string()),
        SafeTensorsError::ParseError(e) => PyValueError::new_err(e),
        SafeTensorsError::TensorNotFound(name) => PyKeyError::new_err(name),
        SafeTensorsError::UnsupportedDtype(dtype) => PyValueError::new_err(dtype),
    }
}

/// Python wrapper for Dtype enum
#[pyclass(name = "Dtype", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PyDtype {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Float64 = 3,
    Float8E4M3 = 4,  // FP8 E4M3
    Float8E5M2 = 5,  // FP8 E5M2
    Int32 = 6,
    Int64 = 7,
    Int16 = 8,
    Int8 = 9,
    UInt8 = 10,
    Bool = 11,
}

impl From<Dtype> for PyDtype {
    fn from(dtype: Dtype) -> Self {
        match dtype {
            Dtype::Float32 => PyDtype::Float32,
            Dtype::Float16 => PyDtype::Float16,
            Dtype::BFloat16 => PyDtype::BFloat16,
            Dtype::Float64 => PyDtype::Float64,
            Dtype::Float8E4M3 => PyDtype::Float8E4M3,
            Dtype::Float8E5M2 => PyDtype::Float8E5M2,
            Dtype::Int32 => PyDtype::Int32,
            Dtype::Int64 => PyDtype::Int64,
            Dtype::Int16 => PyDtype::Int16,
            Dtype::Int8 => PyDtype::Int8,
            Dtype::UInt8 => PyDtype::UInt8,
            Dtype::Bool => PyDtype::Bool,
        }
    }
}

#[pymethods]
impl PyDtype {
    /// Size in bytes of a single element
    #[getter]
    fn element_size(&self) -> usize {
        match self {
            PyDtype::Float64 | PyDtype::Int64 => 8,
            PyDtype::Float32 | PyDtype::Int32 => 4,
            PyDtype::Float16 | PyDtype::BFloat16 | PyDtype::Int16 => 2,
            PyDtype::Int8 | PyDtype::UInt8 | PyDtype::Bool | PyDtype::Float8E4M3 | PyDtype::Float8E5M2 => 1,
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyDtype::Float32 => "Dtype.Float32",
            PyDtype::Float16 => "Dtype.Float16",
            PyDtype::BFloat16 => "Dtype.BFloat16",
            PyDtype::Float64 => "Dtype.Float64",
            PyDtype::Float8E4M3 => "Dtype.Float8E4M3",
            PyDtype::Float8E5M2 => "Dtype.Float8E5M2",
            PyDtype::Int32 => "Dtype.Int32",
            PyDtype::Int64 => "Dtype.Int64",
            PyDtype::Int16 => "Dtype.Int16",
            PyDtype::Int8 => "Dtype.Int8",
            PyDtype::UInt8 => "Dtype.UInt8",
            PyDtype::Bool => "Dtype.Bool",
        }
    }
}

/// Metadata for a single tensor
#[pyclass(name = "TensorInfo")]
#[derive(Clone)]
pub struct PyTensorInfo {
    /// Tensor name
    #[pyo3(get)]
    pub name: String,
    /// Data type
    #[pyo3(get)]
    pub dtype: PyDtype,
    /// Shape dimensions
    #[pyo3(get)]
    pub shape: Vec<usize>,
    /// Byte offset within the data section
    #[pyo3(get)]
    pub offset: usize,
    /// Total size in bytes
    #[pyo3(get)]
    pub size_bytes: usize,
}

#[pymethods]
impl PyTensorInfo {
    /// Total number of elements
    #[getter]
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorInfo(name='{}', dtype={:?}, shape={:?}, size_bytes={})",
            self.name, self.dtype, self.shape, self.size_bytes
        )
    }
}

/// Memory-mapped SafeTensors file
#[pyclass(name = "SafeTensorsFile")]
pub struct PySafeTensorsFile {
    inner: Arc<SafeTensorsFile>,
}

#[pymethods]
impl PySafeTensorsFile {
    /// Open a safetensors file with memory mapping
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let file = SafeTensorsFile::open(path).map_err(to_py_err)?;
        Ok(PySafeTensorsFile {
            inner: Arc::new(file),
        })
    }

    /// Get list of all tensor names
    #[getter]
    fn tensor_names(&self) -> Vec<String> {
        self.inner.tensor_names().iter().map(|s| s.to_string()).collect()
    }

    /// Get tensor info by name
    fn tensor_info(&self, name: &str) -> PyResult<PyTensorInfo> {
        let info = self.inner.tensor_info(name)
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))?;
        Ok(PyTensorInfo {
            name: info.name.clone(),
            dtype: info.dtype.into(),
            shape: info.shape.clone(),
            offset: info.offset,
            size_bytes: info.size_bytes,
        })
    }

    /// Get tensor data as bytes
    fn tensor_bytes(&self, name: &str) -> PyResult<Vec<u8>> {
        let tensor = self.inner.tensor(name).map_err(to_py_err)?;
        Ok(tensor.data.to_vec())
    }

    /// Get tensor data pointer (for zero-copy GPU transfer)
    /// Returns (ptr, size_bytes) where ptr is the raw mmap address
    fn tensor_data_ptr(&self, name: &str) -> PyResult<(usize, usize)> {
        let tensor = self.inner.tensor(name).map_err(to_py_err)?;
        let ptr = tensor.data.as_ptr() as usize;
        let size = tensor.data.len();
        Ok((ptr, size))
    }

    /// Get tensor as numpy array (only for Float32)
    fn tensor_as_f32(&self, py: Python<'_>, name: &str) -> PyResult<Py<numpy::PyArray1<f32>>> {
        let tensor = self.inner.tensor(name).map_err(to_py_err)?;
        let data = tensor.as_f32()
            .ok_or_else(|| PyValueError::new_err("Tensor is not Float32"))?;
        Ok(numpy::PyArray1::from_slice(py, data).into())
    }

    /// Total file size in bytes
    #[getter]
    fn file_size(&self) -> usize {
        self.inner.file_size()
    }

    /// Number of tensors in the file
    #[getter]
    fn num_tensors(&self) -> usize {
        self.inner.num_tensors()
    }

    fn __repr__(&self) -> String {
        format!(
            "SafeTensorsFile(num_tensors={}, file_size={})",
            self.inner.num_tensors(),
            self.inner.file_size()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.num_tensors()
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.tensor_info(name).is_some()
    }
}

/// Load a safetensors file
#[pyfunction]
fn load_safetensors(path: &str) -> PyResult<PySafeTensorsFile> {
    PySafeTensorsFile::new(path)
}

// ============================================================================
// Tokenizer
// ============================================================================

/// Convert TokenizerError to PyErr
fn tokenizer_err_to_py(e: TokenizerError) -> PyErr {
    match e {
        TokenizerError::IoError(e) => PyIOError::new_err(e.to_string()),
        TokenizerError::ParseError(e) => PyValueError::new_err(e),
        TokenizerError::InvalidToken(t) => PyValueError::new_err(t),
    }
}

/// BPE Tokenizer for GPT-2 style models
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load tokenizer from tokenizer.json file
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(tokenizer_err_to_py)?;
        Ok(PyTokenizer { inner: tokenizer })
    }

    /// Load tokenizer from JSON string
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let tokenizer = Tokenizer::from_json_str(json).map_err(tokenizer_err_to_py)?;
        Ok(PyTokenizer { inner: tokenizer })
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Get BOS token ID if available
    #[getter]
    fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }

    /// Get EOS token ID if available
    #[getter]
    fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id()
    }

    /// Get PAD token ID if available
    #[getter]
    fn pad_token_id(&self) -> Option<u32> {
        self.inner.pad_token_id()
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Decode token IDs to text
    fn decode(&self, token_ids: Vec<u32>) -> String {
        self.inner.decode(&token_ids)
    }

    /// Get token string for an ID
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }

    /// Get ID for a token string
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }

    fn __len__(&self) -> usize {
        self.inner.vocab_size()
    }
}

// ============================================================================
// Lazy Model Loader
// ============================================================================

/// Convert LazyTensorError to PyErr
fn lazy_err_to_py(e: LazyTensorError) -> PyErr {
    match e {
        LazyTensorError::TensorNotFound(name) => PyKeyError::new_err(name),
        LazyTensorError::MemoryError(e) => PyRuntimeError::new_err(e.to_string()),
        LazyTensorError::SafeTensorsError(e) => to_py_err(e),
        LazyTensorError::GpuError(e) => PyRuntimeError::new_err(e),
    }
}

/// Tensor state enum (OnDisk, Loading, OnGpu, Evicted)
#[pyclass(name = "TensorState", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PyTensorState {
    /// On disk only (mmap, not loaded to GPU)
    OnDisk = 0,
    /// Currently loading to GPU
    Loading = 1,
    /// Resident on GPU
    OnGpu = 2,
    /// Evicted from GPU (mmap still valid)
    Evicted = 3,
}

impl From<TensorState> for PyTensorState {
    fn from(state: TensorState) -> Self {
        match state {
            TensorState::OnDisk => PyTensorState::OnDisk,
            TensorState::Loading => PyTensorState::Loading,
            TensorState::OnGpu => PyTensorState::OnGpu,
            TensorState::Evicted => PyTensorState::Evicted,
        }
    }
}

#[pymethods]
impl PyTensorState {
    fn __repr__(&self) -> &'static str {
        match self {
            PyTensorState::OnDisk => "TensorState.OnDisk",
            PyTensorState::Loading => "TensorState.Loading",
            PyTensorState::OnGpu => "TensorState.OnGpu",
            PyTensorState::Evicted => "TensorState.Evicted",
        }
    }
}

/// Memory pool statistics
#[pyclass(name = "PoolStats")]
#[derive(Clone)]
pub struct PyPoolStats {
    /// Maximum memory allowed (quota)
    #[pyo3(get)]
    pub quota: usize,
    /// Currently used memory (active allocations)
    #[pyo3(get)]
    pub used: usize,
    /// Memory in free lists (cached for reuse)
    #[pyo3(get)]
    pub cached: usize,
    /// Available memory (quota - used)
    #[pyo3(get)]
    pub available: usize,
    /// Total number of allocations
    #[pyo3(get)]
    pub allocation_count: u64,
    /// Number of blocks reused from free list
    #[pyo3(get)]
    pub reuse_count: u64,
    /// Number of blocks evicted
    #[pyo3(get)]
    pub eviction_count: u64,
    /// Number of new CUDA allocations
    #[pyo3(get)]
    pub cudamalloc_count: u64,
    /// Number of active blocks
    #[pyo3(get)]
    pub active_blocks: usize,
    /// Number of blocks in free lists
    #[pyo3(get)]
    pub free_blocks: usize,
}

impl From<PoolStats> for PyPoolStats {
    fn from(stats: PoolStats) -> Self {
        PyPoolStats {
            quota: stats.quota,
            used: stats.used,
            cached: stats.cached,
            available: stats.available,
            allocation_count: stats.allocation_count,
            reuse_count: stats.reuse_count,
            eviction_count: stats.eviction_count,
            cudamalloc_count: stats.cudamalloc_count,
            active_blocks: stats.active_blocks,
            free_blocks: stats.free_blocks,
        }
    }
}

#[pymethods]
impl PyPoolStats {
    /// Utilization percentage (used / quota * 100)
    #[getter]
    fn utilization(&self) -> f64 {
        if self.quota == 0 {
            0.0
        } else {
            (self.used as f64 / self.quota as f64) * 100.0
        }
    }

    /// Total blocks (active + free)
    #[getter]
    fn total_blocks(&self) -> usize {
        self.active_blocks + self.free_blocks
    }

    fn __repr__(&self) -> String {
        format!(
            "PoolStats(quota={}, used={}, cached={}, available={}, active_blocks={}, free_blocks={})",
            self.quota, self.used, self.cached, self.available, self.active_blocks, self.free_blocks
        )
    }
}

/// Lazy model loader for large models
///
/// Memory-maps SafeTensors files and loads tensors to GPU on demand.
/// When VRAM budget is exceeded, least-recently-used tensors are evicted.
///
/// Example:
///     loader = LazyModelLoader(memory_budget=8*1024**3)  # 8GB
///     loader.load_file("model-00001-of-00004.safetensors")
///     loader.load_file("model-00002-of-00004.safetensors")
///     # Tensors loaded on first access via get_tensor_ptr()
#[pyclass(name = "LazyModelLoader")]
pub struct PyLazyModelLoader {
    inner: RwLock<LazyModelLoader>,
}

#[pymethods]
impl PyLazyModelLoader {
    /// Create a new lazy model loader
    ///
    /// Args:
    ///     memory_budget: Maximum GPU memory to use in bytes
    ///     enable_eviction: Whether to auto-evict when budget exceeded
    #[new]
    #[pyo3(signature = (memory_budget, enable_eviction=true))]
    fn new(memory_budget: usize, enable_eviction: bool) -> Self {
        PyLazyModelLoader {
            inner: RwLock::new(LazyModelLoader::new(memory_budget, enable_eviction)),
        }
    }

    /// Load a SafeTensors file (mmap only, no GPU transfer yet)
    ///
    /// Args:
    ///     path: Path to the SafeTensors file
    fn load_file(&self, path: &str) -> PyResult<()> {
        let path = std::path::Path::new(path);
        self.inner.write().load_file(path).map_err(lazy_err_to_py)
    }

    /// Get tensor info by name
    ///
    /// Args:
    ///     name: Tensor name
    ///
    /// Returns:
    ///     TensorInfo or None if not found
    fn get(&self, name: &str) -> Option<PyTensorInfo> {
        self.inner.read().get(name).map(|info| PyTensorInfo {
            name: info.name.clone(),
            dtype: info.dtype.into(),
            shape: info.shape.clone(),
            offset: info.offset,
            size_bytes: info.size_bytes,
        })
    }

    /// Get all tensor names
    #[getter]
    fn tensor_names(&self) -> Vec<String> {
        self.inner.read().tensor_names()
    }

    /// Get total model size in bytes (all files)
    #[getter]
    fn total_size(&self) -> usize {
        self.inner.read().total_size()
    }

    /// Get currently loaded size on GPU
    #[getter]
    fn loaded_size(&self) -> usize {
        self.inner.read().loaded_size()
    }

    /// Get memory pool statistics
    #[getter]
    fn pool_stats(&self) -> PyPoolStats {
        self.inner.read().pool_stats().into()
    }

    /// Number of tensors in all files
    #[getter]
    fn num_tensors(&self) -> usize {
        self.inner.read().num_tensors()
    }

    /// Number of files loaded
    #[getter]
    fn num_files(&self) -> usize {
        self.inner.read().num_files()
    }

    /// Get list of tensor names currently on GPU
    fn loaded_tensors(&self) -> Vec<String> {
        self.inner.read().loaded_tensors()
    }

    /// Get number of tensors currently on GPU
    fn num_loaded(&self) -> usize {
        self.inner.read().num_loaded()
    }

    /// Unload entire model from GPU
    ///
    /// Releases all GPU memory but keeps mmap references.
    /// Tensors can be reloaded by accessing them again.
    ///
    /// Returns:
    ///     Number of bytes freed
    fn unload_model(&self) -> PyResult<usize> {
        // Use no-op free function - actual GPU memory is managed by native layer
        let free_fn = |_ptr: u64| -> Result<(), String> { Ok(()) };
        self.inner.read().unload_model(free_fn).map_err(lazy_err_to_py)
    }

    /// Unload tensors matching a prefix
    ///
    /// Useful for unloading specific transformer layers.
    /// E.g., prefix "model.layers.0." unloads all tensors in layer 0.
    ///
    /// Args:
    ///     prefix: Tensor name prefix to match
    ///
    /// Returns:
    ///     Tuple of (num_tensors_unloaded, bytes_freed)
    fn unload_layer(&self, prefix: &str) -> PyResult<(usize, usize)> {
        let free_fn = |_ptr: u64| -> Result<(), String> { Ok(()) };
        self.inner.read().unload_layer(prefix, free_fn).map_err(lazy_err_to_py)
    }

    /// Unload specific tensors by name
    ///
    /// Args:
    ///     names: List of tensor names to unload
    ///
    /// Returns:
    ///     Tuple of (num_tensors_unloaded, bytes_freed)
    fn unload_tensors(&self, names: Vec<String>) -> PyResult<(usize, usize)> {
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let free_fn = |_ptr: u64| -> Result<(), String> { Ok(()) };
        self.inner.read().unload_tensors(&name_refs, free_fn).map_err(lazy_err_to_py)
    }

    /// Clear all data (unload tensors + close mmaps)
    ///
    /// After this, the loader cannot be used until new files are loaded.
    ///
    /// Returns:
    ///     Number of bytes freed from GPU
    fn clear(&self) -> PyResult<usize> {
        let free_fn = |_ptr: u64| -> Result<(), String> { Ok(()) };
        self.inner.write().clear(free_fn).map_err(lazy_err_to_py)
    }

    /// Get tensor names matching a prefix (e.g., "model.layers.0.")
    ///
    /// Args:
    ///     prefix: Tensor name prefix to match
    ///
    /// Returns:
    ///     List of tensor names matching the prefix
    fn get_layer_tensors(&self, prefix: &str) -> Vec<String> {
        self.inner.read().get_layer_tensors(prefix)
    }

    /// Get total size of tensors matching a prefix
    ///
    /// Args:
    ///     prefix: Tensor name prefix to match
    ///
    /// Returns:
    ///     Total size in bytes
    fn layer_size(&self, prefix: &str) -> usize {
        self.inner.read().layer_size(prefix)
    }

    /// Check if a layer is fully loaded on GPU
    ///
    /// Args:
    ///     prefix: Tensor name prefix to match
    ///
    /// Returns:
    ///     True if all tensors in the layer are on GPU
    fn is_layer_loaded(&self, prefix: &str) -> bool {
        self.inner.read().is_layer_loaded(prefix)
    }

    /// Get layer loading state
    ///
    /// Args:
    ///     prefix: Tensor name prefix to match
    ///
    /// Returns:
    ///     Tuple of (total_tensors, loaded_tensors, total_bytes, loaded_bytes)
    fn layer_state(&self, prefix: &str) -> (usize, usize, usize, usize) {
        self.inner.read().layer_state(prefix)
    }

    /// Get raw mmap pointer for a tensor (for zero-copy GPU transfer)
    ///
    /// Args:
    ///     name: Tensor name
    ///
    /// Returns:
    ///     Tuple of (ptr, size_bytes) where ptr is the raw mmap address
    fn tensor_data_ptr(&self, name: &str) -> PyResult<(usize, usize)> {
        let loader = self.inner.read();
        let info = loader.get(name)
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))?;

        // Get raw pointer from first file that contains this tensor
        // (In practice, each tensor is in exactly one file)
        drop(loader);

        // We need to access the underlying SafeTensorsFile to get the pointer
        // For now, return the info we have
        Ok((0, info.size_bytes))  // TODO: Implement proper pointer access
    }

    fn __repr__(&self) -> String {
        let loader = self.inner.read();
        format!(
            "LazyModelLoader(files={}, tensors={}, loaded={}/{})",
            loader.num_files(),
            loader.num_tensors(),
            loader.num_loaded(),
            loader.num_tensors()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.read().num_tensors()
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.read().get(name).is_some()
    }
}

/// Register the llm module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDtype>()?;
    m.add_class::<PyTensorInfo>()?;
    m.add_class::<PySafeTensorsFile>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyTensorState>()?;
    m.add_class::<PyPoolStats>()?;
    m.add_class::<PyLazyModelLoader>()?;
    m.add_function(wrap_pyfunction!(load_safetensors, m)?)?;
    Ok(())
}
