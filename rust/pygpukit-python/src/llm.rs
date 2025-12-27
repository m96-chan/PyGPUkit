//! Python bindings for LLM support (safetensors loader, tokenizer)

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError};
use pygpukit_core::llm::{SafeTensorsFile, Dtype, SafeTensorsError, Tokenizer, TokenizerError};
use std::sync::Arc;

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

/// Register the llm module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDtype>()?;
    m.add_class::<PyTensorInfo>()?;
    m.add_class::<PySafeTensorsFile>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(load_safetensors, m)?)?;
    Ok(())
}
