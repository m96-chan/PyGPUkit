//! SafeTensors file loader for PyGPUkit
//!
//! Provides memory-mapped loading of safetensors files for efficient
//! GPU tensor allocation.

use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Error type for SafeTensors operations
#[derive(Debug)]
pub enum SafeTensorsError {
    IoError(std::io::Error),
    ParseError(String),
    TensorNotFound(String),
    UnsupportedDtype(String),
}

impl std::fmt::Display for SafeTensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafeTensorsError::IoError(e) => write!(f, "IO error: {}", e),
            SafeTensorsError::ParseError(e) => write!(f, "Parse error: {}", e),
            SafeTensorsError::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            SafeTensorsError::UnsupportedDtype(dtype) => write!(f, "Unsupported dtype: {}", dtype),
        }
    }
}

impl std::error::Error for SafeTensorsError {}

impl From<std::io::Error> for SafeTensorsError {
    fn from(e: std::io::Error) -> Self {
        SafeTensorsError::IoError(e)
    }
}

impl From<safetensors::SafeTensorError> for SafeTensorsError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        SafeTensorsError::ParseError(e.to_string())
    }
}

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Float32,
    Float16,
    BFloat16,
    Float64,
    Float8E4M3,  // FP8 E4M3 (1 sign, 4 exponent, 3 mantissa)
    Float8E5M2,  // FP8 E5M2 (1 sign, 5 exponent, 2 mantissa)
    Int32,
    Int64,
    Int16,
    Int8,
    UInt8,
    Bool,
}

impl Dtype {
    /// Size in bytes of a single element
    pub fn element_size(&self) -> usize {
        match self {
            Dtype::Float64 | Dtype::Int64 => 8,
            Dtype::Float32 | Dtype::Int32 => 4,
            Dtype::Float16 | Dtype::BFloat16 | Dtype::Int16 => 2,
            Dtype::Int8 | Dtype::UInt8 | Dtype::Bool | Dtype::Float8E4M3 | Dtype::Float8E5M2 => 1,
        }
    }

    /// Convert from safetensors dtype string
    pub fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self, SafeTensorsError> {
        match dtype {
            safetensors::Dtype::F32 => Ok(Dtype::Float32),
            safetensors::Dtype::F16 => Ok(Dtype::Float16),
            safetensors::Dtype::BF16 => Ok(Dtype::BFloat16),
            safetensors::Dtype::F64 => Ok(Dtype::Float64),
            safetensors::Dtype::F8_E4M3 => Ok(Dtype::Float8E4M3),
            safetensors::Dtype::F8_E5M2 => Ok(Dtype::Float8E5M2),
            safetensors::Dtype::I32 => Ok(Dtype::Int32),
            safetensors::Dtype::I64 => Ok(Dtype::Int64),
            safetensors::Dtype::I16 => Ok(Dtype::Int16),
            safetensors::Dtype::I8 => Ok(Dtype::Int8),
            safetensors::Dtype::U8 => Ok(Dtype::UInt8),
            safetensors::Dtype::BOOL => Ok(Dtype::Bool),
            _ => Err(SafeTensorsError::UnsupportedDtype(format!("{:?}", dtype))),
        }
    }
}

/// Metadata for a single tensor
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (key in safetensors file)
    pub name: String,
    /// Data type
    pub dtype: Dtype,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Byte offset within the data section
    pub offset: usize,
    /// Total size in bytes
    pub size_bytes: usize,
}

impl TensorInfo {
    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// View into tensor data (zero-copy reference to mmap)
pub struct TensorData<'a> {
    /// Tensor metadata
    pub info: TensorInfo,
    /// Raw bytes (slice of mmap)
    pub data: &'a [u8],
}

impl<'a> TensorData<'a> {
    /// Get data as f32 slice (only valid if dtype is Float32)
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.info.dtype != Dtype::Float32 {
            return None;
        }
        // Safety: data is aligned and valid for f32
        let ptr = self.data.as_ptr() as *const f32;
        let len = self.data.len() / 4;
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Get data as f16 bytes (raw bytes, 2 per element)
    pub fn as_f16_bytes(&self) -> Option<&[u8]> {
        if self.info.dtype != Dtype::Float16 {
            return None;
        }
        Some(self.data)
    }

    /// Get data as bf16 bytes (raw bytes, 2 per element)
    pub fn as_bf16_bytes(&self) -> Option<&[u8]> {
        if self.info.dtype != Dtype::BFloat16 {
            return None;
        }
        Some(self.data)
    }
}

/// Memory-mapped SafeTensors file
pub struct SafeTensorsFile {
    /// Memory-mapped file data
    _mmap: Mmap,
    /// Parsed header with tensor metadata
    tensor_infos: HashMap<String, TensorInfo>,
    /// Offset to data section start
    data_offset: usize,
    /// Raw pointer to mmap data (for creating tensor views)
    data_ptr: *const u8,
    /// Total file size
    file_size: usize,
}

// Safety: SafeTensorsFile is Send because the mmap is read-only
// and the data_ptr points to immutable memory
unsafe impl Send for SafeTensorsFile {}
unsafe impl Sync for SafeTensorsFile {}

impl SafeTensorsFile {
    /// Open a safetensors file with memory mapping
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SafeTensorsError> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        let file_size = mmap.len();

        // Parse using safetensors crate
        let tensors = SafeTensors::deserialize(&mmap)?;

        // Extract tensor info
        let mut tensor_infos = HashMap::new();
        for (name, view) in tensors.tensors() {
            let dtype = Dtype::from_safetensors(view.dtype())?;
            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            // Calculate offset from mmap start
            let data_ptr = data.as_ptr();
            let mmap_ptr = mmap.as_ptr();
            let offset = data_ptr as usize - mmap_ptr as usize;

            let info = TensorInfo {
                name: name.to_string(),
                dtype,
                shape,
                offset,
                size_bytes: data.len(),
            };
            tensor_infos.insert(name.to_string(), info);
        }

        // Data offset is after the header
        // Header format: 8-byte size + JSON header + data
        let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let data_offset = 8 + header_size;

        let data_ptr = mmap.as_ptr();

        Ok(SafeTensorsFile {
            _mmap: mmap,
            tensor_infos,
            data_offset,
            data_ptr,
            file_size,
        })
    }

    /// Get list of all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_infos.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor info by name
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_infos.get(name)
    }

    /// Get tensor data by name (zero-copy view into mmap)
    pub fn tensor(&self, name: &str) -> Result<TensorData<'_>, SafeTensorsError> {
        let info = self
            .tensor_infos
            .get(name)
            .ok_or_else(|| SafeTensorsError::TensorNotFound(name.to_string()))?;

        // Safety: offset and size are validated during parsing
        let data = unsafe {
            std::slice::from_raw_parts(self.data_ptr.add(info.offset), info.size_bytes)
        };

        Ok(TensorData {
            info: info.clone(),
            data,
        })
    }

    /// Get all tensors as an iterator
    pub fn tensors(&self) -> impl Iterator<Item = Result<TensorData<'_>, SafeTensorsError>> {
        self.tensor_infos
            .keys()
            .map(|name| self.tensor(name))
    }

    /// Total file size in bytes
    pub fn file_size(&self) -> usize {
        self.file_size
    }

    /// Number of tensors in the file
    pub fn num_tensors(&self) -> usize {
        self.tensor_infos.len()
    }

    /// Data section offset (after header)
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }
}

/// Convenience function to load a safetensors file
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<SafeTensorsFile, SafeTensorsError> {
    SafeTensorsFile::open(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_element_size() {
        assert_eq!(Dtype::Float32.element_size(), 4);
        assert_eq!(Dtype::Float16.element_size(), 2);
        assert_eq!(Dtype::BFloat16.element_size(), 2);
        assert_eq!(Dtype::Float64.element_size(), 8);
        assert_eq!(Dtype::Int8.element_size(), 1);
    }

    #[test]
    fn test_tensor_info_numel() {
        let info = TensorInfo {
            name: "test".to_string(),
            dtype: Dtype::Float32,
            shape: vec![2, 3, 4],
            offset: 0,
            size_bytes: 96,
        };
        assert_eq!(info.numel(), 24);
    }
}
