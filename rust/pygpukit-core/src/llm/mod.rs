//! LLM support module for PyGPUkit
//!
//! Provides:
//! - safetensors file loading
//! - Tensor metadata and data access
//! - GPU tensor allocation helpers
//! - BPE tokenizer for GPT-2 style models

pub mod tensor_loader;
pub mod tokenizer;

pub use tensor_loader::{
    SafeTensorsFile, TensorInfo, TensorData, SafeTensorsError,
    Dtype, load_safetensors,
};
pub use tokenizer::{Tokenizer, TokenizerError};
