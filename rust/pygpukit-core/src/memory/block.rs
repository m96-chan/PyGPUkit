//! Memory block representation
//!
//! A MemoryBlock represents a single allocation in the memory pool.
//! It can exist on GPU, host (CPU), or both.

use std::time::{SystemTime, UNIX_EPOCH};

/// Represents a memory block in the pool.
///
/// Mirrors Python's MemoryBlock dataclass for API compatibility.
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Unique identifier for this block
    pub id: u64,
    /// Size of the block in bytes (rounded to size class)
    pub size: usize,
    /// Device pointer (CUdeviceptr as u64 for FFI)
    pub device_ptr: Option<u64>,
    /// Host-side data (for evicted blocks)
    pub host_data: Option<Vec<u8>>,
    /// Whether block is currently on GPU
    pub on_gpu: bool,
    /// Whether block is currently on host
    pub on_host: bool,
    /// Last access timestamp (Unix time as f64 for Python compat)
    pub last_access: f64,
}

impl MemoryBlock {
    /// Create a new memory block.
    ///
    /// The block starts on GPU with the given device pointer.
    pub fn new(id: u64, size: usize, device_ptr: Option<u64>) -> Self {
        Self {
            id,
            size,
            device_ptr,
            host_data: None,
            on_gpu: true,
            on_host: false,
            last_access: Self::now(),
        }
    }

    /// Update the last access timestamp to current time.
    #[inline]
    pub fn touch(&mut self) {
        self.last_access = Self::now();
    }

    /// Get current Unix timestamp as f64.
    #[inline]
    fn now() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Check if this block is available for use (on GPU).
    #[inline]
    pub fn is_available(&self) -> bool {
        self.on_gpu && self.device_ptr.is_some()
    }

    /// Check if this block has been evicted to host.
    #[inline]
    pub fn is_evicted(&self) -> bool {
        !self.on_gpu && self.on_host
    }
}

impl Default for MemoryBlock {
    fn default() -> Self {
        Self {
            id: 0,
            size: 0,
            device_ptr: None,
            host_data: None,
            on_gpu: false,
            on_host: false,
            last_access: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let block = MemoryBlock::new(1, 1024, Some(0x12345678));
        assert_eq!(block.id, 1);
        assert_eq!(block.size, 1024);
        assert_eq!(block.device_ptr, Some(0x12345678));
        assert!(block.on_gpu);
        assert!(!block.on_host);
        assert!(block.last_access > 0.0);
    }

    #[test]
    fn test_block_touch() {
        let mut block = MemoryBlock::new(1, 1024, None);
        let initial = block.last_access;
        std::thread::sleep(std::time::Duration::from_millis(10));
        block.touch();
        assert!(block.last_access > initial);
    }

    #[test]
    fn test_block_availability() {
        let mut block = MemoryBlock::new(1, 1024, Some(0x1000));
        assert!(block.is_available());
        assert!(!block.is_evicted());

        // Simulate eviction
        block.on_gpu = false;
        block.on_host = true;
        block.device_ptr = None;
        assert!(!block.is_available());
        assert!(block.is_evicted());
    }
}
