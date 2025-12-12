//! Transfer operation definitions
//!
//! Defines the types of memory transfers and their metadata.

use std::time::{SystemTime, UNIX_EPOCH};

/// Type of memory transfer operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferType {
    /// Host to Device transfer
    H2D,
    /// Device to Host transfer
    D2H,
    /// Device to Device transfer
    D2D,
}

impl TransferType {
    /// Returns a human-readable name for this transfer type
    pub fn name(&self) -> &'static str {
        match self {
            TransferType::H2D => "H2D",
            TransferType::D2H => "D2H",
            TransferType::D2D => "D2D",
        }
    }
}

/// State of a transfer operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferState {
    /// Transfer is queued but not started
    Queued,
    /// Transfer is in progress
    InProgress,
    /// Transfer completed successfully
    Completed,
    /// Transfer failed
    Failed,
    /// Transfer was cancelled
    Cancelled,
}

impl TransferState {
    /// Check if this is a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, TransferState::Completed | TransferState::Failed | TransferState::Cancelled)
    }
}

/// Represents a single memory transfer operation
#[derive(Debug, Clone)]
pub struct TransferOp {
    /// Unique operation ID
    pub id: u64,
    /// Type of transfer
    pub transfer_type: TransferType,
    /// Source memory address (device pointer or host pointer as u64)
    pub src_ptr: u64,
    /// Destination memory address
    pub dst_ptr: u64,
    /// Size of transfer in bytes
    pub size: usize,
    /// Current state
    pub state: TransferState,
    /// Stream ID to use (0 = default compute, 1 = memcpy stream)
    pub stream_id: u32,
    /// Timestamp when operation was queued
    pub queued_at: f64,
    /// Timestamp when operation started
    pub started_at: Option<f64>,
    /// Timestamp when operation completed
    pub completed_at: Option<f64>,
    /// Priority (higher = more urgent)
    pub priority: i32,
    /// Error message if failed
    pub error: Option<String>,
    /// Associated task ID (if linked to a scheduler task)
    pub task_id: Option<String>,
}

impl TransferOp {
    /// Create a new transfer operation
    pub fn new(
        id: u64,
        transfer_type: TransferType,
        src_ptr: u64,
        dst_ptr: u64,
        size: usize,
    ) -> Self {
        Self {
            id,
            transfer_type,
            src_ptr,
            dst_ptr,
            size,
            state: TransferState::Queued,
            stream_id: 1, // Default to memcpy stream
            queued_at: Self::now(),
            started_at: None,
            completed_at: None,
            priority: 0,
            error: None,
            task_id: None,
        }
    }

    /// Create an H2D transfer
    pub fn h2d(id: u64, host_ptr: u64, device_ptr: u64, size: usize) -> Self {
        Self::new(id, TransferType::H2D, host_ptr, device_ptr, size)
    }

    /// Create a D2H transfer
    pub fn d2h(id: u64, device_ptr: u64, host_ptr: u64, size: usize) -> Self {
        Self::new(id, TransferType::D2H, device_ptr, host_ptr, size)
    }

    /// Create a D2D transfer
    pub fn d2d(id: u64, src_device_ptr: u64, dst_device_ptr: u64, size: usize) -> Self {
        Self::new(id, TransferType::D2D, src_device_ptr, dst_device_ptr, size)
    }

    /// Set the priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the stream ID
    pub fn with_stream(mut self, stream_id: u32) -> Self {
        self.stream_id = stream_id;
        self
    }

    /// Link to a scheduler task
    pub fn with_task(mut self, task_id: String) -> Self {
        self.task_id = Some(task_id);
        self
    }

    /// Mark as started
    pub fn start(&mut self) {
        if self.state == TransferState::Queued {
            self.state = TransferState::InProgress;
            self.started_at = Some(Self::now());
        }
    }

    /// Mark as completed
    pub fn complete(&mut self) {
        if self.state == TransferState::InProgress {
            self.state = TransferState::Completed;
            self.completed_at = Some(Self::now());
        }
    }

    /// Mark as failed
    pub fn fail(&mut self, error: String) {
        self.state = TransferState::Failed;
        self.completed_at = Some(Self::now());
        self.error = Some(error);
    }

    /// Mark as cancelled
    pub fn cancel(&mut self) {
        if !self.state.is_terminal() {
            self.state = TransferState::Cancelled;
            self.completed_at = Some(Self::now());
        }
    }

    /// Get wait time (time in queue before starting)
    pub fn wait_time(&self) -> f64 {
        self.started_at
            .map(|s| s - self.queued_at)
            .unwrap_or_else(|| Self::now() - self.queued_at)
    }

    /// Get transfer duration (time from start to completion)
    pub fn duration(&self) -> Option<f64> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }

    /// Get bandwidth in GB/s (if completed)
    pub fn bandwidth_gbps(&self) -> Option<f64> {
        self.duration().map(|d| {
            if d > 0.0 {
                (self.size as f64) / d / 1e9
            } else {
                0.0
            }
        })
    }

    /// Get current Unix timestamp
    #[inline]
    fn now() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_op_creation() {
        let op = TransferOp::h2d(1, 0x1000, 0x2000, 1024);
        assert_eq!(op.id, 1);
        assert_eq!(op.transfer_type, TransferType::H2D);
        assert_eq!(op.size, 1024);
        assert_eq!(op.state, TransferState::Queued);
    }

    #[test]
    fn test_transfer_lifecycle() {
        let mut op = TransferOp::d2h(1, 0x2000, 0x1000, 2048);

        assert_eq!(op.state, TransferState::Queued);
        assert!(op.started_at.is_none());

        op.start();
        assert_eq!(op.state, TransferState::InProgress);
        assert!(op.started_at.is_some());

        op.complete();
        assert_eq!(op.state, TransferState::Completed);
        assert!(op.completed_at.is_some());
        assert!(op.duration().is_some());
    }

    #[test]
    fn test_transfer_failure() {
        let mut op = TransferOp::h2d(1, 0x1000, 0x2000, 1024);
        op.start();
        op.fail("CUDA error: out of memory".into());

        assert_eq!(op.state, TransferState::Failed);
        assert_eq!(op.error, Some("CUDA error: out of memory".into()));
    }

    #[test]
    fn test_priority_and_stream() {
        let op = TransferOp::h2d(1, 0x1000, 0x2000, 1024)
            .with_priority(10)
            .with_stream(2);

        assert_eq!(op.priority, 10);
        assert_eq!(op.stream_id, 2);
    }
}
