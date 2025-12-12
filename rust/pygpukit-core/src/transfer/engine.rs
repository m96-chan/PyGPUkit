//! Async Transfer Engine implementation
//!
//! Manages asynchronous memory transfers with multiple streams.

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use parking_lot::RwLock;

use crate::transfer::operation::{TransferOp, TransferState, TransferType};

/// Stream type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamType {
    /// Default compute stream (stream 0)
    Compute,
    /// Dedicated memcpy stream for H2D/D2H
    MemcpyH2D,
    /// Dedicated memcpy stream for D2H (separate to allow overlap)
    MemcpyD2H,
    /// Custom stream with ID
    Custom(u32),
}

impl StreamType {
    /// Convert to stream ID for the C++ backend
    pub fn to_id(&self) -> u32 {
        match self {
            StreamType::Compute => 0,
            StreamType::MemcpyH2D => 1,
            StreamType::MemcpyD2H => 2,
            StreamType::Custom(id) => *id,
        }
    }
}

/// Callback type for executing transfers via C++ backend
pub type TransferCallback = Box<dyn Fn(&TransferOp) -> Result<(), String> + Send + Sync>;

/// Transfer statistics
#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    /// Total transfers queued
    pub total_queued: usize,
    /// Transfers completed successfully
    pub completed_count: usize,
    /// Transfers failed
    pub failed_count: usize,
    /// Total bytes transferred
    pub total_bytes: usize,
    /// Total H2D bytes
    pub h2d_bytes: usize,
    /// Total D2H bytes
    pub d2h_bytes: usize,
    /// Total D2D bytes
    pub d2d_bytes: usize,
    /// Average H2D bandwidth (GB/s)
    pub avg_h2d_bandwidth: f64,
    /// Average D2H bandwidth (GB/s)
    pub avg_d2h_bandwidth: f64,
    /// Pending transfers count
    pub pending_count: usize,
    /// In-progress transfers count
    pub in_progress_count: usize,
}

/// Priority wrapper for transfer operations
struct PriorityTransfer {
    op_id: u64,
    priority: i32,
    queued_order: u64, // For FIFO within same priority
}

impl PartialEq for PriorityTransfer {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.queued_order == other.queued_order
    }
}

impl Eq for PriorityTransfer {}

impl PartialOrd for PriorityTransfer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTransfer {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier queued order (lower = earlier)
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.queued_order.cmp(&self.queued_order), // Reverse for FIFO
            other => other,
        }
    }
}

/// Internal engine state
struct EngineInner {
    /// All transfer operations by ID
    operations: HashMap<u64, TransferOp>,
    /// Priority queue for pending H2D transfers
    h2d_queue: BinaryHeap<PriorityTransfer>,
    /// Priority queue for pending D2H transfers
    d2h_queue: BinaryHeap<PriorityTransfer>,
    /// Priority queue for pending D2D transfers
    d2d_queue: BinaryHeap<PriorityTransfer>,
    /// Currently in-progress transfers by stream
    in_progress: HashMap<u32, Vec<u64>>,
    /// Statistics tracking
    h2d_total_time: f64,
    h2d_total_bytes: usize,
    d2h_total_time: f64,
    d2h_total_bytes: usize,
    completed_count: usize,
    failed_count: usize,
}

/// Async Memory Transfer Engine
///
/// Manages asynchronous memory transfers between host and device with:
/// - Separate streams for H2D and D2H to enable overlap
/// - Priority-based scheduling within each queue
/// - Integration with the scheduler tick loop
///
/// # Example
///
/// ```ignore
/// use pygpukit_core::transfer::{AsyncTransferEngine, TransferOp};
///
/// let engine = AsyncTransferEngine::new(4); // Max 4 concurrent transfers per stream
///
/// // Queue an H2D transfer
/// let op_id = engine.enqueue_h2d(host_ptr, device_ptr, size);
///
/// // In the scheduler tick loop
/// let ready = engine.get_ready_transfers(10);
/// for op in ready {
///     // Execute via C++ backend
///     // ...
///     engine.complete_transfer(op.id);
/// }
/// ```
pub struct AsyncTransferEngine {
    /// Next operation ID
    next_id: AtomicU64,
    /// Queued order counter for FIFO within priority
    queued_order: AtomicU64,
    /// Maximum concurrent transfers per stream
    max_concurrent: usize,
    /// Internal state
    inner: RwLock<EngineInner>,
}

impl AsyncTransferEngine {
    /// Create a new transfer engine
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Maximum concurrent transfers per stream
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            queued_order: AtomicU64::new(0),
            max_concurrent,
            inner: RwLock::new(EngineInner {
                operations: HashMap::new(),
                h2d_queue: BinaryHeap::new(),
                d2h_queue: BinaryHeap::new(),
                d2d_queue: BinaryHeap::new(),
                in_progress: HashMap::new(),
                h2d_total_time: 0.0,
                h2d_total_bytes: 0,
                d2h_total_time: 0.0,
                d2h_total_bytes: 0,
                completed_count: 0,
                failed_count: 0,
            }),
        }
    }

    /// Generate next operation ID
    fn next_op_id(&self) -> u64 {
        self.next_id.fetch_add(1, AtomicOrdering::SeqCst)
    }

    /// Generate next queued order
    fn next_queued_order(&self) -> u64 {
        self.queued_order.fetch_add(1, AtomicOrdering::SeqCst)
    }

    /// Enqueue an H2D transfer
    ///
    /// Returns the operation ID
    pub fn enqueue_h2d(&self, host_ptr: u64, device_ptr: u64, size: usize) -> u64 {
        let id = self.next_op_id();
        let mut op = TransferOp::h2d(id, host_ptr, device_ptr, size);
        op.stream_id = StreamType::MemcpyH2D.to_id();

        self.enqueue_op(op)
    }

    /// Enqueue a D2H transfer
    pub fn enqueue_d2h(&self, device_ptr: u64, host_ptr: u64, size: usize) -> u64 {
        let id = self.next_op_id();
        let mut op = TransferOp::d2h(id, device_ptr, host_ptr, size);
        op.stream_id = StreamType::MemcpyD2H.to_id();

        self.enqueue_op(op)
    }

    /// Enqueue a D2D transfer
    pub fn enqueue_d2d(&self, src_ptr: u64, dst_ptr: u64, size: usize) -> u64 {
        let id = self.next_op_id();
        let mut op = TransferOp::d2d(id, src_ptr, dst_ptr, size);
        op.stream_id = StreamType::Compute.to_id(); // D2D on compute stream

        self.enqueue_op(op)
    }

    /// Enqueue a transfer with priority
    pub fn enqueue_with_priority(
        &self,
        transfer_type: TransferType,
        src_ptr: u64,
        dst_ptr: u64,
        size: usize,
        priority: i32,
    ) -> u64 {
        let id = self.next_op_id();
        let mut op = TransferOp::new(id, transfer_type, src_ptr, dst_ptr, size)
            .with_priority(priority);

        op.stream_id = match transfer_type {
            TransferType::H2D => StreamType::MemcpyH2D.to_id(),
            TransferType::D2H => StreamType::MemcpyD2H.to_id(),
            TransferType::D2D => StreamType::Compute.to_id(),
        };

        self.enqueue_op(op)
    }

    /// Internal: enqueue an operation
    fn enqueue_op(&self, op: TransferOp) -> u64 {
        let id = op.id;
        let priority = op.priority;
        let transfer_type = op.transfer_type;
        let queued_order = self.next_queued_order();

        let mut inner = self.inner.write();
        inner.operations.insert(id, op);

        let priority_entry = PriorityTransfer {
            op_id: id,
            priority,
            queued_order,
        };

        match transfer_type {
            TransferType::H2D => inner.h2d_queue.push(priority_entry),
            TransferType::D2H => inner.d2h_queue.push(priority_entry),
            TransferType::D2D => inner.d2d_queue.push(priority_entry),
        }

        id
    }

    /// Get transfers that are ready to execute
    ///
    /// Returns up to `max_transfers` operations that can be started
    pub fn get_ready_transfers(&self, max_transfers: usize) -> Vec<TransferOp> {
        let mut inner = self.inner.write();
        let mut ready = Vec::new();

        // Process each queue type
        let queues_and_streams = [
            (TransferType::H2D, StreamType::MemcpyH2D.to_id()),
            (TransferType::D2H, StreamType::MemcpyD2H.to_id()),
            (TransferType::D2D, StreamType::Compute.to_id()),
        ];

        for (transfer_type, stream_id) in queues_and_streams {
            if ready.len() >= max_transfers {
                break;
            }

            // Get current in-progress count for this stream
            let current_count = inner.in_progress.get(&stream_id).map(|v| v.len()).unwrap_or(0);
            let available_slots = self.max_concurrent.saturating_sub(current_count);

            if available_slots == 0 {
                continue;
            }

            let to_get = available_slots.min(max_transfers - ready.len());

            // Get the queue for this transfer type
            let queue = match transfer_type {
                TransferType::H2D => &mut inner.h2d_queue,
                TransferType::D2H => &mut inner.d2h_queue,
                TransferType::D2D => &mut inner.d2d_queue,
            };

            // Collect op IDs to process
            let mut ops_to_start = Vec::new();
            while ops_to_start.len() < to_get {
                if let Some(entry) = queue.pop() {
                    ops_to_start.push(entry.op_id);
                } else {
                    break;
                }
            }

            // Start each operation
            for op_id in ops_to_start {
                if let Some(op) = inner.operations.get_mut(&op_id) {
                    if op.state == TransferState::Queued {
                        op.start();
                        ready.push(op.clone());
                    }
                }
                inner.in_progress.entry(stream_id).or_insert_with(Vec::new).push(op_id);
            }
        }

        ready
    }

    /// Mark a transfer as started (called when C++ backend initiates transfer)
    pub fn start_transfer(&self, op_id: u64) -> bool {
        let mut inner = self.inner.write();

        // First get the stream_id if operation is in queued state
        let stream_id = inner.operations.get(&op_id).and_then(|op| {
            if op.state == TransferState::Queued {
                Some(op.stream_id)
            } else {
                None
            }
        });

        if let Some(sid) = stream_id {
            if let Some(op) = inner.operations.get_mut(&op_id) {
                op.start();
            }
            inner.in_progress
                .entry(sid)
                .or_insert_with(Vec::new)
                .push(op_id);
            return true;
        }
        false
    }

    /// Mark a transfer as completed
    pub fn complete_transfer(&self, op_id: u64) -> bool {
        let mut inner = self.inner.write();

        // Get operation info first
        let op_info = inner.operations.get(&op_id).and_then(|op| {
            if op.state == TransferState::InProgress {
                Some((op.stream_id, op.transfer_type, op.size))
            } else {
                None
            }
        });

        if let Some((stream_id, transfer_type, size)) = op_info {
            if let Some(op) = inner.operations.get_mut(&op_id) {
                op.complete();

                // Update stats
                if let Some(duration) = op.duration() {
                    match transfer_type {
                        TransferType::H2D => {
                            inner.h2d_total_time += duration;
                            inner.h2d_total_bytes += size;
                        }
                        TransferType::D2H => {
                            inner.d2h_total_time += duration;
                            inner.d2h_total_bytes += size;
                        }
                        TransferType::D2D => {}
                    }
                }
            }

            // Remove from in-progress
            if let Some(in_prog) = inner.in_progress.get_mut(&stream_id) {
                in_prog.retain(|&id| id != op_id);
            }

            inner.completed_count += 1;
            return true;
        }
        false
    }

    /// Mark a transfer as failed
    pub fn fail_transfer(&self, op_id: u64, error: String) -> bool {
        let mut inner = self.inner.write();

        let stream_id = inner.operations.get(&op_id).map(|op| op.stream_id);

        if let Some(op) = inner.operations.get_mut(&op_id) {
            if op.state == TransferState::InProgress {
                op.fail(error);

                // Remove from in-progress
                if let Some(sid) = stream_id {
                    if let Some(in_prog) = inner.in_progress.get_mut(&sid) {
                        in_prog.retain(|&id| id != op_id);
                    }
                }

                inner.failed_count += 1;
                return true;
            }
        }
        false
    }

    /// Cancel a pending transfer
    pub fn cancel_transfer(&self, op_id: u64) -> bool {
        let mut inner = self.inner.write();

        if let Some(op) = inner.operations.get_mut(&op_id) {
            if op.state == TransferState::Queued {
                op.cancel();
                return true;
            }
        }
        false
    }

    /// Get operation by ID
    pub fn get_operation(&self, op_id: u64) -> Option<TransferOp> {
        self.inner.read().operations.get(&op_id).cloned()
    }

    /// Get transfer statistics
    pub fn stats(&self) -> TransferStats {
        let inner = self.inner.read();

        let pending_count = inner.h2d_queue.len() + inner.d2h_queue.len() + inner.d2d_queue.len();
        let in_progress_count: usize = inner.in_progress.values().map(|v| v.len()).sum();

        let avg_h2d_bandwidth = if inner.h2d_total_time > 0.0 {
            (inner.h2d_total_bytes as f64) / inner.h2d_total_time / 1e9
        } else {
            0.0
        };

        let avg_d2h_bandwidth = if inner.d2h_total_time > 0.0 {
            (inner.d2h_total_bytes as f64) / inner.d2h_total_time / 1e9
        } else {
            0.0
        };

        TransferStats {
            total_queued: inner.operations.len(),
            completed_count: inner.completed_count,
            failed_count: inner.failed_count,
            total_bytes: inner.h2d_total_bytes + inner.d2h_total_bytes,
            h2d_bytes: inner.h2d_total_bytes,
            d2h_bytes: inner.d2h_total_bytes,
            d2d_bytes: 0, // TODO: track D2D separately
            avg_h2d_bandwidth,
            avg_d2h_bandwidth,
            pending_count,
            in_progress_count,
        }
    }

    /// Synchronize a specific stream (wait for all transfers on that stream)
    ///
    /// Returns the IDs of all completed transfers
    pub fn get_in_progress_for_stream(&self, stream_id: u32) -> Vec<u64> {
        self.inner.read()
            .in_progress
            .get(&stream_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Check if any transfers are pending or in progress
    pub fn has_pending_work(&self) -> bool {
        let inner = self.inner.read();
        !inner.h2d_queue.is_empty()
            || !inner.d2h_queue.is_empty()
            || !inner.d2d_queue.is_empty()
            || inner.in_progress.values().any(|v| !v.is_empty())
    }

    /// Clear all completed and cancelled operations to free memory
    pub fn gc(&self) {
        let mut inner = self.inner.write();
        inner.operations.retain(|_, op| !op.state.is_terminal());
    }

    /// Clear all operations (including pending)
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.operations.clear();
        inner.h2d_queue.clear();
        inner.d2h_queue.clear();
        inner.d2d_queue.clear();
        inner.in_progress.clear();
        inner.h2d_total_time = 0.0;
        inner.h2d_total_bytes = 0;
        inner.d2h_total_time = 0.0;
        inner.d2h_total_bytes = 0;
        inner.completed_count = 0;
        inner.failed_count = 0;
    }
}

// Thread-safe
unsafe impl Send for AsyncTransferEngine {}
unsafe impl Sync for AsyncTransferEngine {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = AsyncTransferEngine::new(4);
        let stats = engine.stats();
        assert_eq!(stats.total_queued, 0);
        assert_eq!(stats.pending_count, 0);
    }

    #[test]
    fn test_enqueue_h2d() {
        let engine = AsyncTransferEngine::new(4);

        let op_id = engine.enqueue_h2d(0x1000, 0x2000, 1024);
        assert!(op_id > 0);

        let op = engine.get_operation(op_id).unwrap();
        assert_eq!(op.transfer_type, TransferType::H2D);
        assert_eq!(op.size, 1024);
        assert_eq!(op.state, TransferState::Queued);
    }

    #[test]
    fn test_get_ready_transfers() {
        let engine = AsyncTransferEngine::new(4);

        engine.enqueue_h2d(0x1000, 0x2000, 1024);
        engine.enqueue_d2h(0x2000, 0x1000, 2048);

        let ready = engine.get_ready_transfers(10);
        assert_eq!(ready.len(), 2);

        // Both should be in progress now
        let stats = engine.stats();
        assert_eq!(stats.in_progress_count, 2);
        assert_eq!(stats.pending_count, 0);
    }

    #[test]
    fn test_complete_transfer() {
        let engine = AsyncTransferEngine::new(4);

        let op_id = engine.enqueue_h2d(0x1000, 0x2000, 1024);
        let _ = engine.get_ready_transfers(1);

        assert!(engine.complete_transfer(op_id));

        let op = engine.get_operation(op_id).unwrap();
        assert_eq!(op.state, TransferState::Completed);

        let stats = engine.stats();
        assert_eq!(stats.completed_count, 1);
        assert_eq!(stats.in_progress_count, 0);
    }

    #[test]
    fn test_max_concurrent() {
        let engine = AsyncTransferEngine::new(2);

        // Queue 5 H2D transfers
        for i in 0..5 {
            engine.enqueue_h2d(0x1000 * (i + 1) as u64, 0x2000 * (i + 1) as u64, 1024);
        }

        // Only 2 should be ready (max_concurrent = 2)
        let ready = engine.get_ready_transfers(10);
        assert_eq!(ready.len(), 2);

        let stats = engine.stats();
        assert_eq!(stats.in_progress_count, 2);
        assert_eq!(stats.pending_count, 3);
    }

    #[test]
    fn test_priority_ordering() {
        let engine = AsyncTransferEngine::new(1);

        // Queue with different priorities
        let low_id = engine.enqueue_with_priority(TransferType::H2D, 0x1000, 0x2000, 1024, 0);
        let high_id = engine.enqueue_with_priority(TransferType::H2D, 0x3000, 0x4000, 1024, 10);

        // Higher priority should come first
        let ready = engine.get_ready_transfers(1);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, high_id);

        // Complete it and get next
        engine.complete_transfer(high_id);
        let ready = engine.get_ready_transfers(1);
        assert_eq!(ready[0].id, low_id);
    }

    #[test]
    fn test_separate_streams() {
        let engine = AsyncTransferEngine::new(2);

        // Queue H2D and D2H - they use different streams
        engine.enqueue_h2d(0x1000, 0x2000, 1024);
        engine.enqueue_h2d(0x3000, 0x4000, 1024);
        engine.enqueue_d2h(0x5000, 0x6000, 1024);
        engine.enqueue_d2h(0x7000, 0x8000, 1024);

        // Should get 4 transfers (2 per stream, max_concurrent per stream)
        let ready = engine.get_ready_transfers(10);
        assert_eq!(ready.len(), 4);
    }

    #[test]
    fn test_fail_transfer() {
        let engine = AsyncTransferEngine::new(4);

        let op_id = engine.enqueue_h2d(0x1000, 0x2000, 1024);
        let _ = engine.get_ready_transfers(1);

        assert!(engine.fail_transfer(op_id, "CUDA error".into()));

        let op = engine.get_operation(op_id).unwrap();
        assert_eq!(op.state, TransferState::Failed);
        assert_eq!(op.error, Some("CUDA error".into()));

        let stats = engine.stats();
        assert_eq!(stats.failed_count, 1);
    }
}
