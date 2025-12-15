//! Asynchronous Kernel Execution
//!
//! Provides non-blocking kernel dispatch with Future-based result retrieval:
//! - KernelFuture: Handle for tracking async kernel execution
//! - AsyncExecutor: Manages async kernel lifecycle per stream
//!
//! Design:
//! - dispatch_async() returns immediately with a KernelFuture
//! - Kernel executes on dedicated CUDA stream
//! - wait() blocks until kernel completes (stream synchronize)
//! - is_ready() checks completion without blocking

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use parking_lot::{RwLock, Mutex};

/// State of an async kernel execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FutureState {
    /// Kernel is queued but not yet launched
    Pending,
    /// Kernel has been launched, executing on GPU
    Running,
    /// Kernel execution completed successfully
    Completed,
    /// Kernel execution failed
    Failed,
    /// Kernel was cancelled
    Cancelled,
}

impl FutureState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, FutureState::Completed | FutureState::Failed | FutureState::Cancelled)
    }
}

/// Result of an async kernel execution
#[derive(Debug, Clone)]
pub struct KernelResult {
    /// Whether execution succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution time in seconds
    pub exec_time: f64,
    /// Output data (if any)
    pub output: Option<Vec<u8>>,
}

impl KernelResult {
    pub fn success(exec_time: f64) -> Self {
        Self {
            success: true,
            error: None,
            exec_time,
            output: None,
        }
    }

    pub fn failure(error: String) -> Self {
        Self {
            success: false,
            error: Some(error),
            exec_time: 0.0,
            output: None,
        }
    }

    pub fn with_output(mut self, output: Vec<u8>) -> Self {
        self.output = Some(output);
        self
    }
}

/// Internal state for a kernel future
struct FutureInner {
    state: FutureState,
    result: Option<KernelResult>,
    launched_at: Option<f64>,
    completed_at: Option<f64>,
}

/// Handle for tracking async kernel execution
///
/// Created by `AsyncExecutor::dispatch()`. Use `wait()` to block until
/// completion or `is_ready()` to check without blocking.
///
/// # Example
///
/// ```ignore
/// let future = executor.dispatch(request);
///
/// // Do other work while kernel executes...
///
/// if future.is_ready() {
///     let result = future.wait();
/// }
/// ```
pub struct KernelFuture {
    /// Unique ID for this future
    id: u64,
    /// Stream ID where kernel is executing
    stream_id: u32,
    /// Context ID (LLM ID)
    context_id: String,
    /// Shared state
    inner: Arc<RwLock<FutureInner>>,
    /// Flag for quick ready check
    ready: Arc<AtomicBool>,
}

impl KernelFuture {
    /// Create a new pending future
    fn new(id: u64, stream_id: u32, context_id: String) -> Self {
        Self {
            id,
            stream_id,
            context_id,
            inner: Arc::new(RwLock::new(FutureInner {
                state: FutureState::Pending,
                result: None,
                launched_at: None,
                completed_at: None,
            })),
            ready: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get future ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get stream ID
    pub fn stream_id(&self) -> u32 {
        self.stream_id
    }

    /// Get context ID
    pub fn context_id(&self) -> &str {
        &self.context_id
    }

    /// Check if kernel execution is complete (non-blocking)
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Get current state
    pub fn state(&self) -> FutureState {
        self.inner.read().state
    }

    /// Wait for kernel completion (blocking)
    ///
    /// Returns the kernel result. If already complete, returns immediately.
    /// If still running, blocks until completion.
    ///
    /// Note: The actual blocking is done by C++ backend via stream synchronize.
    /// This method just returns the cached result after sync.
    pub fn wait(&self) -> KernelResult {
        // Spin-wait with yield (in practice, C++ backend does the real sync)
        while !self.is_ready() {
            std::thread::yield_now();
        }

        let inner = self.inner.read();
        inner.result.clone().unwrap_or_else(|| KernelResult::failure("No result available".into()))
    }

    /// Try to get result without blocking
    pub fn try_get(&self) -> Option<KernelResult> {
        if self.is_ready() {
            let inner = self.inner.read();
            inner.result.clone()
        } else {
            None
        }
    }

    /// Get execution time (if completed)
    pub fn exec_time(&self) -> Option<f64> {
        let inner = self.inner.read();
        match (inner.launched_at, inner.completed_at) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }

    // --- Internal methods (called by AsyncExecutor) ---

    fn mark_launched(&self) {
        let mut inner = self.inner.write();
        inner.state = FutureState::Running;
        inner.launched_at = Some(Self::now());
    }

    fn mark_completed(&self, result: KernelResult) {
        let mut inner = self.inner.write();
        inner.state = FutureState::Completed;
        inner.completed_at = Some(Self::now());
        inner.result = Some(result);
        drop(inner);
        self.ready.store(true, Ordering::SeqCst);
    }

    fn mark_failed(&self, error: String) {
        let mut inner = self.inner.write();
        inner.state = FutureState::Failed;
        inner.completed_at = Some(Self::now());
        inner.result = Some(KernelResult::failure(error));
        drop(inner);
        self.ready.store(true, Ordering::SeqCst);
    }

    fn mark_cancelled(&self) {
        let mut inner = self.inner.write();
        inner.state = FutureState::Cancelled;
        inner.completed_at = Some(Self::now());
        inner.result = Some(KernelResult::failure("Cancelled".into()));
        drop(inner);
        self.ready.store(true, Ordering::SeqCst);
    }

    fn now() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

// Clone creates a new handle to the same future
impl Clone for KernelFuture {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            stream_id: self.stream_id,
            context_id: self.context_id.clone(),
            inner: Arc::clone(&self.inner),
            ready: Arc::clone(&self.ready),
        }
    }
}

/// Async kernel request
#[derive(Debug, Clone)]
pub struct AsyncKernelRequest {
    /// Kernel function handle (CUfunction as u64)
    pub kernel_handle: u64,
    /// Grid dimensions (x, y, z)
    pub grid: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block: (u32, u32, u32),
    /// Shared memory size
    pub shared_mem: u32,
    /// Kernel arguments as raw pointers
    pub args: Vec<u64>,
    /// Optional callback ID for completion notification
    pub callback_id: Option<u64>,
}

impl AsyncKernelRequest {
    pub fn new(kernel_handle: u64) -> Self {
        Self {
            kernel_handle,
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
            args: Vec::new(),
            callback_id: None,
        }
    }

    pub fn with_grid(mut self, x: u32, y: u32, z: u32) -> Self {
        self.grid = (x, y, z);
        self
    }

    pub fn with_block(mut self, x: u32, y: u32, z: u32) -> Self {
        self.block = (x, y, z);
        self
    }

    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    pub fn with_args(mut self, args: Vec<u64>) -> Self {
        self.args = args;
        self
    }

    pub fn linear(kernel_handle: u64, n_elements: usize, block_size: u32) -> Self {
        let grid_x = ((n_elements as u32) + block_size - 1) / block_size;
        Self::new(kernel_handle)
            .with_grid(grid_x, 1, 1)
            .with_block(block_size, 1, 1)
    }
}

/// Statistics for async executor
#[derive(Debug, Clone, Default)]
pub struct AsyncExecStats {
    /// Total dispatches
    pub total_dispatched: u64,
    /// Currently pending (not yet launched)
    pub pending_count: usize,
    /// Currently running
    pub running_count: usize,
    /// Completed successfully
    pub completed_count: u64,
    /// Failed
    pub failed_count: u64,
    /// Cancelled
    pub cancelled_count: u64,
    /// Average execution time
    pub avg_exec_time: f64,
}

/// Internal executor state
struct ExecutorInner {
    /// All futures by ID
    futures: HashMap<u64, KernelFuture>,
    /// Pending queue per stream
    pending: HashMap<u32, Vec<u64>>,
    /// Running per stream
    running: HashMap<u32, Vec<u64>>,
    /// Stats
    total_exec_time: f64,
    completed_count: u64,
    failed_count: u64,
    cancelled_count: u64,
}

/// Async kernel executor
///
/// Manages async kernel dispatch and completion tracking per stream.
/// Each ExecutionContext has its own AsyncExecutor.
pub struct AsyncExecutor {
    /// Context ID (LLM ID)
    context_id: String,
    /// Stream ID for this executor
    stream_id: u32,
    /// Next future ID
    next_id: AtomicU64,
    /// Internal state
    inner: Mutex<ExecutorInner>,
}

impl AsyncExecutor {
    /// Create a new executor for a context
    pub fn new(context_id: String, stream_id: u32) -> Self {
        Self {
            context_id,
            stream_id,
            next_id: AtomicU64::new(1),
            inner: Mutex::new(ExecutorInner {
                futures: HashMap::new(),
                pending: HashMap::new(),
                running: HashMap::new(),
                total_exec_time: 0.0,
                completed_count: 0,
                failed_count: 0,
                cancelled_count: 0,
            }),
        }
    }

    /// Dispatch an async kernel
    ///
    /// Returns a KernelFuture that can be used to wait for completion.
    /// The kernel is queued for execution on this executor's stream.
    pub fn dispatch(&self, _request: AsyncKernelRequest) -> KernelFuture {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let future = KernelFuture::new(id, self.stream_id, self.context_id.clone());

        let mut inner = self.inner.lock();
        inner.futures.insert(id, future.clone());
        inner.pending.entry(self.stream_id).or_default().push(id);

        future
    }

    /// Get futures ready for launch
    ///
    /// Returns future IDs that should be launched via C++ backend.
    pub fn get_pending(&self) -> Vec<u64> {
        let inner = self.inner.lock();
        inner.pending.get(&self.stream_id).cloned().unwrap_or_default()
    }

    /// Mark a future as launched
    pub fn mark_launched(&self, future_id: u64) {
        let mut inner = self.inner.lock();

        // Remove from pending
        if let Some(pending) = inner.pending.get_mut(&self.stream_id) {
            pending.retain(|&id| id != future_id);
        }

        // Add to running
        inner.running.entry(self.stream_id).or_default().push(future_id);

        // Update future state
        if let Some(future) = inner.futures.get(&future_id) {
            future.mark_launched();
        }
    }

    /// Mark a future as completed
    pub fn mark_completed(&self, future_id: u64, exec_time: f64) {
        let mut inner = self.inner.lock();

        // Remove from running
        if let Some(running) = inner.running.get_mut(&self.stream_id) {
            running.retain(|&id| id != future_id);
        }

        // Update stats
        inner.total_exec_time += exec_time;
        inner.completed_count += 1;

        // Update future state
        if let Some(future) = inner.futures.get(&future_id) {
            future.mark_completed(KernelResult::success(exec_time));
        }
    }

    /// Mark a future as failed
    pub fn mark_failed(&self, future_id: u64, error: String) {
        let mut inner = self.inner.lock();

        // Remove from pending or running
        if let Some(pending) = inner.pending.get_mut(&self.stream_id) {
            pending.retain(|&id| id != future_id);
        }
        if let Some(running) = inner.running.get_mut(&self.stream_id) {
            running.retain(|&id| id != future_id);
        }

        inner.failed_count += 1;

        if let Some(future) = inner.futures.get(&future_id) {
            future.mark_failed(error);
        }
    }

    /// Cancel a pending future
    pub fn cancel(&self, future_id: u64) -> bool {
        let mut inner = self.inner.lock();

        // Can only cancel pending futures
        let was_pending = if let Some(pending) = inner.pending.get_mut(&self.stream_id) {
            let before = pending.len();
            pending.retain(|&id| id != future_id);
            pending.len() < before
        } else {
            false
        };

        if was_pending {
            inner.cancelled_count += 1;
            if let Some(future) = inner.futures.get(&future_id) {
                future.mark_cancelled();
            }
        }

        was_pending
    }

    /// Get a future by ID
    pub fn get_future(&self, future_id: u64) -> Option<KernelFuture> {
        self.inner.lock().futures.get(&future_id).cloned()
    }

    /// Check if there's pending work
    pub fn has_pending(&self) -> bool {
        let inner = self.inner.lock();
        !inner.pending.get(&self.stream_id).map(|v| v.is_empty()).unwrap_or(true)
    }

    /// Check if there's running work
    pub fn has_running(&self) -> bool {
        let inner = self.inner.lock();
        !inner.running.get(&self.stream_id).map(|v| v.is_empty()).unwrap_or(true)
    }

    /// Get statistics
    pub fn stats(&self) -> AsyncExecStats {
        let inner = self.inner.lock();

        let pending_count = inner.pending.get(&self.stream_id).map(|v| v.len()).unwrap_or(0);
        let running_count = inner.running.get(&self.stream_id).map(|v| v.len()).unwrap_or(0);

        let avg_exec = if inner.completed_count > 0 {
            inner.total_exec_time / inner.completed_count as f64
        } else {
            0.0
        };

        AsyncExecStats {
            total_dispatched: self.next_id.load(Ordering::SeqCst) - 1,
            pending_count,
            running_count,
            completed_count: inner.completed_count,
            failed_count: inner.failed_count,
            cancelled_count: inner.cancelled_count,
            avg_exec_time: avg_exec,
        }
    }

    /// Garbage collect completed futures
    pub fn gc(&self) {
        let mut inner = self.inner.lock();
        inner.futures.retain(|_, f| !f.state().is_terminal());
    }

    /// Clear all state
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.futures.clear();
        inner.pending.clear();
        inner.running.clear();
        inner.total_exec_time = 0.0;
        inner.completed_count = 0;
        inner.failed_count = 0;
        inner.cancelled_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_future_creation() {
        let future = KernelFuture::new(1, 0, "test".into());
        assert_eq!(future.id(), 1);
        assert_eq!(future.stream_id(), 0);
        assert_eq!(future.state(), FutureState::Pending);
        assert!(!future.is_ready());
    }

    #[test]
    fn test_future_completion() {
        let future = KernelFuture::new(1, 0, "test".into());

        future.mark_launched();
        assert_eq!(future.state(), FutureState::Running);

        future.mark_completed(KernelResult::success(0.1));
        assert_eq!(future.state(), FutureState::Completed);
        assert!(future.is_ready());

        let result = future.wait();
        assert!(result.success);
    }

    #[test]
    fn test_future_failure() {
        let future = KernelFuture::new(1, 0, "test".into());

        future.mark_launched();
        future.mark_failed("CUDA error".into());

        assert_eq!(future.state(), FutureState::Failed);
        assert!(future.is_ready());

        let result = future.wait();
        assert!(!result.success);
        assert_eq!(result.error, Some("CUDA error".into()));
    }

    #[test]
    fn test_executor_dispatch() {
        let executor = AsyncExecutor::new("llm".into(), 0);

        let request = AsyncKernelRequest::linear(0x1000, 1024, 256);
        let future = executor.dispatch(request);

        assert_eq!(future.state(), FutureState::Pending);
        assert!(executor.has_pending());

        let pending = executor.get_pending();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_executor_lifecycle() {
        let executor = AsyncExecutor::new("llm".into(), 0);

        let request = AsyncKernelRequest::new(0x1000);
        let future = executor.dispatch(request);
        let id = future.id();

        // Launch
        executor.mark_launched(id);
        assert!(!executor.has_pending());
        assert!(executor.has_running());
        assert_eq!(future.state(), FutureState::Running);

        // Complete
        executor.mark_completed(id, 0.05);
        assert!(!executor.has_running());
        assert!(future.is_ready());

        let stats = executor.stats();
        assert_eq!(stats.completed_count, 1);
    }

    #[test]
    fn test_executor_cancel() {
        let executor = AsyncExecutor::new("llm".into(), 0);

        let request = AsyncKernelRequest::new(0x1000);
        let future = executor.dispatch(request);
        let id = future.id();

        assert!(executor.cancel(id));
        assert_eq!(future.state(), FutureState::Cancelled);

        let stats = executor.stats();
        assert_eq!(stats.cancelled_count, 1);
    }

    #[test]
    fn test_multiple_dispatches() {
        let executor = AsyncExecutor::new("llm".into(), 0);

        let f1 = executor.dispatch(AsyncKernelRequest::new(0x1000));
        let f2 = executor.dispatch(AsyncKernelRequest::new(0x2000));
        let f3 = executor.dispatch(AsyncKernelRequest::new(0x3000));

        assert_eq!(executor.get_pending().len(), 3);

        executor.mark_launched(f1.id());
        executor.mark_launched(f2.id());

        assert_eq!(executor.get_pending().len(), 1);
        assert!(executor.has_running());

        executor.mark_completed(f1.id(), 0.1);
        executor.mark_completed(f2.id(), 0.2);
        executor.mark_launched(f3.id());
        executor.mark_completed(f3.id(), 0.3);

        let stats = executor.stats();
        assert_eq!(stats.completed_count, 3);
        assert!((stats.avg_exec_time - 0.2).abs() < 0.01);
    }
}
