//! Execution Context for Multi-LLM Scheduling
//!
//! Provides per-LLM execution context with:
//! - Dedicated stream ID for kernel isolation
//! - Memory budget tracking
//! - State management (IDLE, RUNNING, PAUSED)
//! - Async kernel execution with KernelFuture
//! - Per-context session management
//!
//! Each LLM instance is bound to exactly one ExecutionContext.

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::collections::HashMap;
use parking_lot::RwLock;

use super::async_exec::{AsyncExecutor, AsyncKernelRequest, KernelFuture, AsyncExecStats};

/// State of an execution context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContextState {
    /// Context created but not running
    Idle = 0,
    /// Context is actively executing kernels
    Running = 1,
    /// Context is paused (e.g., waiting for memory)
    Paused = 2,
}

impl Default for ContextState {
    fn default() -> Self {
        ContextState::Idle
    }
}

/// Per-LLM Execution Context
///
/// Each LLM instance is bound to exactly one ExecutionContext.
/// Provides:
/// - Dedicated stream ID for kernel isolation
/// - Memory budget tracking
/// - State management
/// - Async kernel execution
/// - Per-context session
pub struct ExecutionContext {
    /// Unique identifier for the LLM instance
    llm_id: String,
    /// Current state
    state: ContextState,
    /// Assigned stream ID (managed by C++ StreamPool)
    stream_id: u32,
    /// Maximum VRAM budget in bytes (0 = unlimited)
    max_vram: usize,
    /// Currently used VRAM
    used_vram: AtomicUsize,
    /// Allocated buffer tracking: buffer_id -> size
    allocated_buffers: RwLock<HashMap<u64, usize>>,
    /// Async kernel executor
    executor: AsyncExecutor,
    /// Per-context session active flag
    session_active: AtomicBool,
}

impl ExecutionContext {
    /// Create a new execution context
    ///
    /// # Arguments
    ///
    /// * `llm_id` - Unique identifier for the LLM instance
    /// * `stream_id` - Assigned stream ID
    /// * `max_vram` - Maximum VRAM budget in bytes (0 = unlimited)
    pub fn new(llm_id: String, stream_id: u32, max_vram: usize) -> Self {
        let executor = AsyncExecutor::new(llm_id.clone(), stream_id);
        Self {
            llm_id,
            state: ContextState::Idle,
            stream_id,
            max_vram,
            used_vram: AtomicUsize::new(0),
            allocated_buffers: RwLock::new(HashMap::new()),
            executor,
            session_active: AtomicBool::new(false),
        }
    }

    // --- Accessors ---

    /// Get the LLM ID
    #[inline]
    pub fn llm_id(&self) -> &str {
        &self.llm_id
    }

    /// Get current state
    #[inline]
    pub fn state(&self) -> ContextState {
        self.state
    }

    /// Get assigned stream ID
    #[inline]
    pub fn stream_id(&self) -> u32 {
        self.stream_id
    }

    /// Get maximum VRAM budget
    #[inline]
    pub fn max_vram(&self) -> usize {
        self.max_vram
    }

    /// Get currently used VRAM
    #[inline]
    pub fn used_vram(&self) -> usize {
        self.used_vram.load(Ordering::SeqCst)
    }

    /// Get available VRAM
    #[inline]
    pub fn available_vram(&self) -> usize {
        if self.max_vram == 0 {
            usize::MAX
        } else {
            self.max_vram.saturating_sub(self.used_vram())
        }
    }

    /// Get number of allocated buffers
    pub fn buffer_count(&self) -> usize {
        self.allocated_buffers.read().len()
    }

    // --- State Management ---

    /// Set context state
    pub fn set_state(&mut self, state: ContextState) {
        self.state = state;
    }

    /// Start the context (set to Running)
    pub fn start(&mut self) {
        self.state = ContextState::Running;
    }

    /// Pause the context
    pub fn pause(&mut self) {
        self.state = ContextState::Paused;
    }

    /// Stop the context (set to Idle)
    pub fn stop(&mut self) {
        self.state = ContextState::Idle;
    }

    /// Check if context is running
    #[inline]
    pub fn is_running(&self) -> bool {
        self.state == ContextState::Running
    }

    // --- Memory Tracking ---

    /// Check if allocation fits within budget
    pub fn can_allocate(&self, size: usize) -> bool {
        if self.max_vram == 0 {
            return true; // Unlimited
        }
        self.used_vram() + size <= self.max_vram
    }

    /// Track a memory allocation
    ///
    /// # Arguments
    ///
    /// * `buffer_id` - Unique buffer identifier
    /// * `size` - Size in bytes
    ///
    /// # Returns
    ///
    /// `true` if allocation fits within budget, `false` otherwise
    pub fn track_allocation(&self, buffer_id: u64, size: usize) -> bool {
        if !self.can_allocate(size) {
            return false;
        }

        let mut buffers = self.allocated_buffers.write();
        buffers.insert(buffer_id, size);
        self.used_vram.fetch_add(size, Ordering::SeqCst);
        true
    }

    /// Track a memory deallocation
    ///
    /// # Arguments
    ///
    /// * `buffer_id` - Buffer identifier to deallocate
    pub fn track_deallocation(&self, buffer_id: u64) {
        let mut buffers = self.allocated_buffers.write();
        if let Some(size) = buffers.remove(&buffer_id) {
            // Saturating sub to handle potential underflow
            let current = self.used_vram.load(Ordering::SeqCst);
            let new_val = current.saturating_sub(size);
            self.used_vram.store(new_val, Ordering::SeqCst);
        }
    }

    /// Get size of a specific buffer
    pub fn get_buffer_size(&self, buffer_id: u64) -> Option<usize> {
        self.allocated_buffers.read().get(&buffer_id).copied()
    }

    /// Clear all tracked allocations
    pub fn clear_allocations(&self) {
        let mut buffers = self.allocated_buffers.write();
        buffers.clear();
        self.used_vram.store(0, Ordering::SeqCst);
    }

    // --- Async Execution ---

    /// Dispatch an async kernel
    ///
    /// Returns a KernelFuture that can be used to wait for completion.
    /// The kernel is queued for execution on this context's stream.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let request = AsyncKernelRequest::linear(kernel_handle, n_elements, 256);
    /// let future = ctx.dispatch_async(request);
    ///
    /// // Do other work...
    ///
    /// let result = future.wait();
    /// ```
    pub fn dispatch_async(&self, request: AsyncKernelRequest) -> KernelFuture {
        self.executor.dispatch(request)
    }

    /// Get pending futures for this context
    pub fn get_pending_futures(&self) -> Vec<u64> {
        self.executor.get_pending()
    }

    /// Mark a future as launched
    pub fn mark_future_launched(&self, future_id: u64) {
        self.executor.mark_launched(future_id);
    }

    /// Mark a future as completed
    pub fn mark_future_completed(&self, future_id: u64, exec_time: f64) {
        self.executor.mark_completed(future_id, exec_time);
    }

    /// Mark a future as failed
    pub fn mark_future_failed(&self, future_id: u64, error: String) {
        self.executor.mark_failed(future_id, error);
    }

    /// Cancel a pending future
    pub fn cancel_future(&self, future_id: u64) -> bool {
        self.executor.cancel(future_id)
    }

    /// Get a future by ID
    pub fn get_future(&self, future_id: u64) -> Option<KernelFuture> {
        self.executor.get_future(future_id)
    }

    /// Check if there are pending kernels
    pub fn has_pending_kernels(&self) -> bool {
        self.executor.has_pending()
    }

    /// Check if there are running kernels
    pub fn has_running_kernels(&self) -> bool {
        self.executor.has_running()
    }

    /// Get async execution statistics
    pub fn async_stats(&self) -> AsyncExecStats {
        self.executor.stats()
    }

    // --- Per-Context Session ---

    /// Start a session for this context
    ///
    /// Unlike the global session, per-context sessions allow independent
    /// LLM execution. Each context can have its own session lifecycle.
    pub fn start_session(&self) {
        self.session_active.store(true, Ordering::SeqCst);
    }

    /// End the session for this context
    ///
    /// This does NOT synchronize the stream - use `sync()` for that.
    /// It just marks the session as inactive.
    pub fn end_session(&self) {
        self.session_active.store(false, Ordering::SeqCst);
    }

    /// Check if a session is active for this context
    pub fn is_session_active(&self) -> bool {
        self.session_active.load(Ordering::SeqCst)
    }

    /// Garbage collect completed futures
    pub fn gc_futures(&self) {
        self.executor.gc();
    }

    /// Clear all async state
    pub fn clear_async_state(&self) {
        self.executor.clear();
    }
}

/// Execution context statistics
#[derive(Debug, Clone, Default)]
pub struct ContextStats {
    /// LLM ID
    pub llm_id: String,
    /// Current state
    pub state: ContextState,
    /// Assigned stream ID
    pub stream_id: u32,
    /// Maximum VRAM budget
    pub max_vram: usize,
    /// Currently used VRAM
    pub used_vram: usize,
    /// Available VRAM
    pub available_vram: usize,
    /// Number of allocated buffers
    pub buffer_count: usize,
}

impl From<&ExecutionContext> for ContextStats {
    fn from(ctx: &ExecutionContext) -> Self {
        Self {
            llm_id: ctx.llm_id.clone(),
            state: ctx.state,
            stream_id: ctx.stream_id,
            max_vram: ctx.max_vram,
            used_vram: ctx.used_vram(),
            available_vram: ctx.available_vram(),
            buffer_count: ctx.buffer_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = ExecutionContext::new("gpt2".to_string(), 0, 1024 * 1024);
        assert_eq!(ctx.llm_id(), "gpt2");
        assert_eq!(ctx.stream_id(), 0);
        assert_eq!(ctx.max_vram(), 1024 * 1024);
        assert_eq!(ctx.used_vram(), 0);
        assert_eq!(ctx.state(), ContextState::Idle);
    }

    #[test]
    fn test_state_transitions() {
        let mut ctx = ExecutionContext::new("gpt2".to_string(), 0, 0);

        assert_eq!(ctx.state(), ContextState::Idle);

        ctx.start();
        assert_eq!(ctx.state(), ContextState::Running);
        assert!(ctx.is_running());

        ctx.pause();
        assert_eq!(ctx.state(), ContextState::Paused);

        ctx.stop();
        assert_eq!(ctx.state(), ContextState::Idle);
    }

    #[test]
    fn test_memory_tracking() {
        let ctx = ExecutionContext::new("gpt2".to_string(), 0, 1000);

        // Track allocations
        assert!(ctx.track_allocation(1, 400));
        assert_eq!(ctx.used_vram(), 400);

        assert!(ctx.track_allocation(2, 400));
        assert_eq!(ctx.used_vram(), 800);

        // Should fail - exceeds budget
        assert!(!ctx.track_allocation(3, 300));
        assert_eq!(ctx.used_vram(), 800);

        // Deallocate
        ctx.track_deallocation(1);
        assert_eq!(ctx.used_vram(), 400);

        // Now should succeed
        assert!(ctx.track_allocation(3, 300));
        assert_eq!(ctx.used_vram(), 700);
    }

    #[test]
    fn test_unlimited_budget() {
        let ctx = ExecutionContext::new("gpt2".to_string(), 0, 0);

        // Should always succeed with unlimited budget
        assert!(ctx.can_allocate(usize::MAX / 2));
        assert!(ctx.track_allocation(1, 1_000_000_000));
        assert_eq!(ctx.available_vram(), usize::MAX);
    }

    #[test]
    fn test_context_stats() {
        let ctx = ExecutionContext::new("gpt2".to_string(), 5, 2000);
        ctx.track_allocation(1, 500);

        let stats = ContextStats::from(&ctx);
        assert_eq!(stats.llm_id, "gpt2");
        assert_eq!(stats.stream_id, 5);
        assert_eq!(stats.max_vram, 2000);
        assert_eq!(stats.used_vram, 500);
        assert_eq!(stats.available_vram, 1500);
        assert_eq!(stats.buffer_count, 1);
    }

    #[test]
    fn test_async_dispatch() {
        let ctx = ExecutionContext::new("tts".to_string(), 0, 0);

        let request = AsyncKernelRequest::new(0x1000);
        let future = ctx.dispatch_async(request);

        assert!(ctx.has_pending_kernels());
        assert!(!ctx.has_running_kernels());

        let pending = ctx.get_pending_futures();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0], future.id());
    }

    #[test]
    fn test_async_lifecycle() {
        let ctx = ExecutionContext::new("llm".to_string(), 1, 0);

        let request = AsyncKernelRequest::linear(0x2000, 1024, 256);
        let future = ctx.dispatch_async(request);
        let id = future.id();

        // Launch
        ctx.mark_future_launched(id);
        assert!(!ctx.has_pending_kernels());
        assert!(ctx.has_running_kernels());

        // Complete
        ctx.mark_future_completed(id, 0.05);
        assert!(!ctx.has_running_kernels());
        assert!(future.is_ready());

        let stats = ctx.async_stats();
        assert_eq!(stats.completed_count, 1);
    }

    #[test]
    fn test_per_context_session() {
        let ctx = ExecutionContext::new("vision".to_string(), 2, 0);

        assert!(!ctx.is_session_active());

        ctx.start_session();
        assert!(ctx.is_session_active());

        ctx.end_session();
        assert!(!ctx.is_session_active());
    }

    #[test]
    fn test_multiple_contexts_independent_sessions() {
        let tts_ctx = ExecutionContext::new("tts".to_string(), 0, 0);
        let llm_ctx = ExecutionContext::new("llm".to_string(), 1, 0);
        let vision_ctx = ExecutionContext::new("vision".to_string(), 2, 0);

        // Start sessions independently
        tts_ctx.start_session();
        assert!(tts_ctx.is_session_active());
        assert!(!llm_ctx.is_session_active());
        assert!(!vision_ctx.is_session_active());

        llm_ctx.start_session();
        assert!(tts_ctx.is_session_active());
        assert!(llm_ctx.is_session_active());
        assert!(!vision_ctx.is_session_active());

        // End TTS session, others continue
        tts_ctx.end_session();
        assert!(!tts_ctx.is_session_active());
        assert!(llm_ctx.is_session_active());

        // Dispatch async kernel on LLM while session is active
        let request = AsyncKernelRequest::new(0x3000);
        let future = llm_ctx.dispatch_async(request);
        assert!(llm_ctx.has_pending_kernels());

        llm_ctx.mark_future_launched(future.id());
        llm_ctx.mark_future_completed(future.id(), 0.1);

        assert!(future.is_ready());
    }

    #[test]
    fn test_cancel_pending_future() {
        let ctx = ExecutionContext::new("test".to_string(), 0, 0);

        let f1 = ctx.dispatch_async(AsyncKernelRequest::new(0x1000));
        let f2 = ctx.dispatch_async(AsyncKernelRequest::new(0x2000));

        assert_eq!(ctx.get_pending_futures().len(), 2);

        // Cancel first
        assert!(ctx.cancel_future(f1.id()));
        assert_eq!(ctx.get_pending_futures().len(), 1);

        // Can't cancel already running
        ctx.mark_future_launched(f2.id());
        assert!(!ctx.cancel_future(f2.id()));
    }
}
