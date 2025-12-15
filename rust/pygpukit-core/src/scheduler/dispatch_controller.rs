//! Multi-LLM Dispatch Controller
//!
//! Manages multiple LLM execution contexts on a single GPU:
//! - Stream pool for multi-LLM execution
//! - Execution context lifecycle management
//! - Global VRAM budget tracking
//! - Session management

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use parking_lot::RwLock;

use super::execution_context::{ExecutionContext, ContextState, ContextStats};
use super::async_exec::{KernelFuture, AsyncKernelRequest, AsyncExecStats};

/// Controller statistics
#[derive(Debug, Clone, Default)]
pub struct ControllerStats {
    /// Whether controller is initialized
    pub initialized: bool,
    /// Device ID
    pub device_id: i32,
    /// Total VRAM budget
    pub total_vram_budget: usize,
    /// Device total memory
    pub device_total_memory: usize,
    /// Total VRAM used across all contexts
    pub used_vram: usize,
    /// Available VRAM
    pub available_vram: usize,
    /// Number of active contexts
    pub context_count: usize,
    /// Number of streams in pool
    pub stream_pool_size: usize,
}

/// Internal controller state
struct ControllerInner {
    /// Device ID
    device_id: i32,
    /// Total VRAM budget for all contexts
    total_vram_budget: usize,
    /// Device total memory (from CUDA)
    device_total_memory: usize,
    /// Execution contexts by LLM ID
    contexts: HashMap<String, ExecutionContext>,
    /// Available stream IDs (simple pool)
    available_streams: Vec<u32>,
    /// Next stream ID to allocate if pool empty
    next_stream_id: u32,
}

/// Multi-LLM Dispatch Controller
///
/// Manages execution contexts for multiple LLM instances on a single GPU.
/// Uses stream-based isolation for concurrent execution.
///
/// # Example
///
/// ```
/// use pygpukit_core::scheduler::{MultiLLMController, ContextState};
///
/// let controller = MultiLLMController::new();
/// // Initialize with device_id=0, device_total_memory=8GB, total_vram_budget=8GB
/// controller.initialize(0, 8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024);
///
/// // Create context for first LLM with 4GB budget
/// let stream_id = controller.create_context("gpt2_a", 4 * 1024 * 1024 * 1024).unwrap();
///
/// // Create context for second LLM with 4GB budget
/// let stream_id2 = controller.create_context("gpt2_b", 4 * 1024 * 1024 * 1024).unwrap();
///
/// // Start session
/// controller.start_session();
/// // ... execute kernels ...
/// controller.end_session();
/// ```
pub struct MultiLLMController {
    /// Whether controller is initialized
    initialized: AtomicBool,
    /// Whether a session is active
    session_active: AtomicBool,
    /// Internal state
    inner: RwLock<ControllerInner>,
}

impl MultiLLMController {
    /// Create a new controller (uninitialized)
    pub fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
            session_active: AtomicBool::new(false),
            inner: RwLock::new(ControllerInner {
                device_id: 0,
                total_vram_budget: 0,
                device_total_memory: 0,
                contexts: HashMap::new(),
                available_streams: Vec::new(),
                next_stream_id: 0,
            }),
        }
    }

    /// Initialize the controller
    ///
    /// # Arguments
    ///
    /// * `device_id` - CUDA device ID
    /// * `total_vram_budget` - Total VRAM budget for all contexts (0 = device total)
    ///
    /// Note: This does NOT call CUDA APIs directly. The caller should:
    /// 1. Initialize CUDA driver context via C++
    /// 2. Get device total memory via C++
    /// 3. Call this with the device info
    pub fn initialize(&self, device_id: i32, device_total_memory: usize, total_vram_budget: usize) {
        let mut inner = self.inner.write();

        inner.device_id = device_id;
        inner.device_total_memory = device_total_memory;
        inner.total_vram_budget = if total_vram_budget == 0 || total_vram_budget > device_total_memory {
            device_total_memory
        } else {
            total_vram_budget
        };

        // Pre-allocate stream IDs (actual CUDA streams created by C++)
        inner.available_streams = (0..8).collect();
        inner.next_stream_id = 8;

        self.initialized.store(true, Ordering::SeqCst);
    }

    /// Check if controller is initialized
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    /// Create an execution context for an LLM
    ///
    /// # Arguments
    ///
    /// * `llm_id` - Unique LLM identifier
    /// * `max_vram` - Maximum VRAM for this LLM (0 = share global budget)
    ///
    /// # Returns
    ///
    /// The assigned stream ID for this context
    ///
    /// # Panics
    ///
    /// Panics if controller is not initialized or llm_id already exists
    pub fn create_context(&self, llm_id: &str, max_vram: usize) -> Result<u32, String> {
        if !self.is_initialized() {
            return Err("Controller not initialized".to_string());
        }

        let mut inner = self.inner.write();

        // Check if context already exists
        if inner.contexts.contains_key(llm_id) {
            return Err(format!("Context already exists for LLM: {}", llm_id));
        }

        // Acquire a stream ID
        let stream_id = inner.available_streams.pop().unwrap_or_else(|| {
            let id = inner.next_stream_id;
            inner.next_stream_id += 1;
            id
        });

        // Create context
        let context = ExecutionContext::new(llm_id.to_string(), stream_id, max_vram);
        inner.contexts.insert(llm_id.to_string(), context);

        Ok(stream_id)
    }

    /// Get an execution context by LLM ID
    pub fn get_context(&self, llm_id: &str) -> Option<ContextStats> {
        let inner = self.inner.read();
        inner.contexts.get(llm_id).map(ContextStats::from)
    }

    /// Get mutable access to context for state changes
    pub fn with_context_mut<F, R>(&self, llm_id: &str, f: F) -> Option<R>
    where
        F: FnOnce(&mut ExecutionContext) -> R,
    {
        let mut inner = self.inner.write();
        inner.contexts.get_mut(llm_id).map(f)
    }

    /// Destroy an execution context
    pub fn destroy_context(&self, llm_id: &str) -> bool {
        let mut inner = self.inner.write();

        if let Some(ctx) = inner.contexts.remove(llm_id) {
            // Return stream ID to pool
            inner.available_streams.push(ctx.stream_id());
            true
        } else {
            false
        }
    }

    /// List all active context IDs
    pub fn list_contexts(&self) -> Vec<String> {
        let inner = self.inner.read();
        inner.contexts.keys().cloned().collect()
    }

    /// Get number of active contexts
    pub fn context_count(&self) -> usize {
        self.inner.read().contexts.len()
    }

    /// Get stream ID for a context
    pub fn get_stream_id(&self, llm_id: &str) -> Option<u32> {
        self.inner.read().contexts.get(llm_id).map(|c| c.stream_id())
    }

    // --- Memory Tracking ---

    /// Track a memory allocation for a context
    pub fn track_allocation(&self, llm_id: &str, buffer_id: u64, size: usize) -> bool {
        let inner = self.inner.read();
        if let Some(ctx) = inner.contexts.get(llm_id) {
            ctx.track_allocation(buffer_id, size)
        } else {
            false
        }
    }

    /// Track a memory deallocation for a context
    pub fn track_deallocation(&self, llm_id: &str, buffer_id: u64) {
        let inner = self.inner.read();
        if let Some(ctx) = inner.contexts.get(llm_id) {
            ctx.track_deallocation(buffer_id);
        }
    }

    /// Get total VRAM used across all contexts
    pub fn used_vram(&self) -> usize {
        let inner = self.inner.read();
        inner.contexts.values().map(|c| c.used_vram()).sum()
    }

    /// Get available VRAM (global budget - used)
    pub fn available_vram(&self) -> usize {
        let inner = self.inner.read();
        inner.total_vram_budget.saturating_sub(
            inner.contexts.values().map(|c| c.used_vram()).sum()
        )
    }

    // --- Session Management ---

    /// Start a session (mark all contexts as running)
    pub fn start_session(&self) {
        if self.session_active.swap(true, Ordering::SeqCst) {
            return; // Already active
        }

        let mut inner = self.inner.write();
        for ctx in inner.contexts.values_mut() {
            if ctx.state() == ContextState::Idle {
                ctx.start();
            }
        }
    }

    /// End a session (mark all contexts as idle)
    pub fn end_session(&self) {
        if !self.session_active.swap(false, Ordering::SeqCst) {
            return; // Not active
        }

        let mut inner = self.inner.write();
        for ctx in inner.contexts.values_mut() {
            ctx.stop();
        }
    }

    /// Check if a session is active
    #[inline]
    pub fn is_session_active(&self) -> bool {
        self.session_active.load(Ordering::SeqCst)
    }

    // --- Statistics ---

    /// Get controller statistics
    pub fn stats(&self) -> ControllerStats {
        let inner = self.inner.read();
        let used = inner.contexts.values().map(|c| c.used_vram()).sum();

        ControllerStats {
            initialized: self.is_initialized(),
            device_id: inner.device_id,
            total_vram_budget: inner.total_vram_budget,
            device_total_memory: inner.device_total_memory,
            used_vram: used,
            available_vram: inner.total_vram_budget.saturating_sub(used),
            context_count: inner.contexts.len(),
            stream_pool_size: inner.available_streams.len() + inner.contexts.len(),
        }
    }

    /// Reset the controller (destroy all contexts)
    pub fn reset(&self) {
        self.session_active.store(false, Ordering::SeqCst);

        let mut inner = self.inner.write();

        // Collect stream IDs first to avoid borrow conflict
        let stream_ids: Vec<u32> = inner.contexts.values().map(|c| c.stream_id()).collect();

        // Return all stream IDs to pool
        for stream_id in stream_ids {
            inner.available_streams.push(stream_id);
        }

        inner.contexts.clear();
    }

    // --- Async Execution ---

    /// Dispatch an async kernel for a specific LLM context
    ///
    /// Returns a KernelFuture that can be used to wait for completion.
    ///
    /// # Arguments
    ///
    /// * `llm_id` - LLM identifier
    /// * `request` - Kernel dispatch request
    ///
    /// # Returns
    ///
    /// KernelFuture for tracking execution, or error if context not found
    pub fn dispatch_async(&self, llm_id: &str, request: AsyncKernelRequest) -> Result<KernelFuture, String> {
        let inner = self.inner.read();
        let ctx = inner.contexts.get(llm_id)
            .ok_or_else(|| format!("Context not found: {}", llm_id))?;

        Ok(ctx.dispatch_async(request))
    }

    /// Get pending futures for a context
    pub fn get_pending_futures(&self, llm_id: &str) -> Option<Vec<u64>> {
        let inner = self.inner.read();
        inner.contexts.get(llm_id).map(|c| c.get_pending_futures())
    }

    /// Mark a future as launched
    pub fn mark_future_launched(&self, llm_id: &str, future_id: u64) {
        let inner = self.inner.read();
        if let Some(ctx) = inner.contexts.get(llm_id) {
            ctx.mark_future_launched(future_id);
        }
    }

    /// Mark a future as completed
    pub fn mark_future_completed(&self, llm_id: &str, future_id: u64, exec_time: f64) {
        let inner = self.inner.read();
        if let Some(ctx) = inner.contexts.get(llm_id) {
            ctx.mark_future_completed(future_id, exec_time);
        }
    }

    /// Mark a future as failed
    pub fn mark_future_failed(&self, llm_id: &str, future_id: u64, error: String) {
        let inner = self.inner.read();
        if let Some(ctx) = inner.contexts.get(llm_id) {
            ctx.mark_future_failed(future_id, error);
        }
    }

    /// Cancel a pending future
    pub fn cancel_future(&self, llm_id: &str, future_id: u64) -> bool {
        let inner = self.inner.read();
        inner.contexts.get(llm_id)
            .map(|c| c.cancel_future(future_id))
            .unwrap_or(false)
    }

    /// Get a future by ID from a context
    pub fn get_future(&self, llm_id: &str, future_id: u64) -> Option<KernelFuture> {
        let inner = self.inner.read();
        inner.contexts.get(llm_id)
            .and_then(|c| c.get_future(future_id))
    }

    /// Get async execution stats for a context
    pub fn async_stats(&self, llm_id: &str) -> Option<AsyncExecStats> {
        let inner = self.inner.read();
        inner.contexts.get(llm_id).map(|c| c.async_stats())
    }

    // --- Per-Context Session Management ---

    /// Start a session for a specific context
    ///
    /// Unlike the global session, per-context sessions allow independent
    /// LLM execution. Each context can have its own session lifecycle.
    pub fn start_context_session(&self, llm_id: &str) -> bool {
        let inner = self.inner.read();
        if let Some(ctx) = inner.contexts.get(llm_id) {
            ctx.start_session();
            true
        } else {
            false
        }
    }

    /// End a session for a specific context
    pub fn end_context_session(&self, llm_id: &str) -> bool {
        let inner = self.inner.read();
        if let Some(ctx) = inner.contexts.get(llm_id) {
            ctx.end_session();
            true
        } else {
            false
        }
    }

    /// Check if a specific context has an active session
    pub fn is_context_session_active(&self, llm_id: &str) -> Option<bool> {
        let inner = self.inner.read();
        inner.contexts.get(llm_id).map(|c| c.is_session_active())
    }
}

impl Default for MultiLLMController {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-safe
unsafe impl Send for MultiLLMController {}
unsafe impl Sync for MultiLLMController {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let controller = MultiLLMController::new();
        assert!(!controller.is_initialized());
    }

    #[test]
    fn test_initialization() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        assert!(controller.is_initialized());
        let stats = controller.stats();
        assert_eq!(stats.device_id, 0);
        assert_eq!(stats.total_vram_budget, 8_000_000_000);
    }

    #[test]
    fn test_create_context() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        let stream_id = controller.create_context("gpt2_a", 4_000_000_000).unwrap();
        assert!(stream_id < 8); // From pre-allocated pool

        let ctx = controller.get_context("gpt2_a").unwrap();
        assert_eq!(ctx.llm_id, "gpt2_a");
        assert_eq!(ctx.stream_id, stream_id);
        assert_eq!(ctx.max_vram, 4_000_000_000);
    }

    #[test]
    fn test_multiple_contexts() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        let s1 = controller.create_context("gpt2_a", 2_000_000_000).unwrap();
        let s2 = controller.create_context("gpt2_b", 2_000_000_000).unwrap();
        let s3 = controller.create_context("llama", 2_000_000_000).unwrap();

        // All should have different stream IDs
        assert_ne!(s1, s2);
        assert_ne!(s2, s3);

        assert_eq!(controller.context_count(), 3);

        let ids = controller.list_contexts();
        assert!(ids.contains(&"gpt2_a".to_string()));
        assert!(ids.contains(&"gpt2_b".to_string()));
        assert!(ids.contains(&"llama".to_string()));
    }

    #[test]
    fn test_duplicate_context_error() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        controller.create_context("gpt2", 0).unwrap();
        let result = controller.create_context("gpt2", 0);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already exists"));
    }

    #[test]
    fn test_destroy_context() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        let stream_id = controller.create_context("gpt2", 0).unwrap();
        assert!(controller.destroy_context("gpt2"));
        assert!(controller.get_context("gpt2").is_none());

        // Stream ID should be reusable
        let new_stream = controller.create_context("llama", 0).unwrap();
        assert_eq!(new_stream, stream_id);
    }

    #[test]
    fn test_memory_tracking() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        controller.create_context("gpt2", 1_000_000).unwrap();

        assert!(controller.track_allocation("gpt2", 1, 500_000));
        assert!(controller.track_allocation("gpt2", 2, 400_000));

        let ctx = controller.get_context("gpt2").unwrap();
        assert_eq!(ctx.used_vram, 900_000);

        // Should fail - exceeds per-context budget
        assert!(!controller.track_allocation("gpt2", 3, 200_000));

        // Deallocate and retry
        controller.track_deallocation("gpt2", 1);
        assert!(controller.track_allocation("gpt2", 3, 200_000));
    }

    #[test]
    fn test_session_management() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        controller.create_context("gpt2", 0).unwrap();

        assert!(!controller.is_session_active());

        controller.start_session();
        assert!(controller.is_session_active());

        let ctx = controller.get_context("gpt2").unwrap();
        assert_eq!(ctx.state, ContextState::Running);

        controller.end_session();
        assert!(!controller.is_session_active());

        let ctx = controller.get_context("gpt2").unwrap();
        assert_eq!(ctx.state, ContextState::Idle);
    }

    #[test]
    fn test_global_vram_tracking() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 1_000_000, 0);

        controller.create_context("gpt2_a", 0).unwrap();
        controller.create_context("gpt2_b", 0).unwrap();

        controller.track_allocation("gpt2_a", 1, 300_000);
        controller.track_allocation("gpt2_b", 2, 200_000);

        assert_eq!(controller.used_vram(), 500_000);
        assert_eq!(controller.available_vram(), 500_000);
    }

    #[test]
    fn test_reset() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        controller.create_context("gpt2_a", 0).unwrap();
        controller.create_context("gpt2_b", 0).unwrap();
        controller.start_session();

        controller.reset();

        assert!(!controller.is_session_active());
        assert_eq!(controller.context_count(), 0);
    }

    #[test]
    fn test_async_dispatch() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        controller.create_context("tts", 0).unwrap();

        let request = AsyncKernelRequest::new(0x1000);
        let future = controller.dispatch_async("tts", request).unwrap();

        assert_eq!(future.context_id(), "tts");
        assert!(!future.is_ready());

        let pending = controller.get_pending_futures("tts").unwrap();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_async_lifecycle() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        controller.create_context("llm", 0).unwrap();

        let request = AsyncKernelRequest::linear(0x2000, 1024, 256);
        let future = controller.dispatch_async("llm", request).unwrap();
        let id = future.id();

        // Launch
        controller.mark_future_launched("llm", id);
        assert_eq!(controller.get_pending_futures("llm").unwrap().len(), 0);

        // Complete
        controller.mark_future_completed("llm", id, 0.05);
        assert!(future.is_ready());

        let stats = controller.async_stats("llm").unwrap();
        assert_eq!(stats.completed_count, 1);
    }

    #[test]
    fn test_per_context_session() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        controller.create_context("tts", 0).unwrap();
        controller.create_context("llm", 0).unwrap();

        // Start session for TTS only
        assert!(controller.start_context_session("tts"));
        assert_eq!(controller.is_context_session_active("tts"), Some(true));
        assert_eq!(controller.is_context_session_active("llm"), Some(false));

        // Start session for LLM
        assert!(controller.start_context_session("llm"));
        assert_eq!(controller.is_context_session_active("llm"), Some(true));

        // End TTS session, LLM continues
        assert!(controller.end_context_session("tts"));
        assert_eq!(controller.is_context_session_active("tts"), Some(false));
        assert_eq!(controller.is_context_session_active("llm"), Some(true));
    }

    #[test]
    fn test_multi_context_async_dispatch() {
        let controller = MultiLLMController::new();
        controller.initialize(0, 8_000_000_000, 0);

        controller.create_context("tts", 0).unwrap();
        controller.create_context("llm", 0).unwrap();
        controller.create_context("vision", 0).unwrap();

        // Start independent sessions
        controller.start_context_session("tts");
        controller.start_context_session("llm");
        controller.start_context_session("vision");

        // Dispatch kernels to different contexts
        let tts_future = controller.dispatch_async("tts", AsyncKernelRequest::new(0x1000)).unwrap();
        let llm_future = controller.dispatch_async("llm", AsyncKernelRequest::new(0x2000)).unwrap();
        let vision_future = controller.dispatch_async("vision", AsyncKernelRequest::new(0x3000)).unwrap();

        // Each context has exactly one pending
        assert_eq!(controller.get_pending_futures("tts").unwrap().len(), 1);
        assert_eq!(controller.get_pending_futures("llm").unwrap().len(), 1);
        assert_eq!(controller.get_pending_futures("vision").unwrap().len(), 1);

        // Complete them in different order
        controller.mark_future_launched("llm", llm_future.id());
        controller.mark_future_completed("llm", llm_future.id(), 0.1);

        controller.mark_future_launched("tts", tts_future.id());
        controller.mark_future_completed("tts", tts_future.id(), 0.05);

        controller.mark_future_launched("vision", vision_future.id());
        controller.mark_future_completed("vision", vision_future.id(), 0.2);

        // All should be ready
        assert!(tts_future.is_ready());
        assert!(llm_future.is_ready());
        assert!(vision_future.is_ready());

        // Check completion times
        assert!(tts_future.wait().exec_time < llm_future.wait().exec_time);
    }
}
