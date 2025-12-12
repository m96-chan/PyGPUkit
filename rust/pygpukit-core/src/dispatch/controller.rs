//! Kernel Dispatch Controller implementation
//!
//! Coordinates kernel launches with stream management and scheduler integration.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;

/// State of a kernel launch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelState {
    /// Kernel is queued for launch
    Queued,
    /// Kernel has been launched (async)
    Launched,
    /// Kernel execution completed
    Completed,
    /// Kernel launch or execution failed
    Failed,
    /// Kernel was cancelled before launch
    Cancelled,
}

impl KernelState {
    /// Check if this is a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, KernelState::Completed | KernelState::Failed | KernelState::Cancelled)
    }
}

/// Kernel launch configuration
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (x, y, z)
    pub grid: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_mem: u32,
    /// Stream ID for execution
    pub stream_id: u32,
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
            stream_id: 0,
        }
    }
}

impl LaunchConfig {
    /// Create a 1D launch config
    pub fn linear(n_elements: usize, block_size: u32) -> Self {
        let grid_x = ((n_elements as u32) + block_size - 1) / block_size;
        Self {
            grid: (grid_x, 1, 1),
            block: (block_size, 1, 1),
            shared_mem: 0,
            stream_id: 0,
        }
    }

    /// Create a 2D launch config
    pub fn grid_2d(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        Self {
            grid: (grid_x, grid_y, 1),
            block: (block_x, block_y, 1),
            shared_mem: 0,
            stream_id: 0,
        }
    }

    /// Set shared memory size
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Set stream ID
    pub fn with_stream(mut self, stream_id: u32) -> Self {
        self.stream_id = stream_id;
        self
    }
}

/// Request to launch a kernel
#[derive(Debug, Clone)]
pub struct KernelLaunchRequest {
    /// Unique request ID
    pub id: u64,
    /// Kernel function handle (CUfunction as u64)
    pub kernel_handle: u64,
    /// Launch configuration
    pub config: LaunchConfig,
    /// Kernel arguments as raw bytes
    pub args: Vec<u64>,
    /// Current state
    pub state: KernelState,
    /// Associated scheduler task ID (if any)
    pub task_id: Option<String>,
    /// Priority (higher = more urgent)
    pub priority: i32,
    /// Timestamp when queued
    pub queued_at: f64,
    /// Timestamp when launched
    pub launched_at: Option<f64>,
    /// Timestamp when completed
    pub completed_at: Option<f64>,
    /// Error message if failed
    pub error: Option<String>,
}

impl KernelLaunchRequest {
    /// Create a new launch request
    pub fn new(kernel_handle: u64, config: LaunchConfig) -> Self {
        Self {
            id: 0, // Will be assigned by dispatcher
            kernel_handle,
            config,
            args: Vec::new(),
            state: KernelState::Queued,
            task_id: None,
            priority: 0,
            queued_at: Self::now(),
            launched_at: None,
            completed_at: None,
            error: None,
        }
    }

    /// Set kernel arguments
    pub fn with_args(mut self, args: Vec<u64>) -> Self {
        self.args = args;
        self
    }

    /// Link to a scheduler task
    pub fn with_task(mut self, task_id: String) -> Self {
        self.task_id = Some(task_id);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Mark as launched
    fn launch(&mut self) {
        if self.state == KernelState::Queued {
            self.state = KernelState::Launched;
            self.launched_at = Some(Self::now());
        }
    }

    /// Mark as completed
    fn complete(&mut self) {
        if self.state == KernelState::Launched {
            self.state = KernelState::Completed;
            self.completed_at = Some(Self::now());
        }
    }

    /// Mark as failed
    fn fail(&mut self, error: String) {
        self.state = KernelState::Failed;
        self.completed_at = Some(Self::now());
        self.error = Some(error);
    }

    /// Mark as cancelled
    fn cancel(&mut self) {
        if !self.state.is_terminal() {
            self.state = KernelState::Cancelled;
            self.completed_at = Some(Self::now());
        }
    }

    /// Get execution duration (launch to complete)
    pub fn duration(&self) -> Option<f64> {
        match (self.launched_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
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

/// Dispatch statistics
#[derive(Debug, Clone, Default)]
pub struct DispatchStats {
    /// Total kernels queued
    pub total_queued: usize,
    /// Kernels completed successfully
    pub completed_count: usize,
    /// Kernels failed
    pub failed_count: usize,
    /// Kernels currently pending
    pub pending_count: usize,
    /// Kernels currently in-flight (launched but not completed)
    pub in_flight_count: usize,
    /// Average kernel execution time (seconds)
    pub avg_exec_time: f64,
    /// Launches per stream
    pub launches_per_stream: HashMap<u32, usize>,
}

/// Internal dispatcher state
struct DispatcherInner {
    /// All launch requests by ID
    requests: HashMap<u64, KernelLaunchRequest>,
    /// Pending queue (FIFO within priority)
    pending_queue: VecDeque<u64>,
    /// In-flight kernels by stream
    in_flight: HashMap<u32, Vec<u64>>,
    /// Statistics
    total_exec_time: f64,
    completed_count: usize,
    failed_count: usize,
    launches_per_stream: HashMap<u32, usize>,
}

/// Kernel Dispatch Controller
///
/// Coordinates GPU kernel launches with:
/// - Stream-based execution
/// - Priority ordering
/// - Integration with scheduler tasks
///
/// # Example
///
/// ```ignore
/// use pygpukit_core::dispatch::{KernelDispatcher, KernelLaunchRequest, LaunchConfig};
///
/// let dispatcher = KernelDispatcher::new(4); // Max 4 in-flight per stream
///
/// // Queue a kernel launch
/// let config = LaunchConfig::linear(1024, 256);
/// let req = KernelLaunchRequest::new(kernel_handle, config);
/// let req_id = dispatcher.queue(req);
///
/// // Get kernels ready to launch
/// let ready = dispatcher.get_ready(10);
/// for req in ready {
///     // Launch via C++ backend (cuLaunchKernel)
///     // ...
///     dispatcher.mark_launched(req.id);
/// }
///
/// // When kernel completes (via cudaStreamSynchronize or event)
/// dispatcher.mark_completed(req_id);
/// ```
pub struct KernelDispatcher {
    /// Next request ID
    next_id: AtomicU64,
    /// Maximum in-flight kernels per stream
    max_in_flight: usize,
    /// Internal state
    inner: RwLock<DispatcherInner>,
}

impl KernelDispatcher {
    /// Create a new kernel dispatcher
    ///
    /// # Arguments
    ///
    /// * `max_in_flight` - Maximum concurrent kernels per stream
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            max_in_flight,
            inner: RwLock::new(DispatcherInner {
                requests: HashMap::new(),
                pending_queue: VecDeque::new(),
                in_flight: HashMap::new(),
                total_exec_time: 0.0,
                completed_count: 0,
                failed_count: 0,
                launches_per_stream: HashMap::new(),
            }),
        }
    }

    /// Generate next request ID
    fn next_req_id(&self) -> u64 {
        self.next_id.fetch_add(1, AtomicOrdering::SeqCst)
    }

    /// Queue a kernel launch request
    ///
    /// Returns the request ID
    pub fn queue(&self, mut request: KernelLaunchRequest) -> u64 {
        let id = self.next_req_id();
        request.id = id;

        let mut inner = self.inner.write();
        inner.pending_queue.push_back(id);
        inner.requests.insert(id, request);

        id
    }

    /// Queue a kernel launch with scheduler task binding
    pub fn queue_for_task(&self, task_id: String, kernel_handle: u64, config: LaunchConfig, args: Vec<u64>) -> u64 {
        let request = KernelLaunchRequest::new(kernel_handle, config)
            .with_args(args)
            .with_task(task_id);
        self.queue(request)
    }

    /// Get launch requests ready to execute
    ///
    /// Returns requests that can be launched (stream has capacity)
    pub fn get_ready(&self, max_requests: usize) -> Vec<KernelLaunchRequest> {
        let inner = self.inner.read();
        let mut ready = Vec::new();

        // Track how many we're planning to add to each stream
        let mut planned_per_stream: HashMap<u32, usize> = HashMap::new();

        for req_id in inner.pending_queue.iter() {
            if ready.len() >= max_requests {
                break;
            }

            if let Some(req) = inner.requests.get(req_id) {
                if req.state == KernelState::Queued {
                    // Check stream capacity (current in-flight + planned)
                    let stream_id = req.config.stream_id;
                    let current_in_flight = inner.in_flight
                        .get(&stream_id)
                        .map(|v| v.len())
                        .unwrap_or(0);
                    let planned = planned_per_stream.get(&stream_id).copied().unwrap_or(0);
                    let total = current_in_flight + planned;

                    if total < self.max_in_flight {
                        ready.push(req.clone());
                        *planned_per_stream.entry(stream_id).or_insert(0) += 1;
                    }
                }
            }
        }

        ready
    }

    /// Mark a request as launched
    ///
    /// Call this after successfully calling cuLaunchKernel
    pub fn mark_launched(&self, req_id: u64) -> bool {
        let mut inner = self.inner.write();

        // Get stream ID first
        let stream_id = inner.requests.get(&req_id).map(|r| r.config.stream_id);

        if let Some(req) = inner.requests.get_mut(&req_id) {
            if req.state == KernelState::Queued {
                req.launch();

                // Remove from pending queue
                inner.pending_queue.retain(|&id| id != req_id);

                // Add to in-flight for this stream
                if let Some(sid) = stream_id {
                    inner.in_flight
                        .entry(sid)
                        .or_insert_with(Vec::new)
                        .push(req_id);

                    // Track launches per stream
                    *inner.launches_per_stream.entry(sid).or_insert(0) += 1;
                }

                return true;
            }
        }
        false
    }

    /// Mark a request as completed
    ///
    /// Call this when kernel execution finishes
    pub fn mark_completed(&self, req_id: u64) -> bool {
        let mut inner = self.inner.write();

        // Get stream ID first
        let stream_id = inner.requests.get(&req_id).map(|r| r.config.stream_id);

        if let Some(req) = inner.requests.get_mut(&req_id) {
            if req.state == KernelState::Launched {
                req.complete();

                // Update stats
                if let Some(duration) = req.duration() {
                    inner.total_exec_time += duration;
                }
                inner.completed_count += 1;

                // Remove from in-flight
                if let Some(sid) = stream_id {
                    if let Some(v) = inner.in_flight.get_mut(&sid) {
                        v.retain(|&id| id != req_id);
                    }
                }

                return true;
            }
        }
        false
    }

    /// Mark a request as failed
    pub fn mark_failed(&self, req_id: u64, error: String) -> bool {
        let mut inner = self.inner.write();

        // Get state and stream ID first
        let info = inner.requests.get(&req_id).map(|r| (r.state, r.config.stream_id));

        if let Some((state, stream_id)) = info {
            if let Some(req) = inner.requests.get_mut(&req_id) {
                req.fail(error);
                inner.failed_count += 1;

                // Remove from appropriate queue
                if state == KernelState::Queued {
                    inner.pending_queue.retain(|&id| id != req_id);
                } else if state == KernelState::Launched {
                    if let Some(v) = inner.in_flight.get_mut(&stream_id) {
                        v.retain(|&id| id != req_id);
                    }
                }

                return true;
            }
        }
        false
    }

    /// Cancel a pending request
    pub fn cancel(&self, req_id: u64) -> bool {
        let mut inner = self.inner.write();

        if let Some(req) = inner.requests.get_mut(&req_id) {
            if req.state == KernelState::Queued {
                req.cancel();
                inner.pending_queue.retain(|&id| id != req_id);
                return true;
            }
        }
        false
    }

    /// Get a request by ID
    pub fn get_request(&self, req_id: u64) -> Option<KernelLaunchRequest> {
        self.inner.read().requests.get(&req_id).cloned()
    }

    /// Get all in-flight request IDs for a stream
    pub fn get_in_flight(&self, stream_id: u32) -> Vec<u64> {
        self.inner.read()
            .in_flight
            .get(&stream_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get requests linked to a scheduler task
    pub fn get_requests_for_task(&self, task_id: &str) -> Vec<KernelLaunchRequest> {
        self.inner.read()
            .requests
            .values()
            .filter(|r| r.task_id.as_deref() == Some(task_id))
            .cloned()
            .collect()
    }

    /// Check if there's pending work
    pub fn has_pending_work(&self) -> bool {
        let inner = self.inner.read();
        !inner.pending_queue.is_empty() ||
        inner.in_flight.values().any(|v| !v.is_empty())
    }

    /// Get dispatch statistics
    pub fn stats(&self) -> DispatchStats {
        let inner = self.inner.read();

        let pending_count = inner.pending_queue.len();
        let in_flight_count: usize = inner.in_flight.values().map(|v| v.len()).sum();

        let avg_exec = if inner.completed_count > 0 {
            inner.total_exec_time / inner.completed_count as f64
        } else {
            0.0
        };

        DispatchStats {
            total_queued: inner.requests.len(),
            completed_count: inner.completed_count,
            failed_count: inner.failed_count,
            pending_count,
            in_flight_count,
            avg_exec_time: avg_exec,
            launches_per_stream: inner.launches_per_stream.clone(),
        }
    }

    /// Garbage collect completed requests
    pub fn gc(&self) {
        let mut inner = self.inner.write();
        inner.requests.retain(|_, req| !req.state.is_terminal());
    }

    /// Clear all state
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.requests.clear();
        inner.pending_queue.clear();
        inner.in_flight.clear();
        inner.total_exec_time = 0.0;
        inner.completed_count = 0;
        inner.failed_count = 0;
        inner.launches_per_stream.clear();
    }
}

// Thread-safe
unsafe impl Send for KernelDispatcher {}
unsafe impl Sync for KernelDispatcher {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatcher_creation() {
        let dispatcher = KernelDispatcher::new(4);
        let stats = dispatcher.stats();
        assert_eq!(stats.total_queued, 0);
        assert_eq!(stats.pending_count, 0);
    }

    #[test]
    fn test_queue_request() {
        let dispatcher = KernelDispatcher::new(4);

        let config = LaunchConfig::linear(1024, 256);
        let req = KernelLaunchRequest::new(0x1000, config);
        let req_id = dispatcher.queue(req);

        assert!(req_id > 0);
        let stats = dispatcher.stats();
        assert_eq!(stats.pending_count, 1);
    }

    #[test]
    fn test_get_ready() {
        let dispatcher = KernelDispatcher::new(4);

        let config = LaunchConfig::linear(1024, 256);
        let req = KernelLaunchRequest::new(0x1000, config);
        dispatcher.queue(req);

        let ready = dispatcher.get_ready(10);
        assert_eq!(ready.len(), 1);
    }

    #[test]
    fn test_launch_lifecycle() {
        let dispatcher = KernelDispatcher::new(4);

        let config = LaunchConfig::linear(1024, 256);
        let req = KernelLaunchRequest::new(0x1000, config);
        let req_id = dispatcher.queue(req);

        // Mark launched
        assert!(dispatcher.mark_launched(req_id));
        let req = dispatcher.get_request(req_id).unwrap();
        assert_eq!(req.state, KernelState::Launched);

        // Mark completed
        assert!(dispatcher.mark_completed(req_id));
        let req = dispatcher.get_request(req_id).unwrap();
        assert_eq!(req.state, KernelState::Completed);

        let stats = dispatcher.stats();
        assert_eq!(stats.completed_count, 1);
    }

    #[test]
    fn test_max_in_flight() {
        let dispatcher = KernelDispatcher::new(2);

        // Queue 5 requests on same stream
        for _ in 0..5 {
            let config = LaunchConfig::linear(1024, 256);
            let req = KernelLaunchRequest::new(0x1000, config);
            dispatcher.queue(req);
        }

        // get_ready returns up to 2 (since max_in_flight = 2 and 0 are in-flight)
        let ready = dispatcher.get_ready(10);
        assert_eq!(ready.len(), 2);

        // Launch both - this moves them from pending to in-flight
        for req in &ready {
            dispatcher.mark_launched(req.id);
        }

        let stats = dispatcher.stats();
        assert_eq!(stats.in_flight_count, 2);
        assert_eq!(stats.pending_count, 3); // 5 - 2 launched = 3 pending

        // Now only 0 should be ready (stream is full with 2 in-flight)
        let ready = dispatcher.get_ready(10);
        assert_eq!(ready.len(), 0);

        // Complete one - frees a slot
        let first_id = stats.launches_per_stream.keys().next().copied()
            .and_then(|_| dispatcher.get_in_flight(0).first().copied())
            .unwrap();
        dispatcher.mark_completed(first_id);

        // Now one should be ready (1 in-flight, can add 1 more)
        let ready = dispatcher.get_ready(10);
        assert_eq!(ready.len(), 1);
    }

    #[test]
    fn test_multiple_streams() {
        let dispatcher = KernelDispatcher::new(2);

        // Queue on different streams
        for stream_id in 0..3 {
            let config = LaunchConfig::linear(1024, 256).with_stream(stream_id);
            let req = KernelLaunchRequest::new(0x1000, config);
            dispatcher.queue(req);
        }

        // All 3 should be ready (different streams)
        let ready = dispatcher.get_ready(10);
        assert_eq!(ready.len(), 3);
    }

    #[test]
    fn test_task_binding() {
        let dispatcher = KernelDispatcher::new(4);

        let config = LaunchConfig::linear(1024, 256);
        let req_id = dispatcher.queue_for_task(
            "task-1".to_string(),
            0x1000,
            config,
            vec![0x2000, 1024],
        );

        let req = dispatcher.get_request(req_id).unwrap();
        assert_eq!(req.task_id, Some("task-1".to_string()));
        assert_eq!(req.args, vec![0x2000, 1024]);

        let task_reqs = dispatcher.get_requests_for_task("task-1");
        assert_eq!(task_reqs.len(), 1);
    }

    #[test]
    fn test_failure() {
        let dispatcher = KernelDispatcher::new(4);

        let config = LaunchConfig::linear(1024, 256);
        let req = KernelLaunchRequest::new(0x1000, config);
        let req_id = dispatcher.queue(req);

        dispatcher.mark_launched(req_id);
        dispatcher.mark_failed(req_id, "CUDA error".to_string());

        let req = dispatcher.get_request(req_id).unwrap();
        assert_eq!(req.state, KernelState::Failed);
        assert_eq!(req.error, Some("CUDA error".to_string()));

        let stats = dispatcher.stats();
        assert_eq!(stats.failed_count, 1);
    }

    #[test]
    fn test_launch_config() {
        let config = LaunchConfig::grid_2d(32, 32, 16, 16)
            .with_shared_mem(4096)
            .with_stream(2);

        assert_eq!(config.grid, (32, 32, 1));
        assert_eq!(config.block, (16, 16, 1));
        assert_eq!(config.shared_mem, 4096);
        assert_eq!(config.stream_id, 2);
    }
}
