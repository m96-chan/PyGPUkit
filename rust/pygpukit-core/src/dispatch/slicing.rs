//! Micro-Slicing Framework
//!
//! Splits GPU kernels into small runnable slices to improve
//! latency and fairness under QoS constraints.

use std::collections::VecDeque;

/// Slice configuration
#[derive(Debug, Clone)]
pub struct SliceConfig {
    /// Maximum work items per slice
    pub max_items_per_slice: usize,
    /// Maximum duration per slice in milliseconds
    pub max_duration_ms: f64,
    /// Minimum number of slices to create
    pub min_slices: usize,
    /// Maximum number of slices to create
    pub max_slices: usize,
    /// Enable adaptive slice sizing
    pub adaptive: bool,
}

impl Default for SliceConfig {
    fn default() -> Self {
        Self {
            max_items_per_slice: 65536,
            max_duration_ms: 1.0,
            min_slices: 1,
            max_slices: 256,
            adaptive: true,
        }
    }
}

impl SliceConfig {
    /// Create with max items per slice
    pub fn with_max_items(max_items: usize) -> Self {
        Self {
            max_items_per_slice: max_items,
            ..Default::default()
        }
    }

    /// Set max duration
    pub fn max_duration(mut self, ms: f64) -> Self {
        self.max_duration_ms = ms;
        self
    }

    /// Set min slices
    pub fn min_slices(mut self, n: usize) -> Self {
        self.min_slices = n;
        self
    }

    /// Set max slices
    pub fn max_slices(mut self, n: usize) -> Self {
        self.max_slices = n;
        self
    }
}

/// A single kernel slice
#[derive(Debug, Clone)]
pub struct KernelSlice {
    /// Slice ID within the kernel
    pub id: usize,
    /// Starting offset in work items
    pub offset: usize,
    /// Number of work items in this slice
    pub count: usize,
    /// Grid dimensions for this slice
    pub grid: (u32, u32, u32),
    /// Whether this slice has been executed
    pub executed: bool,
    /// Execution time in milliseconds (if executed)
    pub exec_time_ms: Option<f64>,
}

impl KernelSlice {
    /// Create a new slice
    pub fn new(id: usize, offset: usize, count: usize, grid: (u32, u32, u32)) -> Self {
        Self {
            id,
            offset,
            count,
            grid,
            executed: false,
            exec_time_ms: None,
        }
    }

    /// Mark as executed with timing
    pub fn complete(&mut self, exec_time_ms: f64) {
        self.executed = true;
        self.exec_time_ms = Some(exec_time_ms);
    }
}

/// Sliced kernel representation
#[derive(Debug)]
pub struct SlicedKernel {
    /// Kernel handle
    pub kernel_handle: u64,
    /// Block dimensions
    pub block: (u32, u32, u32),
    /// Shared memory per block
    pub shared_mem: u32,
    /// Total work items
    pub total_items: usize,
    /// Slices
    pub slices: Vec<KernelSlice>,
    /// Current slice index
    pub current_slice: usize,
    /// Associated task ID
    pub task_id: Option<String>,
    /// Priority
    pub priority: i32,
}

impl SlicedKernel {
    /// Create a sliced kernel
    pub fn new(
        kernel_handle: u64,
        block: (u32, u32, u32),
        shared_mem: u32,
        total_items: usize,
        slices: Vec<KernelSlice>,
    ) -> Self {
        Self {
            kernel_handle,
            block,
            shared_mem,
            total_items,
            slices,
            current_slice: 0,
            task_id: None,
            priority: 0,
        }
    }

    /// Set task ID
    pub fn with_task(mut self, task_id: String) -> Self {
        self.task_id = Some(task_id);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Get next slice to execute
    pub fn next_slice(&mut self) -> Option<&KernelSlice> {
        if self.current_slice < self.slices.len() {
            let slice = &self.slices[self.current_slice];
            Some(slice)
        } else {
            None
        }
    }

    /// Mark current slice as completed
    pub fn complete_slice(&mut self, exec_time_ms: f64) {
        if self.current_slice < self.slices.len() {
            self.slices[self.current_slice].complete(exec_time_ms);
            self.current_slice += 1;
        }
    }

    /// Check if all slices are executed
    pub fn is_complete(&self) -> bool {
        self.current_slice >= self.slices.len()
    }

    /// Get completion progress (0.0 - 1.0)
    pub fn progress(&self) -> f64 {
        if self.slices.is_empty() {
            1.0
        } else {
            self.current_slice as f64 / self.slices.len() as f64
        }
    }

    /// Get total execution time so far
    pub fn total_exec_time_ms(&self) -> f64 {
        self.slices
            .iter()
            .filter_map(|s| s.exec_time_ms)
            .sum()
    }

    /// Get number of remaining slices
    pub fn remaining_slices(&self) -> usize {
        self.slices.len().saturating_sub(self.current_slice)
    }
}

/// Slice scheduler for interleaving slices across tasks
#[derive(Debug)]
pub struct SliceScheduler {
    config: SliceConfig,
    /// Queue of sliced kernels
    queue: VecDeque<SlicedKernel>,
    /// Statistics
    total_slices: usize,
    completed_slices: usize,
    total_kernels: usize,
    completed_kernels: usize,
}

impl SliceScheduler {
    /// Create a new slice scheduler
    pub fn new(config: SliceConfig) -> Self {
        Self {
            config,
            queue: VecDeque::new(),
            total_slices: 0,
            completed_slices: 0,
            total_kernels: 0,
            completed_kernels: 0,
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(SliceConfig::default())
    }

    /// Slice a kernel and add to queue
    pub fn submit(&mut self, kernel_handle: u64, total_items: usize, block: (u32, u32, u32), shared_mem: u32) -> usize {
        let slices = self.create_slices(total_items, block.0);
        let num_slices = slices.len();

        let sliced = SlicedKernel::new(kernel_handle, block, shared_mem, total_items, slices);
        self.queue.push_back(sliced);
        self.total_slices += num_slices;
        self.total_kernels += 1;

        num_slices
    }

    /// Submit with task and priority
    pub fn submit_for_task(
        &mut self,
        task_id: String,
        kernel_handle: u64,
        total_items: usize,
        block: (u32, u32, u32),
        shared_mem: u32,
        priority: i32,
    ) -> usize {
        let slices = self.create_slices(total_items, block.0);
        let num_slices = slices.len();

        let sliced = SlicedKernel::new(kernel_handle, block, shared_mem, total_items, slices)
            .with_task(task_id)
            .with_priority(priority);

        // Insert sorted by priority (higher first)
        let pos = self.queue.iter().position(|k| k.priority < priority);
        match pos {
            Some(i) => self.queue.insert(i, sliced),
            None => self.queue.push_back(sliced),
        }

        self.total_slices += num_slices;
        self.total_kernels += 1;
        num_slices
    }

    /// Create slices for a kernel
    fn create_slices(&self, total_items: usize, block_x: u32) -> Vec<KernelSlice> {
        let items_per_block = block_x as usize;

        // Calculate number of slices
        let num_slices = (total_items / self.config.max_items_per_slice).max(1);
        let num_slices = num_slices.clamp(self.config.min_slices, self.config.max_slices);

        let items_per_slice = (total_items + num_slices - 1) / num_slices;
        let _blocks_per_slice = (items_per_slice + items_per_block - 1) / items_per_block;

        let mut slices = Vec::new();
        let mut offset = 0;

        for id in 0..num_slices {
            let remaining = total_items.saturating_sub(offset);
            if remaining == 0 {
                break;
            }

            let count = remaining.min(items_per_slice);
            let grid_x = ((count + items_per_block - 1) / items_per_block) as u32;

            slices.push(KernelSlice::new(id, offset, count, (grid_x, 1, 1)));
            offset += count;
        }

        slices
    }

    /// Get next slice to execute (round-robin fair scheduling)
    pub fn get_next_slice(&mut self) -> Option<SliceInfo> {
        if self.queue.is_empty() {
            return None;
        }

        // Rotate to front kernel and get slice
        let kernel = self.queue.front_mut()?;
        let slice = kernel.next_slice()?.clone();

        Some(SliceInfo {
            kernel_handle: kernel.kernel_handle,
            block: kernel.block,
            shared_mem: kernel.shared_mem,
            slice_id: slice.id,
            offset: slice.offset,
            count: slice.count,
            grid: slice.grid,
            task_id: kernel.task_id.clone(),
            priority: kernel.priority,
        })
    }

    /// Complete the current slice of the front kernel
    pub fn complete_slice(&mut self, exec_time_ms: f64) {
        if let Some(kernel) = self.queue.front_mut() {
            kernel.complete_slice(exec_time_ms);
            self.completed_slices += 1;

            // If kernel is complete, remove it and rotate
            if kernel.is_complete() {
                self.queue.pop_front();
                self.completed_kernels += 1;
            } else {
                // Rotate to back for fairness
                if let Some(k) = self.queue.pop_front() {
                    self.queue.push_back(k);
                }
            }
        }
    }

    /// Get number of pending slices
    pub fn pending_slices(&self) -> usize {
        self.queue.iter().map(|k| k.remaining_slices()).sum()
    }

    /// Get number of pending kernels
    pub fn pending_kernels(&self) -> usize {
        self.queue.len()
    }

    /// Get statistics
    pub fn stats(&self) -> SliceStats {
        SliceStats {
            total_slices: self.total_slices,
            completed_slices: self.completed_slices,
            pending_slices: self.pending_slices(),
            total_kernels: self.total_kernels,
            completed_kernels: self.completed_kernels,
            pending_kernels: self.pending_kernels(),
        }
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.queue.clear();
        self.total_slices = 0;
        self.completed_slices = 0;
        self.total_kernels = 0;
        self.completed_kernels = 0;
    }

    /// Get config
    pub fn config(&self) -> &SliceConfig {
        &self.config
    }
}

/// Information about a slice to execute
#[derive(Debug, Clone)]
pub struct SliceInfo {
    /// Kernel handle
    pub kernel_handle: u64,
    /// Block dimensions
    pub block: (u32, u32, u32),
    /// Shared memory
    pub shared_mem: u32,
    /// Slice ID
    pub slice_id: usize,
    /// Offset in work items
    pub offset: usize,
    /// Count of work items
    pub count: usize,
    /// Grid dimensions for this slice
    pub grid: (u32, u32, u32),
    /// Associated task ID
    pub task_id: Option<String>,
    /// Priority
    pub priority: i32,
}

/// Slice scheduler statistics
#[derive(Debug, Clone, Default)]
pub struct SliceStats {
    /// Total slices created
    pub total_slices: usize,
    /// Completed slices
    pub completed_slices: usize,
    /// Pending slices
    pub pending_slices: usize,
    /// Total kernels submitted
    pub total_kernels: usize,
    /// Completed kernels
    pub completed_kernels: usize,
    /// Pending kernels
    pub pending_kernels: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_config() {
        let config = SliceConfig::with_max_items(1024)
            .max_duration(2.0)
            .min_slices(2);

        assert_eq!(config.max_items_per_slice, 1024);
        assert!((config.max_duration_ms - 2.0).abs() < 0.001);
        assert_eq!(config.min_slices, 2);
    }

    #[test]
    fn test_slicing() {
        let config = SliceConfig::with_max_items(1000).min_slices(1).max_slices(10);
        let mut scheduler = SliceScheduler::new(config);

        // Submit kernel with 5000 items
        let num_slices = scheduler.submit(0xDEADBEEF, 5000, (256, 1, 1), 0);

        assert!(num_slices >= 1);
        assert!(num_slices <= 10);

        let stats = scheduler.stats();
        assert_eq!(stats.total_kernels, 1);
        assert_eq!(stats.total_slices, num_slices);
    }

    #[test]
    fn test_slice_execution() {
        let config = SliceConfig::with_max_items(100).min_slices(1);
        let mut scheduler = SliceScheduler::new(config);

        scheduler.submit(0xDEADBEEF, 500, (256, 1, 1), 0);

        // Execute all slices
        while let Some(slice_info) = scheduler.get_next_slice() {
            assert!(slice_info.count > 0);
            scheduler.complete_slice(0.1);
        }

        let stats = scheduler.stats();
        assert_eq!(stats.pending_slices, 0);
        assert_eq!(stats.completed_kernels, 1);
    }

    #[test]
    fn test_fair_scheduling() {
        let config = SliceConfig::with_max_items(100).min_slices(2);
        let mut scheduler = SliceScheduler::new(config);

        // Submit two kernels
        scheduler.submit(0x1111, 200, (256, 1, 1), 0);
        scheduler.submit(0x2222, 200, (256, 1, 1), 0);

        // First two slices should be from different kernels (round-robin)
        let slice1 = scheduler.get_next_slice().unwrap();
        scheduler.complete_slice(0.1);

        let slice2 = scheduler.get_next_slice().unwrap();

        // After completing first kernel's slice, it should rotate to second
        // (depends on implementation details, but should be fair)
        assert!(slice1.kernel_handle != slice2.kernel_handle || scheduler.pending_kernels() == 1);
    }

    #[test]
    fn test_priority_scheduling() {
        let config = SliceConfig::with_max_items(1000);
        let mut scheduler = SliceScheduler::new(config);

        // Submit low priority first
        scheduler.submit_for_task("low".into(), 0x1111, 1000, (256, 1, 1), 0, 0);
        // Submit high priority second
        scheduler.submit_for_task("high".into(), 0x2222, 1000, (256, 1, 1), 0, 100);

        // High priority should come first
        let slice = scheduler.get_next_slice().unwrap();
        assert_eq!(slice.priority, 100);
        assert_eq!(slice.task_id, Some("high".into()));
    }
}
