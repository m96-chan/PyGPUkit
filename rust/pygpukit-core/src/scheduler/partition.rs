//! GPU Partitioning
//!
//! Provides logical partitioning of GPU resources across tasks or tenants.
//! Supports partitioning of:
//! - Compute resources (SM units)
//! - Memory quota
//! - Bandwidth allocation
//! - Stream capacity

use std::collections::HashMap;

/// Partition resource limits
#[derive(Debug, Clone)]
pub struct PartitionLimits {
    /// Memory quota in bytes (0 = unlimited)
    pub memory_quota: usize,
    /// Compute share (0.0 - 1.0, fraction of GPU)
    pub compute_share: f64,
    /// Bandwidth share (0.0 - 1.0)
    pub bandwidth_share: f64,
    /// Maximum concurrent streams
    pub max_streams: usize,
    /// Maximum pending kernels
    pub max_pending_kernels: usize,
    /// Maximum pending transfers
    pub max_pending_transfers: usize,
}

impl Default for PartitionLimits {
    fn default() -> Self {
        Self {
            memory_quota: 0, // Unlimited
            compute_share: 1.0,
            bandwidth_share: 1.0,
            max_streams: 16,
            max_pending_kernels: 1024,
            max_pending_transfers: 256,
        }
    }
}

impl PartitionLimits {
    /// Create with memory quota
    pub fn with_memory(memory_quota: usize) -> Self {
        Self {
            memory_quota,
            ..Default::default()
        }
    }

    /// Create with compute share
    pub fn with_compute(compute_share: f64) -> Self {
        Self {
            compute_share: compute_share.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Set memory quota
    pub fn memory(mut self, quota: usize) -> Self {
        self.memory_quota = quota;
        self
    }

    /// Set compute share
    pub fn compute(mut self, share: f64) -> Self {
        self.compute_share = share.clamp(0.0, 1.0);
        self
    }

    /// Set bandwidth share
    pub fn bandwidth(mut self, share: f64) -> Self {
        self.bandwidth_share = share.clamp(0.0, 1.0);
        self
    }

    /// Set max streams
    pub fn streams(mut self, max: usize) -> Self {
        self.max_streams = max;
        self
    }
}

/// Partition resource usage
#[derive(Debug, Clone, Default)]
pub struct PartitionUsage {
    /// Current memory usage in bytes
    pub memory_used: usize,
    /// Active streams
    pub active_streams: usize,
    /// Pending kernels
    pub pending_kernels: usize,
    /// Pending transfers
    pub pending_transfers: usize,
    /// Total kernels executed
    pub total_kernels: usize,
    /// Total transfers completed
    pub total_transfers: usize,
    /// Compute time in milliseconds
    pub compute_time_ms: f64,
}

impl PartitionUsage {
    /// Check if memory would exceed quota
    pub fn would_exceed_memory(&self, limits: &PartitionLimits, size: usize) -> bool {
        limits.memory_quota > 0 && self.memory_used + size > limits.memory_quota
    }

    /// Check if stream limit reached
    pub fn stream_limit_reached(&self, limits: &PartitionLimits) -> bool {
        self.active_streams >= limits.max_streams
    }

    /// Check if kernel limit reached
    pub fn kernel_limit_reached(&self, limits: &PartitionLimits) -> bool {
        self.pending_kernels >= limits.max_pending_kernels
    }

    /// Check if transfer limit reached
    pub fn transfer_limit_reached(&self, limits: &PartitionLimits) -> bool {
        self.pending_transfers >= limits.max_pending_transfers
    }
}

/// A GPU partition
#[derive(Debug, Clone)]
pub struct Partition {
    /// Partition ID
    pub id: String,
    /// Partition name
    pub name: String,
    /// Resource limits
    pub limits: PartitionLimits,
    /// Current usage
    pub usage: PartitionUsage,
    /// Associated task IDs
    pub tasks: Vec<String>,
    /// Creation timestamp
    pub created_at: f64,
    /// Whether partition is enabled
    pub enabled: bool,
}

impl Partition {
    /// Create a new partition
    pub fn new(id: String, name: String, limits: PartitionLimits) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        Self {
            id,
            name,
            limits,
            usage: PartitionUsage::default(),
            tasks: Vec::new(),
            created_at: now,
            enabled: true,
        }
    }

    /// Check if memory allocation is allowed
    pub fn can_allocate_memory(&self, size: usize) -> bool {
        !self.usage.would_exceed_memory(&self.limits, size)
    }

    /// Check if stream creation is allowed
    pub fn can_create_stream(&self) -> bool {
        !self.usage.stream_limit_reached(&self.limits)
    }

    /// Check if kernel submission is allowed
    pub fn can_submit_kernel(&self) -> bool {
        !self.usage.kernel_limit_reached(&self.limits)
    }

    /// Check if transfer submission is allowed
    pub fn can_submit_transfer(&self) -> bool {
        !self.usage.transfer_limit_reached(&self.limits)
    }

    /// Allocate memory
    pub fn allocate_memory(&mut self, size: usize) -> bool {
        if self.can_allocate_memory(size) {
            self.usage.memory_used += size;
            true
        } else {
            false
        }
    }

    /// Free memory
    pub fn free_memory(&mut self, size: usize) {
        self.usage.memory_used = self.usage.memory_used.saturating_sub(size);
    }

    /// Register a stream
    pub fn register_stream(&mut self) -> bool {
        if self.can_create_stream() {
            self.usage.active_streams += 1;
            true
        } else {
            false
        }
    }

    /// Release a stream
    pub fn release_stream(&mut self) {
        self.usage.active_streams = self.usage.active_streams.saturating_sub(1);
    }

    /// Submit a kernel
    pub fn submit_kernel(&mut self) -> bool {
        if self.can_submit_kernel() {
            self.usage.pending_kernels += 1;
            true
        } else {
            false
        }
    }

    /// Complete a kernel
    pub fn complete_kernel(&mut self, exec_time_ms: f64) {
        self.usage.pending_kernels = self.usage.pending_kernels.saturating_sub(1);
        self.usage.total_kernels += 1;
        self.usage.compute_time_ms += exec_time_ms;
    }

    /// Submit a transfer
    pub fn submit_transfer(&mut self) -> bool {
        if self.can_submit_transfer() {
            self.usage.pending_transfers += 1;
            true
        } else {
            false
        }
    }

    /// Complete a transfer
    pub fn complete_transfer(&mut self) {
        self.usage.pending_transfers = self.usage.pending_transfers.saturating_sub(1);
        self.usage.total_transfers += 1;
    }

    /// Add a task to this partition
    pub fn add_task(&mut self, task_id: String) {
        if !self.tasks.contains(&task_id) {
            self.tasks.push(task_id);
        }
    }

    /// Remove a task from this partition
    pub fn remove_task(&mut self, task_id: &str) {
        self.tasks.retain(|t| t != task_id);
    }

    /// Check if task belongs to this partition
    pub fn has_task(&self, task_id: &str) -> bool {
        self.tasks.iter().any(|t| t == task_id)
    }

    /// Get memory utilization (0.0 - 1.0)
    pub fn memory_utilization(&self) -> f64 {
        if self.limits.memory_quota > 0 {
            self.usage.memory_used as f64 / self.limits.memory_quota as f64
        } else {
            0.0
        }
    }
}

/// Partition manager configuration
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Total GPU memory available for partitioning
    pub total_memory: usize,
    /// Total compute capacity (normalized to 1.0)
    pub total_compute: f64,
    /// Total bandwidth capacity (normalized to 1.0)
    pub total_bandwidth: f64,
    /// Allow overcommit
    pub allow_overcommit: bool,
    /// Overcommit ratio (1.0 = no overcommit)
    pub overcommit_ratio: f64,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            total_memory: 8 * 1024 * 1024 * 1024, // 8GB default
            total_compute: 1.0,
            total_bandwidth: 1.0,
            allow_overcommit: false,
            overcommit_ratio: 1.0,
        }
    }
}

impl PartitionConfig {
    /// Create with total memory
    pub fn with_memory(total_memory: usize) -> Self {
        Self {
            total_memory,
            ..Default::default()
        }
    }

    /// Enable overcommit
    pub fn overcommit(mut self, ratio: f64) -> Self {
        self.allow_overcommit = true;
        self.overcommit_ratio = ratio;
        self
    }
}

/// Partition manager statistics
#[derive(Debug, Clone, Default)]
pub struct PartitionStats {
    /// Total partitions
    pub partition_count: usize,
    /// Active partitions
    pub active_partitions: usize,
    /// Total memory allocated across partitions
    pub total_memory_allocated: usize,
    /// Total compute share allocated
    pub total_compute_allocated: f64,
    /// Total bandwidth allocated
    pub total_bandwidth_allocated: f64,
    /// Available memory
    pub available_memory: usize,
    /// Available compute
    pub available_compute: f64,
    /// Available bandwidth
    pub available_bandwidth: f64,
}

/// Partition manager
///
/// Manages GPU resource partitions for multi-tenant or multi-task isolation.
#[derive(Debug)]
pub struct PartitionManager {
    config: PartitionConfig,
    /// Partitions by ID
    partitions: HashMap<String, Partition>,
    /// Task to partition mapping
    task_partition: HashMap<String, String>,
    /// Default partition ID
    default_partition: Option<String>,
}

impl PartitionManager {
    /// Create a new partition manager
    pub fn new(config: PartitionConfig) -> Self {
        Self {
            config,
            partitions: HashMap::new(),
            task_partition: HashMap::new(),
            default_partition: None,
        }
    }

    /// Create with defaults
    pub fn with_defaults() -> Self {
        Self::new(PartitionConfig::default())
    }

    /// Create with total memory
    pub fn with_memory(total_memory: usize) -> Self {
        Self::new(PartitionConfig::with_memory(total_memory))
    }

    /// Create a new partition
    pub fn create_partition(&mut self, id: &str, name: &str, limits: PartitionLimits) -> Result<(), PartitionError> {
        if self.partitions.contains_key(id) {
            return Err(PartitionError::AlreadyExists { id: id.into() });
        }

        // Validate limits
        self.validate_limits(&limits)?;

        let partition = Partition::new(id.into(), name.into(), limits);
        self.partitions.insert(id.into(), partition);

        // Set as default if first
        if self.default_partition.is_none() {
            self.default_partition = Some(id.into());
        }

        Ok(())
    }

    /// Validate partition limits against available resources
    fn validate_limits(&self, limits: &PartitionLimits) -> Result<(), PartitionError> {
        let stats = self.stats();
        // Check memory
        if limits.memory_quota > 0 && limits.memory_quota > stats.available_memory && !self.config.allow_overcommit {
            return Err(PartitionError::InsufficientResources {
                resource: "memory".into(),
                requested: limits.memory_quota,
                available: stats.available_memory,
            });
        }

        // Check compute
        if limits.compute_share > stats.available_compute && !self.config.allow_overcommit {
            return Err(PartitionError::InsufficientResources {
                resource: "compute".into(),
                requested: (limits.compute_share * 100.0) as usize,
                available: (stats.available_compute * 100.0) as usize,
            });
        }

        // Check bandwidth
        if limits.bandwidth_share > stats.available_bandwidth && !self.config.allow_overcommit {
            return Err(PartitionError::InsufficientResources {
                resource: "bandwidth".into(),
                requested: (limits.bandwidth_share * 100.0) as usize,
                available: (stats.available_bandwidth * 100.0) as usize,
            });
        }

        Ok(())
    }

    /// Delete a partition
    pub fn delete_partition(&mut self, id: &str) -> Result<Partition, PartitionError> {
        let partition = self.partitions.remove(id)
            .ok_or(PartitionError::NotFound { id: id.into() })?;

        // Remove task mappings
        self.task_partition.retain(|_, p| p != id);

        // Update default
        if self.default_partition.as_deref() == Some(id) {
            self.default_partition = self.partitions.keys().next().cloned();
        }

        Ok(partition)
    }

    /// Get a partition
    pub fn get(&self, id: &str) -> Option<&Partition> {
        self.partitions.get(id)
    }

    /// Get a mutable partition
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Partition> {
        self.partitions.get_mut(id)
    }

    /// Assign a task to a partition
    pub fn assign_task(&mut self, task_id: &str, partition_id: &str) -> Result<(), PartitionError> {
        if !self.partitions.contains_key(partition_id) {
            return Err(PartitionError::NotFound { id: partition_id.into() });
        }

        // Remove from old partition if any
        if let Some(old_id) = self.task_partition.get(task_id).cloned() {
            if let Some(old_partition) = self.partitions.get_mut(&old_id) {
                old_partition.remove_task(task_id);
            }
        }

        // Add to new partition
        self.task_partition.insert(task_id.into(), partition_id.into());
        if let Some(partition) = self.partitions.get_mut(partition_id) {
            partition.add_task(task_id.into());
        }

        Ok(())
    }

    /// Get partition for a task
    pub fn get_task_partition(&self, task_id: &str) -> Option<&Partition> {
        let partition_id = self.task_partition.get(task_id)
            .or(self.default_partition.as_ref())?;
        self.partitions.get(partition_id)
    }

    /// Get mutable partition for a task
    pub fn get_task_partition_mut(&mut self, task_id: &str) -> Option<&mut Partition> {
        let partition_id = self.task_partition.get(task_id)
            .or(self.default_partition.as_ref())
            .cloned();
        partition_id.and_then(|id| self.partitions.get_mut(&id))
    }

    /// Unassign a task from its partition
    pub fn unassign_task(&mut self, task_id: &str) {
        if let Some(partition_id) = self.task_partition.remove(task_id) {
            if let Some(partition) = self.partitions.get_mut(&partition_id) {
                partition.remove_task(task_id);
            }
        }
    }

    /// Set default partition
    pub fn set_default(&mut self, id: &str) -> Result<(), PartitionError> {
        if !self.partitions.contains_key(id) {
            return Err(PartitionError::NotFound { id: id.into() });
        }
        self.default_partition = Some(id.into());
        Ok(())
    }

    /// Get default partition
    pub fn default_partition(&self) -> Option<&Partition> {
        self.default_partition.as_ref().and_then(|id| self.partitions.get(id))
    }

    /// List all partition IDs
    pub fn partition_ids(&self) -> Vec<&str> {
        self.partitions.keys().map(|s| s.as_str()).collect()
    }

    /// Get statistics
    pub fn stats(&self) -> PartitionStats {
        let mut total_memory = 0;
        let mut total_compute = 0.0;
        let mut total_bandwidth = 0.0;
        let mut active = 0;

        for partition in self.partitions.values() {
            if partition.enabled {
                active += 1;
                total_memory += partition.limits.memory_quota;
                total_compute += partition.limits.compute_share;
                total_bandwidth += partition.limits.bandwidth_share;
            }
        }

        let effective_total = if self.config.allow_overcommit {
            (self.config.total_memory as f64 * self.config.overcommit_ratio) as usize
        } else {
            self.config.total_memory
        };

        PartitionStats {
            partition_count: self.partitions.len(),
            active_partitions: active,
            total_memory_allocated: total_memory,
            total_compute_allocated: total_compute,
            total_bandwidth_allocated: total_bandwidth,
            available_memory: effective_total.saturating_sub(total_memory),
            available_compute: (self.config.total_compute - total_compute).max(0.0),
            available_bandwidth: (self.config.total_bandwidth - total_bandwidth).max(0.0),
        }
    }

    /// Clear all partitions
    pub fn clear(&mut self) {
        self.partitions.clear();
        self.task_partition.clear();
        self.default_partition = None;
    }

    /// Get config
    pub fn config(&self) -> &PartitionConfig {
        &self.config
    }
}

/// Partition errors
#[derive(Debug, Clone)]
pub enum PartitionError {
    /// Partition already exists
    AlreadyExists { id: String },
    /// Partition not found
    NotFound { id: String },
    /// Insufficient resources
    InsufficientResources {
        resource: String,
        requested: usize,
        available: usize,
    },
    /// Operation not allowed
    NotAllowed { reason: String },
}

impl std::fmt::Display for PartitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PartitionError::AlreadyExists { id } => {
                write!(f, "Partition '{}' already exists", id)
            }
            PartitionError::NotFound { id } => {
                write!(f, "Partition '{}' not found", id)
            }
            PartitionError::InsufficientResources { resource, requested, available } => {
                write!(f, "Insufficient {}: requested {}, available {}", resource, requested, available)
            }
            PartitionError::NotAllowed { reason } => {
                write!(f, "Operation not allowed: {}", reason)
            }
        }
    }
}

impl std::error::Error for PartitionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_limits() {
        let limits = PartitionLimits::with_memory(1024 * 1024 * 1024)
            .compute(0.5)
            .bandwidth(0.3);

        assert_eq!(limits.memory_quota, 1024 * 1024 * 1024);
        assert!((limits.compute_share - 0.5).abs() < 0.001);
        assert!((limits.bandwidth_share - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_partition_creation() {
        let mut manager = PartitionManager::with_defaults();

        let limits = PartitionLimits::with_memory(1024 * 1024 * 1024);
        manager.create_partition("p1", "Partition 1", limits).unwrap();

        assert!(manager.get("p1").is_some());
        assert_eq!(manager.partition_ids().len(), 1);
    }

    #[test]
    fn test_duplicate_partition() {
        let mut manager = PartitionManager::with_defaults();

        let limits = PartitionLimits::default();
        manager.create_partition("p1", "Partition 1", limits.clone()).unwrap();

        let result = manager.create_partition("p1", "Duplicate", limits);
        assert!(matches!(result, Err(PartitionError::AlreadyExists { .. })));
    }

    #[test]
    fn test_task_assignment() {
        let mut manager = PartitionManager::with_defaults();

        let limits = PartitionLimits::default();
        manager.create_partition("p1", "Partition 1", limits).unwrap();

        manager.assign_task("task-1", "p1").unwrap();

        let partition = manager.get_task_partition("task-1");
        assert!(partition.is_some());
        assert_eq!(partition.unwrap().id, "p1");
    }

    #[test]
    fn test_default_partition() {
        let mut manager = PartitionManager::with_defaults();

        let limits = PartitionLimits::default();
        manager.create_partition("default", "Default", limits).unwrap();

        // Unassigned task should use default
        let partition = manager.get_task_partition("unknown-task");
        assert!(partition.is_some());
        assert_eq!(partition.unwrap().id, "default");
    }

    #[test]
    fn test_memory_allocation() {
        let mut manager = PartitionManager::with_defaults();

        let limits = PartitionLimits::with_memory(1000);
        manager.create_partition("p1", "Partition 1", limits).unwrap();

        let partition = manager.get_mut("p1").unwrap();

        // Should succeed
        assert!(partition.allocate_memory(500));
        assert_eq!(partition.usage.memory_used, 500);

        // Should succeed
        assert!(partition.allocate_memory(400));
        assert_eq!(partition.usage.memory_used, 900);

        // Should fail (exceeds quota)
        assert!(!partition.allocate_memory(200));
        assert_eq!(partition.usage.memory_used, 900);

        // Free and try again
        partition.free_memory(500);
        assert!(partition.allocate_memory(200));
    }

    #[test]
    fn test_kernel_submission() {
        let mut manager = PartitionManager::with_defaults();

        let limits = PartitionLimits::default().streams(2);
        manager.create_partition("p1", "Partition 1", limits).unwrap();

        let partition = manager.get_mut("p1").unwrap();
        partition.limits.max_pending_kernels = 3;

        assert!(partition.submit_kernel());
        assert!(partition.submit_kernel());
        assert!(partition.submit_kernel());
        assert!(!partition.submit_kernel()); // Limit reached

        partition.complete_kernel(1.0);
        assert!(partition.submit_kernel());
        assert_eq!(partition.usage.total_kernels, 1);
    }

    #[test]
    fn test_stream_limits() {
        let mut manager = PartitionManager::with_defaults();

        let limits = PartitionLimits::default().streams(2);
        manager.create_partition("p1", "Partition 1", limits).unwrap();

        let partition = manager.get_mut("p1").unwrap();

        assert!(partition.register_stream());
        assert!(partition.register_stream());
        assert!(!partition.register_stream()); // Limit

        partition.release_stream();
        assert!(partition.register_stream());
    }

    #[test]
    fn test_stats() {
        let mut manager = PartitionManager::with_memory(10_000_000_000); // 10GB

        let limits1 = PartitionLimits::with_memory(1_000_000_000).compute(0.3).bandwidth(0.3);
        let limits2 = PartitionLimits::with_memory(2_000_000_000).compute(0.4).bandwidth(0.4);

        manager.create_partition("p1", "P1", limits1).unwrap();
        manager.create_partition("p2", "P2", limits2).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.partition_count, 2);
        assert_eq!(stats.total_memory_allocated, 3_000_000_000);
        assert!((stats.total_compute_allocated - 0.7).abs() < 0.001);
        assert_eq!(stats.available_memory, 7_000_000_000);
    }

    #[test]
    fn test_delete_partition() {
        let mut manager = PartitionManager::with_defaults();

        // Use partial limits so both partitions can be created
        let limits1 = PartitionLimits::with_compute(0.4).bandwidth(0.4);
        let limits2 = PartitionLimits::with_compute(0.4).bandwidth(0.4);
        manager.create_partition("p1", "P1", limits1).unwrap();
        manager.create_partition("p2", "P2", limits2).unwrap();

        manager.assign_task("task-1", "p1").unwrap();

        // Delete p1
        manager.delete_partition("p1").unwrap();

        assert!(manager.get("p1").is_none());
        assert!(manager.get_task_partition("task-1").is_none() ||
                manager.get_task_partition("task-1").unwrap().id != "p1");
    }

    #[test]
    fn test_reassign_task() {
        let mut manager = PartitionManager::with_defaults();

        // Use partial limits so both partitions can be created
        let limits1 = PartitionLimits::with_compute(0.4).bandwidth(0.4);
        let limits2 = PartitionLimits::with_compute(0.4).bandwidth(0.4);
        manager.create_partition("p1", "P1", limits1).unwrap();
        manager.create_partition("p2", "P2", limits2).unwrap();

        manager.assign_task("task-1", "p1").unwrap();
        assert!(manager.get("p1").unwrap().has_task("task-1"));

        manager.assign_task("task-1", "p2").unwrap();
        assert!(!manager.get("p1").unwrap().has_task("task-1"));
        assert!(manager.get("p2").unwrap().has_task("task-1"));
    }
}
