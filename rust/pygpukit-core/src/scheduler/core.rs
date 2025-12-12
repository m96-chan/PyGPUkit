//! Scheduler core implementation
//!
//! Provides task scheduling with bandwidth pacing and memory tracking.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use crate::scheduler::task::{TaskMeta, TaskState, TaskStats};
use crate::scheduler::admission::{AdmissionController, AdmissionConfig, AdmissionDecision};

/// Scheduler statistics
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total tasks submitted
    pub total_submitted: usize,
    /// Tasks currently pending
    pub pending_count: usize,
    /// Tasks currently running
    pub running_count: usize,
    /// Tasks completed successfully
    pub completed_count: usize,
    /// Tasks that failed
    pub failed_count: usize,
    /// Tasks cancelled
    pub cancelled_count: usize,
    /// Total memory reserved by running tasks
    pub reserved_memory: usize,
    /// Available memory (total - reserved)
    pub available_memory: usize,
    /// Average wait time (seconds)
    pub avg_wait_time: f64,
    /// Average execution time (seconds)
    pub avg_exec_time: f64,
}

/// Internal scheduler state
struct SchedulerInner {
    /// All tasks by ID
    tasks: HashMap<String, TaskMeta>,
    /// Pending task queue (FIFO order)
    pending_queue: VecDeque<String>,
    /// Running task IDs
    running: Vec<String>,
    /// Memory reserved by running tasks
    reserved_memory: usize,
    /// Statistics tracking
    total_wait_time: f64,
    total_exec_time: f64,
    completed_count: usize,
    /// Admission controller
    admission: AdmissionController,
}

/// Thread-safe task scheduler with bandwidth pacing.
///
/// # Example
///
/// ```
/// use pygpukit_core::scheduler::{Scheduler, TaskMeta};
///
/// let scheduler = Scheduler::new(Some(1024 * 1024 * 100), 10.0, 100.0);
/// let task = TaskMeta::with_memory("task-1".into(), "Compute".into(), 1024);
/// scheduler.submit(task);
///
/// let runnable = scheduler.get_runnable_tasks(1);
/// ```
pub struct Scheduler {
    /// Total memory available for scheduling
    total_memory: Option<usize>,
    /// Scheduling tick interval (ms)
    sched_tick_ms: f64,
    /// Bandwidth window (ms)
    window_ms: f64,
    /// Internal state
    inner: RwLock<SchedulerInner>,
}

impl Scheduler {
    /// Create a new scheduler.
    ///
    /// # Arguments
    ///
    /// * `total_memory` - Total GPU memory available (None for unlimited)
    /// * `sched_tick_ms` - Scheduling tick interval in milliseconds
    /// * `window_ms` - Bandwidth pacing window in milliseconds
    pub fn new(total_memory: Option<usize>, sched_tick_ms: f64, window_ms: f64) -> Self {
        let admission_config = match total_memory {
            Some(mem) => AdmissionConfig::with_memory(mem),
            None => AdmissionConfig::default(),
        };

        Self {
            total_memory,
            sched_tick_ms,
            window_ms,
            inner: RwLock::new(SchedulerInner {
                tasks: HashMap::new(),
                pending_queue: VecDeque::new(),
                running: Vec::new(),
                reserved_memory: 0,
                total_wait_time: 0.0,
                total_exec_time: 0.0,
                completed_count: 0,
                admission: AdmissionController::new(admission_config),
            }),
        }
    }

    /// Evaluate admission for a task without submitting it.
    ///
    /// This performs a dry-run admission check to determine if
    /// a task would be admitted, queued, or rejected.
    pub fn evaluate_admission(&self, task: &TaskMeta) -> AdmissionDecision {
        let inner = self.inner.read();
        inner.admission.evaluate(task)
    }

    /// Admit a task through the admission control pipeline.
    ///
    /// Returns an AdmissionDecision indicating whether the task
    /// was admitted, queued for best-effort, or rejected.
    ///
    /// If admitted or queued, the task is automatically submitted.
    pub fn admit(&self, task: TaskMeta) -> AdmissionDecision {
        let mut inner = self.inner.write();
        let decision = inner.admission.admit(&task);

        match &decision {
            AdmissionDecision::Admit { reserved_memory, .. } => {
                // Task admitted - add to scheduler
                let task_id = task.id.clone();
                inner.pending_queue.push_back(task_id.clone());
                inner.reserved_memory += reserved_memory;
                inner.tasks.insert(task_id, task);
            }
            AdmissionDecision::Queue { .. } => {
                // Task queued for best-effort - still add to scheduler
                let task_id = task.id.clone();
                let memory = task.memory_estimate;
                inner.pending_queue.push_back(task_id.clone());
                inner.reserved_memory += memory;
                inner.tasks.insert(task_id, task);
            }
            AdmissionDecision::Reject { .. } => {
                // Task rejected - do not add
            }
        }

        decision
    }

    /// Submit a task for scheduling.
    ///
    /// Memory is reserved immediately upon submission to ensure
    /// consistent resource tracking across pending and running states.
    pub fn submit(&self, task: TaskMeta) -> String {
        let task_id = task.id.clone();
        let memory = task.memory_estimate;
        let mut inner = self.inner.write();
        inner.pending_queue.push_back(task_id.clone());
        inner.reserved_memory += memory;  // Reserve memory at submit time
        inner.tasks.insert(task_id.clone(), task);
        task_id
    }

    /// Get tasks that are ready to run.
    ///
    /// Returns up to `max_tasks` task IDs that can be started.
    /// Note: Memory is already reserved at submit time, so no memory check needed here.
    pub fn get_runnable_tasks(&self, max_tasks: usize) -> Vec<String> {
        let mut inner = self.inner.write();
        let mut runnable = Vec::new();
        let mut to_remove = Vec::new();

        for (idx, task_id) in inner.pending_queue.iter().enumerate() {
            if runnable.len() >= max_tasks {
                break;
            }

            if let Some(task) = inner.tasks.get(task_id) {
                // Check dependencies
                let deps_satisfied = task.dependencies.iter().all(|dep_id| {
                    inner.tasks.get(dep_id)
                        .map(|t| t.is_terminal())
                        .unwrap_or(true)
                });

                if !deps_satisfied {
                    continue;
                }

                // Memory was already reserved at submit time, no need to check here
                runnable.push(task_id.clone());
                to_remove.push(idx);
            }
        }

        // Remove from pending queue (reverse order to maintain indices)
        for idx in to_remove.into_iter().rev() {
            inner.pending_queue.remove(idx);
        }

        // Start tasks (memory already reserved at submit time)
        for task_id in &runnable {
            if let Some(task) = inner.tasks.get_mut(task_id) {
                task.start();
            }
            inner.running.push(task_id.clone());
        }

        runnable
    }

    /// Check if a specific task should run now.
    pub fn should_run(&self, task_id: &str) -> bool {
        let inner = self.inner.read();

        if let Some(task) = inner.tasks.get(task_id) {
            if task.state != TaskState::Pending {
                return false;
            }

            // Check dependencies
            let deps_satisfied = task.dependencies.iter().all(|dep_id| {
                inner.tasks.get(dep_id)
                    .map(|t| t.is_terminal())
                    .unwrap_or(true)
            });

            if !deps_satisfied {
                return false;
            }

            // Check memory
            if let Some(total) = self.total_memory {
                if inner.reserved_memory + task.memory_estimate > total {
                    return false;
                }
            }

            true
        } else {
            false
        }
    }

    /// Mark a task as started.
    pub fn start_task(&self, task_id: &str) -> bool {
        let mut inner = self.inner.write();

        if let Some(task) = inner.tasks.get_mut(task_id) {
            if task.state == TaskState::Pending {
                task.start();
                // Memory was already reserved at submit time, don't add again
                inner.running.push(task_id.to_string());

                // Remove from pending queue
                inner.pending_queue.retain(|id| id != task_id);
                return true;
            }
        }
        false
    }

    /// Mark a task as completed successfully.
    pub fn complete_task(&self, task_id: &str) -> bool {
        let mut inner = self.inner.write();

        // Get task info first to avoid borrow issues
        let task_info = inner.tasks.get(task_id).and_then(|task| {
            if task.state == TaskState::Running {
                let wait_time = task.started_at.unwrap_or(task.submitted_at) - task.submitted_at;
                Some((task.memory_estimate, wait_time))
            } else {
                None
            }
        });

        if let Some((memory_estimate, wait_time)) = task_info {
            if let Some(task) = inner.tasks.get_mut(task_id) {
                task.complete();
                let exec_time = task.duration().unwrap_or(0.0);
                inner.total_exec_time += exec_time;
            }
            inner.reserved_memory = inner.reserved_memory.saturating_sub(memory_estimate);
            inner.running.retain(|id| id != task_id);
            inner.total_wait_time += wait_time;
            inner.completed_count += 1;
            return true;
        }
        false
    }

    /// Mark a task as failed.
    pub fn fail_task(&self, task_id: &str, error: String) -> bool {
        let mut inner = self.inner.write();

        // Get task state and memory estimate first to avoid borrow issues
        let task_info = inner.tasks.get(task_id).and_then(|task| {
            if task.state == TaskState::Running || task.state == TaskState::Pending {
                Some((task.state, task.memory_estimate))
            } else {
                None
            }
        });

        if let Some((state, memory_estimate)) = task_info {
            if let Some(task) = inner.tasks.get_mut(task_id) {
                task.fail(error);
            }
            // Release memory (reserved at submit time)
            inner.reserved_memory = inner.reserved_memory.saturating_sub(memory_estimate);
            if state == TaskState::Running {
                inner.running.retain(|id| id != task_id);
            } else {
                inner.pending_queue.retain(|id| id != task_id);
            }
            return true;
        }
        false
    }

    /// Cancel a task.
    pub fn cancel_task(&self, task_id: &str) -> bool {
        let mut inner = self.inner.write();

        // Get task state and memory info first to avoid borrow issues
        let task_info = inner.tasks.get(task_id).and_then(|task| {
            if !task.is_terminal() {
                Some((task.state, task.memory_estimate))
            } else {
                None
            }
        });

        if let Some((state, memory_estimate)) = task_info {
            // Release memory (reserved at submit time)
            inner.reserved_memory = inner.reserved_memory.saturating_sub(memory_estimate);
            if state == TaskState::Running {
                inner.running.retain(|id| id != task_id);
            } else {
                inner.pending_queue.retain(|id| id != task_id);
            }
            if let Some(task) = inner.tasks.get_mut(task_id) {
                task.cancel();
            }
            return true;
        }
        false
    }

    /// Get task by ID.
    pub fn get_task(&self, task_id: &str) -> Option<TaskMeta> {
        self.inner.read().tasks.get(task_id).cloned()
    }

    /// Get task state.
    pub fn get_task_state(&self, task_id: &str) -> Option<TaskState> {
        self.inner.read().tasks.get(task_id).map(|t| t.state)
    }

    /// Get scheduler statistics.
    pub fn stats(&self) -> SchedulerStats {
        let inner = self.inner.read();

        let pending_count = inner.pending_queue.len();
        let running_count = inner.running.len();
        let failed_count = inner.tasks.values()
            .filter(|t| t.state == TaskState::Failed)
            .count();
        let cancelled_count = inner.tasks.values()
            .filter(|t| t.state == TaskState::Cancelled)
            .count();

        let completed = inner.completed_count;
        let avg_wait = if completed > 0 {
            inner.total_wait_time / completed as f64
        } else {
            0.0
        };
        let avg_exec = if completed > 0 {
            inner.total_exec_time / completed as f64
        } else {
            0.0
        };

        SchedulerStats {
            total_submitted: inner.tasks.len(),
            pending_count,
            running_count,
            completed_count: completed,
            failed_count,
            cancelled_count,
            reserved_memory: inner.reserved_memory,
            available_memory: self.total_memory
                .map(|t| t.saturating_sub(inner.reserved_memory))
                .unwrap_or(usize::MAX),
            avg_wait_time: avg_wait,
            avg_exec_time: avg_exec,
        }
    }

    /// Get individual task statistics.
    pub fn task_stats(&self, task_id: &str) -> Option<TaskStats> {
        let inner = self.inner.read();
        let task = inner.tasks.get(task_id)?;

        let wait_time = task.started_at
            .map(|s| s - task.submitted_at)
            .unwrap_or_else(|| Self::now() - task.submitted_at);

        let exec_time = task.duration().unwrap_or(0.0);

        Some(TaskStats {
            id: task.id.clone(),
            name: task.name.clone(),
            state: task.state,
            wait_time,
            exec_time,
            memory_used: task.memory_estimate,
        })
    }

    /// Clear all tasks.
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.tasks.clear();
        inner.pending_queue.clear();
        inner.running.clear();
        inner.reserved_memory = 0;
        inner.total_wait_time = 0.0;
        inner.total_exec_time = 0.0;
        inner.completed_count = 0;
        inner.admission.reset();
    }

    /// Get admission control statistics.
    pub fn admission_stats(&self) -> crate::scheduler::admission::AdmissionStats {
        self.inner.read().admission.stats()
    }

    /// Get scheduling tick interval.
    #[inline]
    pub fn sched_tick_ms(&self) -> f64 {
        self.sched_tick_ms
    }

    /// Get bandwidth window.
    #[inline]
    pub fn window_ms(&self) -> f64 {
        self.window_ms
    }

    /// Get total memory.
    #[inline]
    pub fn total_memory(&self) -> Option<usize> {
        self.total_memory
    }

    /// Get current Unix timestamp.
    #[inline]
    fn now() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

// Thread-safe
unsafe impl Send for Scheduler {}
unsafe impl Sync for Scheduler {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let sched = Scheduler::new(Some(1024 * 1024), 10.0, 100.0);
        assert_eq!(sched.total_memory(), Some(1024 * 1024));
        assert_eq!(sched.sched_tick_ms(), 10.0);
    }

    #[test]
    fn test_submit_and_run() {
        let sched = Scheduler::new(None, 10.0, 100.0);

        let task = TaskMeta::new("task-1".into(), "Test".into());
        sched.submit(task);

        let runnable = sched.get_runnable_tasks(10);
        assert_eq!(runnable.len(), 1);
        assert_eq!(runnable[0], "task-1");

        let state = sched.get_task_state("task-1");
        assert_eq!(state, Some(TaskState::Running));
    }

    #[test]
    fn test_complete_task() {
        let sched = Scheduler::new(None, 10.0, 100.0);

        let task = TaskMeta::new("task-1".into(), "Test".into());
        sched.submit(task);
        sched.get_runnable_tasks(1);

        assert!(sched.complete_task("task-1"));
        assert_eq!(sched.get_task_state("task-1"), Some(TaskState::Completed));

        let stats = sched.stats();
        assert_eq!(stats.completed_count, 1);
    }

    #[test]
    fn test_fail_task() {
        let sched = Scheduler::new(None, 10.0, 100.0);

        let task = TaskMeta::new("task-1".into(), "Test".into());
        sched.submit(task);
        sched.get_runnable_tasks(1);

        assert!(sched.fail_task("task-1", "Out of memory".into()));

        let task = sched.get_task("task-1").unwrap();
        assert_eq!(task.state, TaskState::Failed);
        assert_eq!(task.error, Some("Out of memory".into()));
    }

    #[test]
    fn test_memory_reservation() {
        let sched = Scheduler::new(Some(1000), 10.0, 100.0);

        // Submit tasks - memory is reserved at submit time
        let task1 = TaskMeta::with_memory("task-1".into(), "T1".into(), 400);
        let task2 = TaskMeta::with_memory("task-2".into(), "T2".into(), 400);
        sched.submit(task1);
        sched.submit(task2);

        // Both tasks should run (total 800 <= 1000)
        let runnable = sched.get_runnable_tasks(10);
        assert_eq!(runnable.len(), 2);

        // Memory is reserved at submit time
        let stats = sched.stats();
        assert_eq!(stats.reserved_memory, 800);
        assert_eq!(stats.running_count, 2);

        // Complete first task - releases memory
        sched.complete_task("task-1");

        let stats = sched.stats();
        assert_eq!(stats.reserved_memory, 400);
        assert_eq!(stats.completed_count, 1);

        // Complete second task
        sched.complete_task("task-2");

        let stats = sched.stats();
        assert_eq!(stats.reserved_memory, 0);
        assert_eq!(stats.completed_count, 2);
    }

    #[test]
    fn test_dependencies() {
        let sched = Scheduler::new(None, 10.0, 100.0);

        let task1 = TaskMeta::new("task-1".into(), "T1".into());
        let task2 = TaskMeta::new("task-2".into(), "T2".into())
            .with_dependencies(vec!["task-1".into()]);

        sched.submit(task1);
        sched.submit(task2);

        // Only task-1 should be runnable (task-2 depends on it)
        let runnable = sched.get_runnable_tasks(10);
        assert_eq!(runnable.len(), 1);
        assert_eq!(runnable[0], "task-1");

        // Complete task-1
        sched.complete_task("task-1");

        // Now task-2 should be runnable
        let runnable = sched.get_runnable_tasks(10);
        assert_eq!(runnable.len(), 1);
        assert_eq!(runnable[0], "task-2");
    }

    #[test]
    fn test_cancel_task() {
        let sched = Scheduler::new(None, 10.0, 100.0);

        let task = TaskMeta::new("task-1".into(), "Test".into());
        sched.submit(task);

        assert!(sched.cancel_task("task-1"));
        assert_eq!(sched.get_task_state("task-1"), Some(TaskState::Cancelled));
    }

    #[test]
    fn test_stats() {
        let sched = Scheduler::new(Some(10000), 10.0, 100.0);

        for i in 0..5 {
            let task = TaskMeta::with_memory(
                format!("task-{}", i),
                format!("Task {}", i),
                100,
            );
            sched.submit(task);
        }

        let stats = sched.stats();
        assert_eq!(stats.total_submitted, 5);
        assert_eq!(stats.pending_count, 5);
        assert_eq!(stats.running_count, 0);
    }
}
