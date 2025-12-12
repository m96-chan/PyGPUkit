//! Task representation and state management
//!
//! Mirrors Python's Task dataclass for API compatibility.

use std::time::{SystemTime, UNIX_EPOCH};

/// Task execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskState {
    /// Waiting to be scheduled
    Pending,
    /// Currently running
    Running,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed,
    /// Cancelled by user
    Cancelled,
}

impl Default for TaskState {
    fn default() -> Self {
        Self::Pending
    }
}

/// Task scheduling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskPolicy {
    /// First-in-first-out
    Fifo,
    /// Shortest job first
    Sjf,
    /// Priority-based
    Priority,
}

impl Default for TaskPolicy {
    fn default() -> Self {
        Self::Fifo
    }
}

/// Task metadata and state
#[derive(Debug, Clone)]
pub struct TaskMeta {
    /// Unique task identifier
    pub id: String,
    /// Task name/description
    pub name: String,
    /// Current state
    pub state: TaskState,
    /// Scheduling policy
    pub policy: TaskPolicy,
    /// Priority (higher = more important)
    pub priority: i32,
    /// Estimated memory requirement in bytes
    pub memory_estimate: usize,
    /// Submission timestamp
    pub submitted_at: f64,
    /// Start timestamp (if running/completed)
    pub started_at: Option<f64>,
    /// Completion timestamp (if completed/failed)
    pub completed_at: Option<f64>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Dependencies (task IDs that must complete first)
    pub dependencies: Vec<String>,
}

impl TaskMeta {
    /// Create a new task with default settings.
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            state: TaskState::Pending,
            policy: TaskPolicy::Fifo,
            priority: 0,
            memory_estimate: 0,
            submitted_at: Self::now(),
            started_at: None,
            completed_at: None,
            error: None,
            dependencies: Vec::new(),
        }
    }

    /// Create a task with memory estimate.
    pub fn with_memory(id: String, name: String, memory_estimate: usize) -> Self {
        let mut task = Self::new(id, name);
        task.memory_estimate = memory_estimate;
        task
    }

    /// Set task priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set scheduling policy.
    pub fn with_policy(mut self, policy: TaskPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Add dependencies.
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Mark task as running.
    pub fn start(&mut self) {
        self.state = TaskState::Running;
        self.started_at = Some(Self::now());
    }

    /// Mark task as completed.
    pub fn complete(&mut self) {
        self.state = TaskState::Completed;
        self.completed_at = Some(Self::now());
    }

    /// Mark task as failed.
    pub fn fail(&mut self, error: String) {
        self.state = TaskState::Failed;
        self.completed_at = Some(Self::now());
        self.error = Some(error);
    }

    /// Mark task as cancelled.
    pub fn cancel(&mut self) {
        self.state = TaskState::Cancelled;
        self.completed_at = Some(Self::now());
    }

    /// Get elapsed time since submission.
    pub fn elapsed(&self) -> f64 {
        Self::now() - self.submitted_at
    }

    /// Get execution duration (if started).
    pub fn duration(&self) -> Option<f64> {
        let start = self.started_at?;
        let end = self.completed_at.unwrap_or_else(Self::now);
        Some(end - start)
    }

    /// Check if task is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            TaskState::Completed | TaskState::Failed | TaskState::Cancelled
        )
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

/// Statistics for a single task
#[derive(Debug, Clone, Default)]
pub struct TaskStats {
    /// Task ID
    pub id: String,
    /// Task name
    pub name: String,
    /// Current state
    pub state: TaskState,
    /// Wait time before execution (seconds)
    pub wait_time: f64,
    /// Execution time (seconds)
    pub exec_time: f64,
    /// Memory used (bytes)
    pub memory_used: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_creation() {
        let task = TaskMeta::new("task-1".into(), "Test Task".into());
        assert_eq!(task.id, "task-1");
        assert_eq!(task.state, TaskState::Pending);
        assert!(task.submitted_at > 0.0);
    }

    #[test]
    fn test_task_lifecycle() {
        let mut task = TaskMeta::new("task-1".into(), "Test".into());
        assert_eq!(task.state, TaskState::Pending);
        assert!(!task.is_terminal());

        task.start();
        assert_eq!(task.state, TaskState::Running);
        assert!(task.started_at.is_some());

        task.complete();
        assert_eq!(task.state, TaskState::Completed);
        assert!(task.is_terminal());
        assert!(task.duration().is_some());
    }

    #[test]
    fn test_task_failure() {
        let mut task = TaskMeta::new("task-1".into(), "Test".into());
        task.start();
        task.fail("Out of memory".into());

        assert_eq!(task.state, TaskState::Failed);
        assert_eq!(task.error, Some("Out of memory".into()));
        assert!(task.is_terminal());
    }

    #[test]
    fn test_task_builder() {
        let task = TaskMeta::with_memory("task-1".into(), "Heavy".into(), 1024 * 1024)
            .with_priority(10)
            .with_policy(TaskPolicy::Priority)
            .with_dependencies(vec!["task-0".into()]);

        assert_eq!(task.memory_estimate, 1024 * 1024);
        assert_eq!(task.priority, 10);
        assert_eq!(task.policy, TaskPolicy::Priority);
        assert_eq!(task.dependencies, vec!["task-0"]);
    }
}
