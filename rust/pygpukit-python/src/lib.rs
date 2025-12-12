//! PyGPUkit Rust Python bindings
//!
//! Provides PyO3 bindings for the Rust memory pool and scheduler.

use pyo3::prelude::*;

mod memory;
mod scheduler;

/// PyGPUkit Rust module
#[pymodule]
fn _pygpukit_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Memory submodule
    let memory_module = PyModule::new(m.py(), "memory")?;
    memory::register(&memory_module)?;
    m.add_submodule(&memory_module)?;

    // Scheduler submodule
    let scheduler_module = PyModule::new(m.py(), "scheduler")?;
    scheduler::register(&scheduler_module)?;
    m.add_submodule(&scheduler_module)?;

    // Also export at top level for convenience
    m.add_class::<memory::PyMemoryPool>()?;
    m.add_class::<memory::PyMemoryBlock>()?;
    m.add_class::<memory::PyPoolStats>()?;
    m.add_class::<scheduler::PyScheduler>()?;
    m.add_class::<scheduler::PyTaskMeta>()?;
    m.add_class::<scheduler::PySchedulerStats>()?;
    m.add_class::<scheduler::PyTaskStats>()?;

    Ok(())
}
