//! PyGPUkit Rust Python bindings
//!
//! Provides PyO3 bindings for the Rust memory pool, scheduler, transfer engine, and kernel dispatcher.

use pyo3::prelude::*;

mod memory;
mod scheduler;
mod transfer;
mod dispatch;

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

    // Transfer submodule
    let transfer_module = PyModule::new(m.py(), "transfer")?;
    transfer::register(&transfer_module)?;
    m.add_submodule(&transfer_module)?;

    // Dispatch submodule
    let dispatch_module = PyModule::new(m.py(), "dispatch")?;
    dispatch::register(&dispatch_module)?;
    m.add_submodule(&dispatch_module)?;

    // Also export at top level for convenience
    m.add_class::<memory::PyMemoryPool>()?;
    m.add_class::<memory::PyMemoryBlock>()?;
    m.add_class::<memory::PyPoolStats>()?;
    m.add_class::<scheduler::PyScheduler>()?;
    m.add_class::<scheduler::PyTaskMeta>()?;
    m.add_class::<scheduler::PySchedulerStats>()?;
    m.add_class::<scheduler::PyTaskStats>()?;
    m.add_class::<transfer::PyAsyncTransferEngine>()?;
    m.add_class::<transfer::PyTransferOp>()?;
    m.add_class::<transfer::PyTransferStats>()?;
    m.add_class::<dispatch::PyKernelDispatcher>()?;
    m.add_class::<dispatch::PyLaunchConfig>()?;
    m.add_class::<dispatch::PyDispatchStats>()?;

    Ok(())
}
