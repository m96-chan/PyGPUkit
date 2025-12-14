//! PyGPUkit Rust Python bindings
//!
//! Provides PyO3 bindings for the Rust memory pool, scheduler, transfer engine, and kernel dispatcher.

use pyo3::prelude::*;

mod errors;
mod memory;
mod scheduler;
mod transfer;
mod dispatch;
mod device;

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

    // Device submodule
    let device_module = PyModule::new(m.py(), "device")?;
    device::register(&device_module)?;
    m.add_submodule(&device_module)?;

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
    // Pacing
    m.add_class::<dispatch::PyKernelPacingEngine>()?;
    m.add_class::<dispatch::PyPacingConfig>()?;
    m.add_class::<dispatch::PyPacingDecision>()?;
    m.add_class::<dispatch::PyPacingStats>()?;
    // Admission control
    m.add_class::<scheduler::PyAdmissionConfig>()?;
    m.add_class::<scheduler::PyAdmissionController>()?;
    m.add_class::<scheduler::PyAdmissionDecision>()?;
    m.add_class::<scheduler::PyAdmissionStats>()?;
    m.add_class::<scheduler::PyRejectReasonEnum>()?;
    m.add_class::<scheduler::PyRejectReasonDetails>()?;
    // QoS policy
    m.add_class::<scheduler::PyQosClass>()?;
    m.add_class::<scheduler::PyQosPolicy>()?;
    m.add_class::<scheduler::PyQosTaskMeta>()?;
    m.add_class::<scheduler::PyQosEvaluation>()?;
    m.add_class::<scheduler::PyQosPolicyEvaluator>()?;
    m.add_class::<scheduler::PyQosStats>()?;
    m.add_class::<scheduler::PyResourceRequirements>()?;
    // Slicing
    m.add_class::<dispatch::PySliceConfig>()?;
    m.add_class::<dispatch::PySliceScheduler>()?;
    m.add_class::<dispatch::PySliceInfo>()?;
    m.add_class::<dispatch::PySliceStats>()?;
    m.add_class::<dispatch::PyKernelSlice>()?;
    // Pinned memory
    m.add_class::<transfer::PyPinnedMemoryManager>()?;
    m.add_class::<transfer::PyPinnedPoolConfig>()?;
    m.add_class::<transfer::PyPinnedBlock>()?;
    m.add_class::<transfer::PyPinnedStats>()?;
    // Kernel cache
    m.add_class::<dispatch::PyKernelCache>()?;
    m.add_class::<dispatch::PyCacheConfig>()?;
    m.add_class::<dispatch::PyCachedKernel>()?;
    m.add_class::<dispatch::PyCacheStats>()?;
    m.add_class::<dispatch::PyCompileOptions>()?;
    // Partitioning
    m.add_class::<scheduler::PyPartitionManager>()?;
    m.add_class::<scheduler::PyPartitionConfig>()?;
    m.add_class::<scheduler::PyPartitionLimits>()?;
    m.add_class::<scheduler::PyPartitionUsage>()?;
    m.add_class::<scheduler::PyPartition>()?;
    m.add_class::<scheduler::PyPartitionStats>()?;
    // Device capabilities
    m.add_class::<device::PyKernelType>()?;
    m.add_class::<device::PyDeviceCapabilities>()?;

    Ok(())
}
