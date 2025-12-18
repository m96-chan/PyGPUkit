"""Multi-LLM Execution Context Management.

Provides execution context management for running multiple LLM instances
concurrently on a single GPU with stream-based isolation.

Example:
    >>> from pygpukit.scheduler import create_context, session
    >>>
    >>> # Create execution contexts for two LLMs
    >>> ctx1 = create_context("gpt2_a", max_vram=4 * GB)
    >>> ctx2 = create_context("gpt2_b", max_vram=4 * GB)
    >>>
    >>> # Run both LLMs in a session
    >>> with session():
    ...     llm1.generate("Hello")
    ...     llm2.generate("World")
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Constants
KB = 1024
MB = 1024 * KB
GB = 1024 * MB

# Try to import Rust backend
_controller = None

try:
    import _pygpukit_rust._pygpukit_rust as _rust

    _MultiLLMController = _rust.MultiLLMController
    _ContextState = _rust.ContextState
    _ContextStats = _rust.ContextStats
    _ControllerStats = _rust.ControllerStats
    # Async types
    _FutureState = _rust.FutureState
    _KernelFuture = _rust.KernelFuture
    _KernelResult = _rust.KernelResult
    _AsyncKernelRequest = _rust.AsyncKernelRequest
    _AsyncExecStats = _rust.AsyncExecStats
    HAS_MULTI_LLM = True
except ImportError:
    _MultiLLMController = None
    _ContextState = None
    _ContextStats = None
    _ControllerStats = None
    _FutureState = None
    _KernelFuture = None
    _KernelResult = None
    _AsyncKernelRequest = None
    _AsyncExecStats = None
    HAS_MULTI_LLM = False


def _get_controller():
    """Get or create the global controller instance."""
    global _controller
    if _controller is None:
        if not HAS_MULTI_LLM:
            raise RuntimeError(
                "Multi-LLM scheduler requires Rust backend. "
                "Please rebuild PyGPUkit with Rust support."
            )
        _controller = _MultiLLMController()
    return _controller


def initialize(
    device_id: int = 0,
    device_total_memory: int = 0,
    total_vram_budget: int = 0,
) -> None:
    """Initialize the multi-LLM scheduler.

    This must be called before creating execution contexts.
    If not called explicitly, it will be called automatically
    with default parameters on first context creation.

    Args:
        device_id: CUDA device ID (default 0)
        device_total_memory: Total device memory in bytes (0 = auto-detect)
        total_vram_budget: Total VRAM budget for all contexts (0 = device total)
    """
    controller = _get_controller()
    if not controller.is_initialized():
        controller.initialize(device_id, device_total_memory, total_vram_budget)


def create_context(
    llm_id: str,
    max_vram: int = 0,
    *,
    device_id: int = 0,
) -> ExecutionContext:
    """Create an execution context for an LLM instance.

    Each LLM must have exactly one execution context. The context
    provides a dedicated CUDA stream for kernel isolation and
    optional VRAM budget tracking.

    Args:
        llm_id: Unique identifier for the LLM instance
        max_vram: Maximum VRAM budget for this LLM in bytes (0 = share global budget)
        device_id: CUDA device ID (used for auto-initialization)

    Returns:
        ExecutionContext for the LLM

    Raises:
        RuntimeError: If context already exists for llm_id

    Example:
        >>> ctx = create_context("gpt2_a", max_vram=4 * GB)
        >>> print(ctx.stream_id)
        0
    """
    controller = _get_controller()

    # Auto-initialize if needed
    if not controller.is_initialized():
        initialize(device_id=device_id)

    stream_id = controller.create_context(llm_id, max_vram)
    return ExecutionContext(llm_id, stream_id, max_vram)


def get_context(llm_id: str) -> ExecutionContext | None:
    """Get an existing execution context by LLM ID.

    Args:
        llm_id: LLM identifier

    Returns:
        ExecutionContext if found, None otherwise
    """
    controller = _get_controller()
    stats = controller.get_context(llm_id)
    if stats is None:
        return None
    return ExecutionContext(stats.llm_id, stats.stream_id, stats.max_vram)


def destroy_context(llm_id: str) -> bool:
    """Destroy an execution context.

    Args:
        llm_id: LLM identifier

    Returns:
        True if context was destroyed, False if not found
    """
    controller = _get_controller()
    return controller.destroy_context(llm_id)


def list_contexts() -> list[str]:
    """List all active context LLM IDs.

    Returns:
        List of LLM identifiers
    """
    controller = _get_controller()
    return controller.list_contexts()


@contextmanager
def session() -> Generator[None, None, None]:
    """Context manager for a multi-LLM session.

    Within a session, all contexts are marked as running.
    When the session ends, all contexts are synchronized
    and marked as idle.

    Example:
        >>> with session():
        ...     llm1.generate("Hello")
        ...     llm2.generate("World")
        ... # All streams synchronized here
    """
    controller = _get_controller()
    controller.start_session()
    try:
        yield
    finally:
        controller.end_session()


class context_session:
    """Context manager for a per-context session.

    Supports both sync `with` and async `async with`.

    Unlike the global session(), this starts a session for a specific context,
    allowing independent LLM execution. Each context can have its own
    session lifecycle.

    Args:
        ctx: The ExecutionContext to start a session for

    Example (sync):
        >>> with context_session(tts_ctx), context_session(llm_ctx):
        ...     tts_future = tts_ctx.dispatch_async(tts_request)
        ...     llm_future = llm_ctx.dispatch_async(llm_request)
        ...     text = llm_future.wait()
        ...     audio = tts_future.wait()

    Example (async):
        >>> async with context_session(llm_ctx), context_session(tts_ctx):
        ...     llm_f = llm_ctx.dispatch_async(llm_req)
        ...     tts_f = tts_ctx.dispatch_async(tts_req)
        ...     text, audio = await asyncio.gather(llm_f, tts_f)
    """

    def __init__(self, ctx: ExecutionContext):
        self._ctx = ctx

    # Sync context manager
    def __enter__(self) -> None:
        self._ctx.start_session()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._ctx.end_session()

    # Async context manager
    async def __aenter__(self) -> None:
        self._ctx.start_session()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._ctx.end_session()


def is_session_active() -> bool:
    """Check if a session is currently active.

    Returns:
        True if session is active
    """
    controller = _get_controller()
    return controller.is_session_active()


def stats() -> SchedulerStats:
    """Get scheduler statistics.

    Returns:
        SchedulerStats object with current state
    """
    controller = _get_controller()
    return SchedulerStats(controller.stats())


def reset() -> None:
    """Reset the scheduler, destroying all contexts."""
    controller = _get_controller()
    controller.reset()


class AsyncKernelRequest:
    """Request for async kernel dispatch.

    Use this to specify kernel dispatch parameters.

    Example:
        >>> request = AsyncKernelRequest.linear(kernel_handle, 1024, 256)
        >>> future = ctx.dispatch_async(request)
    """

    def __init__(self, kernel_handle: int):
        """Create a new async kernel request.

        Args:
            kernel_handle: Kernel function handle (CUfunction as int)
        """
        if not HAS_MULTI_LLM:
            raise RuntimeError("Multi-LLM scheduler requires Rust backend.")
        self._inner = _AsyncKernelRequest(kernel_handle)

    @classmethod
    def linear(
        cls, kernel_handle: int, n_elements: int, block_size: int = 256
    ) -> AsyncKernelRequest:
        """Create a linear kernel request (1D grid).

        Args:
            kernel_handle: Kernel function handle
            n_elements: Number of elements to process
            block_size: Threads per block (default 256)
        """
        if not HAS_MULTI_LLM:
            raise RuntimeError("Multi-LLM scheduler requires Rust backend.")
        obj = cls.__new__(cls)
        obj._inner = _AsyncKernelRequest.linear(kernel_handle, n_elements, block_size)
        return obj

    def with_grid(self, x: int, y: int = 1, z: int = 1) -> AsyncKernelRequest:
        """Set grid dimensions."""
        new_obj = AsyncKernelRequest.__new__(AsyncKernelRequest)
        new_obj._inner = self._inner.with_grid(x, y, z)
        return new_obj

    def with_block(self, x: int, y: int = 1, z: int = 1) -> AsyncKernelRequest:
        """Set block dimensions."""
        new_obj = AsyncKernelRequest.__new__(AsyncKernelRequest)
        new_obj._inner = self._inner.with_block(x, y, z)
        return new_obj

    def with_shared_mem(self, bytes: int) -> AsyncKernelRequest:
        """Set shared memory size."""
        new_obj = AsyncKernelRequest.__new__(AsyncKernelRequest)
        new_obj._inner = self._inner.with_shared_mem(bytes)
        return new_obj

    def with_args(self, args: list[int]) -> AsyncKernelRequest:
        """Set kernel arguments (as list of u64 pointers)."""
        new_obj = AsyncKernelRequest.__new__(AsyncKernelRequest)
        new_obj._inner = self._inner.with_args(args)
        return new_obj

    @property
    def kernel_handle(self) -> int:
        return self._inner.kernel_handle

    @property
    def grid(self) -> tuple[int, int, int]:
        return self._inner.grid

    @property
    def block(self) -> tuple[int, int, int]:
        return self._inner.block

    def __repr__(self) -> str:
        return (
            f"AsyncKernelRequest(handle=0x{self.kernel_handle:x}, "
            f"grid={self.grid}, block={self.block})"
        )


class KernelFuture:
    """Handle for tracking async kernel execution.

    Supports both synchronous `wait()` and Python asyncio `await`.

    Example (sync):
        >>> future = ctx.dispatch_async(request)
        >>> result = future.wait()  # Blocking

    Example (async):
        >>> async with context_session(ctx):
        ...     future = ctx.dispatch_async(request)
        ...     result = await future  # Non-blocking in event loop
    """

    def __init__(self, inner: Any):
        self._inner = inner

    @property
    def id(self) -> int:
        """Future ID."""
        return self._inner.id

    @property
    def stream_id(self) -> int:
        """Stream ID where kernel is executing."""
        return self._inner.stream_id

    @property
    def context_id(self) -> str:
        """Context ID (LLM ID)."""
        return self._inner.context_id

    @property
    def state(self) -> str:
        """Current state: 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'."""
        state_val = self._inner.state
        if hasattr(state_val, "value"):
            state_val = state_val.value
        return ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"][state_val]

    def is_ready(self) -> bool:
        """Check if kernel execution is complete (non-blocking)."""
        return self._inner.is_ready()

    def wait(self) -> KernelResult:
        """Wait for kernel completion (blocking).

        Returns the kernel result. If already complete, returns immediately.
        If still running, blocks until completion.
        """
        return KernelResult(self._inner.wait())

    def try_get(self) -> KernelResult | None:
        """Try to get result without blocking.

        Returns None if not ready yet.
        """
        result = self._inner.try_get()
        return KernelResult(result) if result is not None else None

    def exec_time(self) -> float | None:
        """Get execution time (if completed)."""
        return self._inner.exec_time()

    def _wait_sync(self) -> KernelResult:
        """Synchronous wait (for executor)."""
        return KernelResult(self._inner.wait())

    def __await__(self) -> Generator[Any, None, KernelResult]:
        """Make KernelFuture awaitable for asyncio.

        Uses run_in_executor to avoid blocking the event loop.
        The blocking `wait()` runs in the default ThreadPoolExecutor.

        Example:
            >>> result = await future
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, self._wait_sync).__await__()

    def __repr__(self) -> str:
        return f"KernelFuture(id={self.id}, context='{self.context_id}', state={self.state})"


class KernelResult:
    """Result of an async kernel execution."""

    def __init__(self, inner: Any):
        self._inner = inner

    @property
    def success(self) -> bool:
        """Whether execution succeeded."""
        return self._inner.success

    @property
    def error(self) -> str | None:
        """Error message if failed."""
        return self._inner.error

    @property
    def exec_time(self) -> float:
        """Execution time in seconds."""
        return self._inner.exec_time

    @property
    def output(self) -> bytes | None:
        """Output data as bytes (if any)."""
        return self._inner.output

    def __repr__(self) -> str:
        if self.success:
            return f"KernelResult(success=True, exec_time={self.exec_time:.4f}s)"
        return f"KernelResult(success=False, error='{self.error}')"


class ExecutionContext:
    """Execution context for an LLM instance.

    Each LLM is bound to exactly one ExecutionContext, which provides:
    - Dedicated CUDA stream for kernel isolation
    - Optional VRAM budget tracking
    - State management (IDLE, RUNNING, PAUSED)
    - Async kernel dispatch with KernelFuture

    Do not instantiate directly; use `create_context()` instead.

    Example:
        >>> ctx = create_context("gpt2", max_vram=4 * GB)
        >>>
        >>> # Async execution
        >>> request = AsyncKernelRequest.linear(kernel_handle, 1024, 256)
        >>> future = ctx.dispatch_async(request)
        >>> # Do other work...
        >>> result = future.wait()
    """

    def __init__(self, llm_id: str, stream_id: int, max_vram: int):
        self._llm_id = llm_id
        self._stream_id = stream_id
        self._max_vram = max_vram

    @property
    def llm_id(self) -> str:
        """LLM identifier."""
        return self._llm_id

    @property
    def stream_id(self) -> int:
        """Assigned CUDA stream ID."""
        return self._stream_id

    @property
    def max_vram(self) -> int:
        """Maximum VRAM budget in bytes (0 = unlimited)."""
        return self._max_vram

    @property
    def stats(self) -> ContextStats | None:
        """Get current context statistics."""
        controller = _get_controller()
        rust_stats = controller.get_context(self._llm_id)
        if rust_stats is None:
            return None
        return ContextStats(rust_stats)

    def track_allocation(self, buffer_id: int, size: int) -> bool:
        """Track a memory allocation.

        Args:
            buffer_id: Unique buffer identifier
            size: Size in bytes

        Returns:
            True if allocation fits within budget
        """
        controller = _get_controller()
        return controller.track_allocation(self._llm_id, buffer_id, size)

    def track_deallocation(self, buffer_id: int) -> None:
        """Track a memory deallocation.

        Args:
            buffer_id: Buffer identifier
        """
        controller = _get_controller()
        controller.track_deallocation(self._llm_id, buffer_id)

    # --- Async Execution ---

    def dispatch_async(self, request: AsyncKernelRequest) -> KernelFuture:
        """Dispatch an async kernel.

        Returns a KernelFuture that can be used to wait for completion.
        The kernel is queued for execution on this context's stream.

        Args:
            request: Kernel dispatch request

        Returns:
            KernelFuture for tracking execution

        Example:
            >>> request = AsyncKernelRequest.linear(kernel_handle, 1024, 256)
            >>> future = ctx.dispatch_async(request)
            >>> # Do other work while kernel executes...
            >>> result = future.wait()
        """
        controller = _get_controller()
        rust_future = controller.dispatch_async(self._llm_id, request._inner)
        return KernelFuture(rust_future)

    def start_session(self) -> None:
        """Start a per-context session.

        Unlike the global session(), per-context sessions allow independent
        LLM execution. Each context can have its own session lifecycle.

        Example:
            >>> tts_ctx.start_session()
            >>> llm_ctx.start_session()
            >>>
            >>> # Both run independently
            >>> tts_future = tts_ctx.dispatch_async(tts_request)
            >>> llm_future = llm_ctx.dispatch_async(llm_request)
            >>>
            >>> # Wait for results in any order
            >>> llm_result = llm_future.wait()
            >>> tts_result = tts_future.wait()
        """
        controller = _get_controller()
        controller.start_context_session(self._llm_id)

    def end_session(self) -> None:
        """End the per-context session."""
        controller = _get_controller()
        controller.end_context_session(self._llm_id)

    def is_session_active(self) -> bool:
        """Check if a session is active for this context."""
        controller = _get_controller()
        result = controller.is_context_session_active(self._llm_id)
        return result if result is not None else False

    def destroy(self) -> bool:
        """Destroy this context.

        Returns:
            True if context was destroyed
        """
        return destroy_context(self._llm_id)

    def __repr__(self) -> str:
        return f"ExecutionContext(llm_id='{self._llm_id}', stream_id={self._stream_id})"


class ContextStats:
    """Statistics for an execution context."""

    def __init__(self, rust_stats: Any):
        self._inner = rust_stats

    @property
    def llm_id(self) -> str:
        return self._inner.llm_id

    @property
    def state(self) -> str:
        """Current state: 'IDLE', 'RUNNING', or 'PAUSED'."""
        state_val = self._inner.state
        if hasattr(state_val, "value"):
            state_val = state_val.value
        return ["IDLE", "RUNNING", "PAUSED"][state_val]

    @property
    def stream_id(self) -> int:
        return self._inner.stream_id

    @property
    def max_vram(self) -> int:
        return self._inner.max_vram

    @property
    def used_vram(self) -> int:
        return self._inner.used_vram

    @property
    def available_vram(self) -> int:
        return self._inner.available_vram

    @property
    def buffer_count(self) -> int:
        return self._inner.buffer_count

    def __repr__(self) -> str:
        return (
            f"ContextStats(llm_id='{self.llm_id}', state={self.state}, used_vram={self.used_vram})"
        )


class SchedulerStats:
    """Statistics for the multi-LLM scheduler."""

    def __init__(self, rust_stats: Any):
        self._inner = rust_stats

    @property
    def initialized(self) -> bool:
        return self._inner.initialized

    @property
    def device_id(self) -> int:
        return self._inner.device_id

    @property
    def total_vram_budget(self) -> int:
        return self._inner.total_vram_budget

    @property
    def device_total_memory(self) -> int:
        return self._inner.device_total_memory

    @property
    def used_vram(self) -> int:
        return self._inner.used_vram

    @property
    def available_vram(self) -> int:
        return self._inner.available_vram

    @property
    def context_count(self) -> int:
        return self._inner.context_count

    @property
    def stream_pool_size(self) -> int:
        return self._inner.stream_pool_size

    def __repr__(self) -> str:
        return (
            f"SchedulerStats(contexts={self.context_count}, "
            f"used_vram={self.used_vram}, available_vram={self.available_vram})"
        )


# Export constants
__all__ = [
    # Constants
    "KB",
    "MB",
    "GB",
    # Functions
    "initialize",
    "create_context",
    "get_context",
    "destroy_context",
    "list_contexts",
    "session",
    "context_session",
    "is_session_active",
    "stats",
    "reset",
    # Classes
    "ExecutionContext",
    "ContextStats",
    "SchedulerStats",
    # Async execution classes
    "AsyncKernelRequest",
    "KernelFuture",
    "KernelResult",
    # Feature flag
    "HAS_MULTI_LLM",
]
