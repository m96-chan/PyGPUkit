"""Pytest configuration and fixtures for PyGPUkit tests."""

import pytest

from pygpukit.core.backend import CPUSimulationBackend, get_backend, set_backend


@pytest.fixture(autouse=True)
def use_cpu_backend():
    """Use CPU simulation backend for all tests by default."""
    original_backend = get_backend()
    set_backend(CPUSimulationBackend())
    yield
    set_backend(original_backend)


@pytest.fixture
def cpu_backend():
    """Provide a CPU simulation backend."""
    return CPUSimulationBackend()
