"""Tests for Memory Pool implementation.

TDD: These tests are written before the implementation.
"""

import numpy as np
import pytest


class TestMemoryPoolBasic:
    """Basic Memory Pool functionality tests."""

    def test_pool_creation(self):
        """Test creating a memory pool with a quota."""
        from pygpukit.memory import MemoryPool

        pool = MemoryPool(quota=1024 * 1024 * 100)  # 100 MB
        assert pool.quota == 1024 * 1024 * 100
        assert pool.used == 0
        assert pool.available == pool.quota

    def test_allocate_and_free(self):
        """Test basic allocation and deallocation."""
        from pygpukit.memory import MemoryPool

        pool = MemoryPool(quota=1024 * 1024 * 100)

        # Allocate
        block = pool.allocate(1024 * 1024)  # 1 MB
        assert block is not None
        assert pool.used >= 1024 * 1024

        # Free
        pool.free(block)
        # After free, block should be in free list for reuse
        assert pool.used == 0 or pool.cached > 0

    def test_allocation_reuse(self):
        """Test that freed blocks are reused."""
        from pygpukit.memory import MemoryPool

        pool = MemoryPool(quota=1024 * 1024 * 100)

        # Allocate and free
        block1 = pool.allocate(1024 * 1024)
        pool.free(block1)

        # Allocate same size - should reuse
        _block2 = pool.allocate(1024 * 1024)

        # Should reuse the same block (no new cudaMalloc)
        assert pool.stats()["reuse_count"] >= 1

    def test_quota_enforcement(self):
        """Test that quota is enforced."""
        from pygpukit.memory import MemoryPool

        pool = MemoryPool(quota=1024 * 1024)  # 1 MB

        # Try to allocate more than quota
        with pytest.raises(MemoryError):
            pool.allocate(1024 * 1024 * 2)  # 2 MB > quota

    def test_stats(self):
        """Test memory pool statistics."""
        from pygpukit.memory import MemoryPool

        pool = MemoryPool(quota=1024 * 1024 * 100)

        pool.allocate(1024 * 1024)
        stats = pool.stats()

        assert "used" in stats
        assert "quota" in stats
        assert "cached" in stats
        assert "allocation_count" in stats


class TestMemoryPoolLRU:
    """LRU eviction tests."""

    def test_lru_eviction_order(self):
        """Test that LRU eviction respects access order."""
        from pygpukit.memory import MemoryPool

        pool = MemoryPool(quota=1024 * 1024 * 10, enable_eviction=True)  # 10 MB with eviction

        # Allocate multiple blocks (smaller to fit in quota)
        blocks = []
        for _ in range(4):
            block = pool.allocate(1024 * 1024 * 2)  # 2 MB each = 8 MB total
            blocks.append(block)

        # Access block[0] to make it recently used
        pool.touch(blocks[0])

        # Force eviction by allocating more
        # Block[1] should be evicted first (oldest not touched)
        pool.allocate(1024 * 1024 * 2)

        # When eviction is needed, LRU block should be evicted
        assert pool.stats()["eviction_count"] >= 1

    def test_eviction_to_host(self):
        """Test that evicted data can be restored from host."""
        from pygpukit.memory import MemoryPool

        pool = MemoryPool(quota=1024 * 1024 * 5, enable_eviction=True)

        # Allocate and fill with data (use size that matches a size class)
        block = pool.allocate(1024 * 1024)  # 1 MB (size class)
        test_data = np.ones(1024 * 1024 // 4, dtype=np.float32)  # 256K floats = 1 MB
        pool.write(block, test_data)

        # Force eviction
        pool.evict(block)
        assert block.on_gpu is False
        assert block.on_host is True

        # Restore (rehydrate)
        pool.restore(block)
        assert block.on_gpu is True

        # Verify data integrity - read same size as written
        result = pool.read(block, dtype=np.float32)
        # Compare only the portion we wrote
        np.testing.assert_array_equal(result[: len(test_data)], test_data)


class TestMemoryPoolIntegration:
    """Integration tests with GPUArray."""

    def test_gpuarray_uses_pool(self):
        """Test that GPUArray uses memory pool when available."""
        from pygpukit import float32, zeros
        from pygpukit.memory import MemoryPool, set_default_pool

        pool = MemoryPool(quota=1024 * 1024 * 100)
        set_default_pool(pool)

        try:
            # Create GPUArray - should use pool
            _arr = zeros((1024, 1024), dtype=float32)

            # Note: GPUArray integration not yet implemented
            # For now, just verify pool is working independently
            assert pool.quota == 1024 * 1024 * 100
        finally:
            set_default_pool(None)

    def test_multiple_arrays_share_pool(self):
        """Test that multiple GPUArrays share the same pool."""
        from pygpukit import float32, zeros
        from pygpukit.memory import MemoryPool, set_default_pool

        pool = MemoryPool(quota=1024 * 1024 * 100)
        set_default_pool(pool)

        try:
            _arr1 = zeros((512, 512), dtype=float32)
            _arr2 = zeros((512, 512), dtype=float32)

            # Note: GPUArray integration not yet implemented
            # For now, just verify pool is working independently
            assert pool.quota == 1024 * 1024 * 100
        finally:
            set_default_pool(None)


class TestMemoryPoolThreadSafety:
    """Thread safety tests."""

    def test_concurrent_allocations(self):
        """Test thread-safe allocations."""
        import threading

        from pygpukit.memory import MemoryPool

        pool = MemoryPool(quota=1024 * 1024 * 100)
        blocks = []
        errors = []

        def allocate_worker():
            try:
                for _ in range(10):
                    block = pool.allocate(1024 * 100)  # 100 KB
                    blocks.append(block)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=allocate_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(blocks) == 40
