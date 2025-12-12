"""
TDD Tests for v0.2.1 - Admission Control & QoS Policy Spec
Tests written FIRST before implementation fixes.
"""
import pytest

# Skip all tests if Rust module not available
pytest.importorskip("_pygpukit_rust")
import _pygpukit_rust as rust


class TestAdmissionControlSpec:
    """Admission Control specification tests."""

    def test_admission_config_defaults(self):
        """AdmissionConfig should have sensible defaults."""
        config = rust.AdmissionConfig(
            max_memory=1024 * 1024 * 1024,  # 1GB
            max_bandwidth=1.0,
        )
        assert config.max_memory == 1024 * 1024 * 1024
        assert config.max_bandwidth == 1.0

    def test_admission_controller_creation(self):
        """AdmissionController should be creatable with config."""
        config = rust.AdmissionConfig(
            max_memory=1024 * 1024 * 1024,
            max_bandwidth=1.0,
        )
        controller = rust.AdmissionController(config)
        assert controller is not None

    def test_admission_decision_types(self):
        """AdmissionDecision should have Admitted, Queued, Rejected states."""
        # Test that we can create decisions
        admitted = rust.AdmissionDecision.Admitted()
        queued = rust.AdmissionDecision.Queued()

        assert admitted.is_admitted()
        assert not admitted.is_queued()
        assert not admitted.is_rejected()

        assert queued.is_queued()
        assert not queued.is_admitted()

    def test_admission_with_memory_request(self):
        """Admission should consider memory requests."""
        config = rust.AdmissionConfig(
            max_memory=100 * 1024 * 1024,  # 100MB
            max_bandwidth=1.0,
            enable_best_effort=False,  # Strict mode - reject over-quota requests
        )
        controller = rust.AdmissionController(config)

        # Small request should be admitted
        decision1 = controller.try_admit("task1", 10 * 1024 * 1024, 0.1)
        assert decision1.is_admitted()

        # Request exceeding quota should be rejected
        decision2 = controller.try_admit("task2", 200 * 1024 * 1024, 0.1)
        assert decision2.is_rejected()

    def test_admission_release(self):
        """Released resources should be available again."""
        config = rust.AdmissionConfig(
            max_memory=100 * 1024 * 1024,
            max_bandwidth=1.0,
        )
        controller = rust.AdmissionController(config)

        # Admit first task
        decision1 = controller.try_admit("task1", 80 * 1024 * 1024, 0.5)
        assert decision1.is_admitted()

        # Second task should be queued/rejected (not enough memory)
        decision2 = controller.try_admit("task2", 50 * 1024 * 1024, 0.3)
        assert not decision2.is_admitted()

        # Release first task
        controller.release("task1")

        # Now second task should be admittable
        decision3 = controller.try_admit("task2", 50 * 1024 * 1024, 0.3)
        assert decision3.is_admitted()

    def test_admission_stats(self):
        """AdmissionController should provide stats."""
        config = rust.AdmissionConfig(
            max_memory=100 * 1024 * 1024,
            max_bandwidth=1.0,
        )
        controller = rust.AdmissionController(config)

        stats = controller.stats()
        assert hasattr(stats, 'used_memory')
        assert hasattr(stats, 'used_bandwidth')
        assert hasattr(stats, 'admitted_count')
        assert hasattr(stats, 'rejected_count')


class TestQoSPolicySpec:
    """QoS Policy specification tests."""

    def test_qos_class_enum(self):
        """QoS classes should be Guaranteed, Burstable, BestEffort."""
        assert hasattr(rust, 'QosClass')

        # Should be able to get class values
        guaranteed = rust.QosClass.Guaranteed
        burstable = rust.QosClass.Burstable
        best_effort = rust.QosClass.BestEffort

        # They should be distinct
        assert guaranteed != burstable
        assert burstable != best_effort

    def test_qos_task_meta_guaranteed(self):
        """Guaranteed tasks should have strict resource requirements."""
        task = rust.QosTaskMeta.guaranteed("task1", "Test Task", 256 * 1024 * 1024)

        assert task.id == "task1"
        assert task.name == "Test Task"
        assert task.qos_class == rust.QosClass.Guaranteed
        assert task.memory_request == 256 * 1024 * 1024

    def test_qos_task_meta_burstable(self):
        """Burstable tasks should have base + burst ratio."""
        task = rust.QosTaskMeta.burstable("task2", "Burstable Task", 128 * 1024 * 1024, 2.0)

        assert task.id == "task2"
        assert task.qos_class == rust.QosClass.Burstable
        assert task.memory_request == 128 * 1024 * 1024
        assert task.burst_ratio == 2.0

    def test_qos_task_meta_best_effort(self):
        """BestEffort tasks should have minimal requirements."""
        task = rust.QosTaskMeta.best_effort("task3", "Background Task")

        assert task.id == "task3"
        assert task.qos_class == rust.QosClass.BestEffort

    def test_qos_evaluator_creation(self):
        """QosPolicyEvaluator should be creatable with resource limits."""
        evaluator = rust.QosPolicyEvaluator(
            total_memory=8 * 1024 * 1024 * 1024,  # 8GB
            total_bandwidth=1.0,
        )
        assert evaluator is not None

    def test_qos_evaluation_guaranteed_priority(self):
        """Guaranteed tasks should have highest priority."""
        evaluator = rust.QosPolicyEvaluator(
            total_memory=8 * 1024 * 1024 * 1024,
            total_bandwidth=1.0,
        )

        guaranteed = rust.QosTaskMeta.guaranteed("g1", "Guaranteed", 1024 * 1024 * 1024)
        burstable = rust.QosTaskMeta.burstable("b1", "Burstable", 512 * 1024 * 1024, 1.5)
        best_effort = rust.QosTaskMeta.best_effort("be1", "BestEffort")

        _eval_g = evaluator.evaluate(guaranteed)
        _eval_b = evaluator.evaluate(burstable)
        _eval_be = evaluator.evaluate(best_effort)

        # Guaranteed should have highest effective priority
        assert guaranteed.effective_priority() > burstable.effective_priority()
        assert burstable.effective_priority() > best_effort.effective_priority()

    def test_qos_resource_reservation(self):
        """QoS should track reserved resources correctly."""
        evaluator = rust.QosPolicyEvaluator(
            total_memory=1024 * 1024 * 1024,  # 1GB
            total_bandwidth=1.0,
        )

        task = rust.QosTaskMeta.guaranteed("task1", "Test", 512 * 1024 * 1024)
        eval_result = evaluator.evaluate(task)

        assert eval_result.is_admitted()

        # Reserve the resources
        evaluator.reserve(eval_result)

        # Check stats
        stats = evaluator.stats()
        assert stats.guaranteed_memory == 512 * 1024 * 1024

    def test_qos_throttling(self):
        """Burstable tasks should be throttled when exceeding base allocation."""
        evaluator = rust.QosPolicyEvaluator(
            total_memory=1024 * 1024 * 1024,
            total_bandwidth=1.0,
        )

        # Fill up guaranteed capacity
        for i in range(4):
            task = rust.QosTaskMeta.guaranteed(f"g{i}", f"Guaranteed {i}", 200 * 1024 * 1024)
            eval_result = evaluator.evaluate(task)
            if eval_result.is_admitted():
                evaluator.reserve(eval_result)

        # New burstable task should potentially be throttled
        burstable = rust.QosTaskMeta.burstable("b1", "Burstable", 100 * 1024 * 1024, 2.0)
        eval_b = evaluator.evaluate(burstable)

        # Should either be admitted with throttling or queued
        assert eval_b.is_admitted() or eval_b.is_throttled() or eval_b.is_queued()


class TestQoSPolicyIntegration:
    """Integration tests for QoS with other components."""

    def test_qos_with_partitioning(self):
        """QoS should work with GPU partitioning."""
        # Create partition manager
        pm = rust.PartitionManager(rust.PartitionConfig(total_memory=8 * 1024 * 1024 * 1024))

        # Create inference partition
        pm.create_partition(
            "inference", "Inference Partition",
            rust.PartitionLimits().memory(4 * 1024 * 1024 * 1024).compute(0.5)
        )

        # Create QoS evaluator for the partition
        evaluator = rust.QosPolicyEvaluator(
            total_memory=4 * 1024 * 1024 * 1024,  # Partition's quota
            total_bandwidth=0.5,  # Partition's bandwidth share
        )

        # Submit task to partition
        task = rust.QosTaskMeta.guaranteed("inf1", "Inference Task", 1024 * 1024 * 1024)
        eval_result = evaluator.evaluate(task)

        assert eval_result.is_admitted()

    def test_qos_with_admission_control(self):
        """QoS decisions should align with admission control."""
        # Admission control with strict limits
        admission_config = rust.AdmissionConfig(
            max_memory=1024 * 1024 * 1024,
            max_bandwidth=1.0,
        )
        admission = rust.AdmissionController(admission_config)

        # QoS evaluator with same limits
        qos = rust.QosPolicyEvaluator(
            total_memory=1024 * 1024 * 1024,
            total_bandwidth=1.0,
        )

        # Create task
        task = rust.QosTaskMeta.guaranteed("task1", "Test", 512 * 1024 * 1024)

        # Both should agree on admission
        qos_result = qos.evaluate(task)
        admission_result = admission.try_admit("task1", 512 * 1024 * 1024, 0.5)

        assert qos_result.is_admitted() == admission_result.is_admitted()


class TestRejectReason:
    """Tests for rejection reason reporting."""

    def test_reject_reason_memory(self):
        """Should report memory as rejection reason."""
        config = rust.AdmissionConfig(
            max_memory=100 * 1024 * 1024,
            max_bandwidth=1.0,
            enable_best_effort=False,  # Strict mode for rejection
        )
        controller = rust.AdmissionController(config)

        # Request more than available
        decision = controller.try_admit("task1", 200 * 1024 * 1024, 0.1)

        assert decision.is_rejected()
        assert decision.reject_reason() == rust.RejectReason.InsufficientMemory

    def test_reject_reason_bandwidth(self):
        """Should report bandwidth as rejection reason.

        Note: Bandwidth is calculated from memory request (memory/total_memory).
        To trigger bandwidth rejection, request memory that results in
        bandwidth_estimate > max_bandwidth.
        """
        config = rust.AdmissionConfig(
            max_memory=1024 * 1024 * 1024,  # 1GB
            max_bandwidth=0.5,  # 50% bandwidth limit
            enable_best_effort=False,  # Strict mode for rejection
        )
        controller = rust.AdmissionController(config)

        # Request 600MB which results in bandwidth_estimate = 600/1024 = 0.586 > 0.5
        # This should trigger bandwidth rejection (memory is available but bandwidth exceeded)
        decision = controller.try_admit("task1", 600 * 1024 * 1024, 0.0)

        assert decision.is_rejected()
        assert decision.reject_reason() == rust.RejectReason.InsufficientBandwidth
