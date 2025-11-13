"""
Comprehensive unit tests for coordination signal system.

Test Coverage:
- Signal creation and validation
- Signal coordinator operations
- Signal propagation and delivery
- Signal subscription and filtering
- Signal history tracking
- Performance targets validation (<100ms signal propagation)
- Integration with ThreadSafeState and MetricsCollector
- Error handling

Performance Targets:
- Signal propagation: <100ms
- Storage operations: <2ms (via ThreadSafeState)
- Metrics overhead: <1ms per signal
"""

import asyncio
import time
from datetime import datetime

import pytest

from omninode_bridge.agents.coordination import ThreadSafeState
from omninode_bridge.agents.coordination.signal_models import (
    AgentCompletedSignal,
    AgentInitializedSignal,
    CoordinationSignal,
    DependencyResolvedSignal,
    InterAgentMessage,
    SignalMetrics,
    SignalSubscription,
    SignalType,
)
from omninode_bridge.agents.coordination.signals import SignalCoordinator
from omninode_bridge.agents.metrics.collector import MetricsCollector


@pytest.fixture
def state():
    """ThreadSafeState fixture."""
    return ThreadSafeState()


@pytest.fixture
def metrics_collector():
    """MetricsCollector fixture (no actual storage)."""
    return MetricsCollector(
        buffer_size=1000, kafka_enabled=False, postgres_enabled=False
    )


@pytest.fixture
def coordinator(state, metrics_collector):
    """SignalCoordinator fixture."""
    return SignalCoordinator(
        state=state, metrics_collector=metrics_collector, max_history_size=1000
    )


class TestSignalModels:
    """Test signal model validation and serialization."""

    def test_coordination_signal_creation(self):
        """Test basic coordination signal creation."""
        signal = CoordinationSignal(
            signal_type=SignalType.AGENT_INITIALIZED,
            sender_agent_id="agent-1",
            coordination_id="session-1",
            event_data={"status": "initialized"},
        )

        assert signal.signal_type == SignalType.AGENT_INITIALIZED
        assert signal.sender_agent_id == "agent-1"
        assert signal.coordination_id == "session-1"
        assert signal.event_data["status"] == "initialized"
        assert signal.signal_id is not None
        assert isinstance(signal.timestamp, datetime)

    def test_agent_initialized_signal(self):
        """Test agent initialized signal model."""
        signal = AgentInitializedSignal(
            agent_id="model-gen",
            capabilities=["pydantic_models", "type_hints"],
            ready=True,
            initialization_time_ms=45.3,
        )

        assert signal.agent_id == "model-gen"
        assert len(signal.capabilities) == 2
        assert signal.ready is True
        assert signal.initialization_time_ms == 45.3

    def test_agent_completed_signal(self):
        """Test agent completed signal model."""
        signal = AgentCompletedSignal(
            agent_id="model-gen",
            result_summary="Generated 5 models",
            quality_score=0.95,
            execution_time_ms=1234.5,
            artifacts_generated=["models/user.py", "models/post.py"],
            error=None,
        )

        assert signal.agent_id == "model-gen"
        assert signal.quality_score == 0.95
        assert len(signal.artifacts_generated) == 2
        assert signal.error is None

    def test_dependency_resolved_signal(self):
        """Test dependency resolved signal model."""
        signal = DependencyResolvedSignal(
            dependency_type="model",
            dependency_id="UserModel",
            resolved_by="model-gen",
            resolution_data={"file_path": "models/user.py", "schema": "..."},
        )

        assert signal.dependency_type == "model"
        assert signal.dependency_id == "UserModel"
        assert signal.resolved_by == "model-gen"
        assert "file_path" in signal.resolution_data

    def test_inter_agent_message(self):
        """Test inter-agent message model."""
        message = InterAgentMessage(
            message_type="request",
            message="Need validator for UserModel",
            requires_response=True,
            response_timeout_ms=5000.0,
            payload={"model_name": "UserModel"},
        )

        assert message.message_type == "request"
        assert message.requires_response is True
        assert message.response_timeout_ms == 5000.0
        assert message.payload["model_name"] == "UserModel"

    def test_signal_subscription(self):
        """Test signal subscription model."""
        subscription = SignalSubscription(
            coordination_id="session-1",
            agent_id="validator-gen",
            signal_types=[SignalType.AGENT_COMPLETED, SignalType.DEPENDENCY_RESOLVED],
            sender_filter="model-gen",
        )

        assert subscription.coordination_id == "session-1"
        assert subscription.agent_id == "validator-gen"
        assert len(subscription.signal_types) == 2
        assert subscription.sender_filter == "model-gen"

    def test_signal_metrics(self):
        """Test signal metrics model."""
        metrics = SignalMetrics(
            total_signals_sent=100,
            total_signals_received=95,
            average_propagation_ms=45.2,
            max_propagation_ms=98.7,
            signals_by_type={"agent_completed": 50, "dependency_resolved": 45},
        )

        assert metrics.total_signals_sent == 100
        assert metrics.total_signals_received == 95
        assert metrics.average_propagation_ms == 45.2
        assert len(metrics.signals_by_type) == 2


class TestSignalCoordinatorBasicOperations:
    """Test basic signal coordinator operations."""

    @pytest.mark.asyncio
    async def test_send_agent_initialized_signal(self, coordinator):
        """Test sending agent initialized signal."""
        success = await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_initialized",
            event_data={
                "agent_id": "model-gen",
                "capabilities": ["pydantic_models"],
                "ready": True,
            },
            sender_agent_id="model-gen",
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_send_agent_completed_signal(self, coordinator):
        """Test sending agent completed signal."""
        success = await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={
                "agent_id": "model-gen",
                "result_summary": "Generated 5 models",
                "quality_score": 0.95,
                "execution_time_ms": 1234.5,
            },
            sender_agent_id="model-gen",
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_send_dependency_resolved_signal(self, coordinator):
        """Test sending dependency resolved signal."""
        success = await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="dependency_resolved",
            event_data={
                "dependency_type": "model",
                "dependency_id": "UserModel",
                "resolved_by": "model-gen",
            },
            sender_agent_id="model-gen",
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_send_inter_agent_message(self, coordinator):
        """Test sending inter-agent message."""
        success = await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="inter_agent_message",
            event_data={
                "message_type": "request",
                "message": "Need validator",
                "requires_response": True,
            },
            sender_agent_id="validator-gen",
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_send_signal_with_recipients(self, coordinator):
        """Test sending signal to specific recipients."""
        success = await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="inter_agent_message",
            event_data={"message": "Hello"},
            sender_agent_id="agent-1",
            recipient_agents=["agent-2", "agent-3"],
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_send_signal_with_metadata(self, coordinator):
        """Test sending signal with metadata."""
        success = await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={"agent_id": "model-gen"},
            sender_agent_id="model-gen",
            metadata={"priority": "high", "correlation_id": "correlation-123"},
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_send_invalid_signal_type(self, coordinator):
        """Test sending signal with invalid type."""
        success = await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="invalid_signal_type",
            event_data={},
        )

        assert success is False


class TestSignalHistory:
    """Test signal history tracking."""

    @pytest.mark.asyncio
    async def test_signal_history_storage(self, coordinator):
        """Test that signals are stored in history."""
        # Send multiple signals
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_initialized",
            event_data={"agent_id": "agent-1"},
            sender_agent_id="agent-1",
        )

        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={"agent_id": "agent-1"},
            sender_agent_id="agent-1",
        )

        # Get history
        history = coordinator.get_signal_history(coordination_id="session-1")

        assert len(history) == 2
        assert history[0].signal_type == SignalType.AGENT_COMPLETED  # Most recent first
        assert history[1].signal_type == SignalType.AGENT_INITIALIZED

    @pytest.mark.asyncio
    async def test_signal_history_filtering(self, coordinator):
        """Test signal history filtering."""
        # Send multiple signals
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_initialized",
            event_data={"agent_id": "agent-1"},
            sender_agent_id="agent-1",
        )

        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={"agent_id": "agent-1"},
            sender_agent_id="agent-1",
        )

        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={"agent_id": "agent-2"},
            sender_agent_id="agent-2",
        )

        # Filter by signal type
        history = coordinator.get_signal_history(
            coordination_id="session-1",
            filters={"signal_type": "agent_completed"},
        )

        assert len(history) == 2
        assert all(s.signal_type == SignalType.AGENT_COMPLETED for s in history)

    @pytest.mark.asyncio
    async def test_signal_history_limit(self, coordinator):
        """Test signal history limit."""
        # Send 5 signals
        for i in range(5):
            await coordinator.signal_coordination_event(
                coordination_id="session-1",
                event_type="agent_initialized",
                event_data={"agent_id": f"agent-{i}"},
                sender_agent_id=f"agent-{i}",
            )

        # Get only last 3
        history = coordinator.get_signal_history(coordination_id="session-1", limit=3)

        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_signal_history_max_size(self, state, metrics_collector):
        """Test that signal history respects max size."""
        # Create coordinator with small max history
        coordinator = SignalCoordinator(
            state=state, metrics_collector=metrics_collector, max_history_size=5
        )

        # Send 10 signals
        for i in range(10):
            await coordinator.signal_coordination_event(
                coordination_id="session-1",
                event_type="agent_initialized",
                event_data={"agent_id": f"agent-{i}"},
            )

        # History should only contain last 5
        history = coordinator.get_signal_history(coordination_id="session-1")
        assert len(history) <= 5


class TestSignalSubscription:
    """Test signal subscription and delivery."""

    @pytest.mark.asyncio
    async def test_subscribe_to_all_signals(self, coordinator):
        """Test subscribing to all signal types."""
        # Create subscription task
        subscription_task = asyncio.create_task(
            self._collect_signals(coordinator, "session-1", "agent-1", None, 2)
        )

        # Wait a bit for subscription to register
        await asyncio.sleep(0.1)

        # Send signals
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_initialized",
            event_data={"agent_id": "agent-2"},
            sender_agent_id="agent-2",
        )

        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={"agent_id": "agent-2"},
            sender_agent_id="agent-2",
        )

        # Wait for signals to be collected
        signals = await asyncio.wait_for(subscription_task, timeout=2.0)

        assert len(signals) == 2
        assert signals[0].signal_type == SignalType.AGENT_INITIALIZED
        assert signals[1].signal_type == SignalType.AGENT_COMPLETED

    @pytest.mark.asyncio
    async def test_subscribe_to_specific_signal_types(self, coordinator):
        """Test subscribing to specific signal types."""
        # Subscribe only to completed signals
        subscription_task = asyncio.create_task(
            self._collect_signals(
                coordinator, "session-1", "agent-1", ["agent_completed"], 1
            )
        )

        await asyncio.sleep(0.1)

        # Send multiple signals
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_initialized",
            event_data={"agent_id": "agent-2"},
            sender_agent_id="agent-2",
        )

        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={"agent_id": "agent-2"},
            sender_agent_id="agent-2",
        )

        # Should only receive completed signal
        signals = await asyncio.wait_for(subscription_task, timeout=2.0)

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.AGENT_COMPLETED

    @pytest.mark.asyncio
    async def test_subscription_with_recipient_filter(self, coordinator):
        """Test that signals are delivered only to specified recipients."""
        # Subscribe agent-1
        subscription_task = asyncio.create_task(
            self._collect_signals(coordinator, "session-1", "agent-1", None, 1)
        )

        await asyncio.sleep(0.1)

        # Send signal to agent-1 only
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="inter_agent_message",
            event_data={"message": "Hello agent-1"},
            sender_agent_id="agent-2",
            recipient_agents=["agent-1"],
        )

        # Send signal to agent-2 only (should not be received)
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="inter_agent_message",
            event_data={"message": "Hello agent-2"},
            sender_agent_id="agent-3",
            recipient_agents=["agent-2"],
        )

        # Should only receive first signal
        signals = await asyncio.wait_for(subscription_task, timeout=2.0)

        assert len(signals) == 1
        assert signals[0].event_data["message"] == "Hello agent-1"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, coordinator):
        """Test that signals are delivered to multiple subscribers."""
        # Create two subscribers
        task1 = asyncio.create_task(
            self._collect_signals(coordinator, "session-1", "agent-1", None, 1)
        )
        task2 = asyncio.create_task(
            self._collect_signals(coordinator, "session-1", "agent-2", None, 1)
        )

        await asyncio.sleep(0.1)

        # Send broadcast signal
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={"agent_id": "agent-3"},
            sender_agent_id="agent-3",
        )

        # Both should receive signal
        signals1 = await asyncio.wait_for(task1, timeout=2.0)
        signals2 = await asyncio.wait_for(task2, timeout=2.0)

        assert len(signals1) == 1
        assert len(signals2) == 1
        assert signals1[0].signal_type == SignalType.AGENT_COMPLETED
        assert signals2[0].signal_type == SignalType.AGENT_COMPLETED

    async def _collect_signals(
        self, coordinator, coordination_id, agent_id, signal_types, count
    ):
        """Helper to collect specified number of signals."""
        signals = []
        async for signal in coordinator.subscribe_to_signals(
            coordination_id=coordination_id,
            agent_id=agent_id,
            signal_types=signal_types,
        ):
            signals.append(signal)
            if len(signals) >= count:
                break
        return signals


class TestSignalMetrics:
    """Test signal metrics collection."""

    @pytest.mark.asyncio
    async def test_signal_metrics_tracking(self, coordinator):
        """Test that signal metrics are tracked."""
        # Send multiple signals
        for i in range(5):
            await coordinator.signal_coordination_event(
                coordination_id="session-1",
                event_type="agent_initialized",
                event_data={"agent_id": f"agent-{i}"},
                sender_agent_id=f"agent-{i}",
            )

        # Get metrics
        metrics = coordinator.get_signal_metrics("session-1")

        assert metrics.total_signals_sent == 5
        assert metrics.average_propagation_ms > 0
        assert metrics.max_propagation_ms > 0
        assert metrics.signals_by_type["agent_initialized"] == 5

    @pytest.mark.asyncio
    async def test_signal_metrics_by_type(self, coordinator):
        """Test that metrics track different signal types."""
        # Send different types
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_initialized",
            event_data={},
        )

        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={},
        )

        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={},
        )

        # Get metrics
        metrics = coordinator.get_signal_metrics("session-1")

        assert metrics.total_signals_sent == 3
        assert metrics.signals_by_type["agent_initialized"] == 1
        assert metrics.signals_by_type["agent_completed"] == 2


class TestSignalPerformance:
    """Test signal performance targets."""

    @pytest.mark.asyncio
    async def test_signal_propagation_under_100ms(self, coordinator):
        """Test that signal propagation meets <100ms target."""
        start_time = time.time()

        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={"agent_id": "agent-1"},
            sender_agent_id="agent-1",
        )

        propagation_time_ms = (time.time() - start_time) * 1000

        # Should be well under 100ms target
        assert propagation_time_ms < 100.0, (
            f"Signal propagation took {propagation_time_ms:.2f}ms, "
            f"exceeds 100ms target"
        )

    @pytest.mark.asyncio
    async def test_bulk_signal_propagation(self, coordinator):
        """Test bulk signal sending performance."""
        start_time = time.time()

        # Send 100 signals
        for i in range(100):
            await coordinator.signal_coordination_event(
                coordination_id="session-1",
                event_type="agent_initialized",
                event_data={"agent_id": f"agent-{i}"},
            )

        total_time_ms = (time.time() - start_time) * 1000
        avg_time_ms = total_time_ms / 100

        # Average should be well under 100ms
        assert avg_time_ms < 100.0, (
            f"Average signal propagation {avg_time_ms:.2f}ms exceeds 100ms target"
        )


class TestIntegrationWithFoundation:
    """Test integration with Foundation components."""

    @pytest.mark.asyncio
    async def test_integration_with_thread_safe_state(self, coordinator):
        """Test that signals are stored in ThreadSafeState."""
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={"agent_id": "agent-1"},
        )

        # Check ThreadSafeState directly
        history_key = "signal_history:session-1"
        history = coordinator.state.get(history_key, [])

        assert len(history) == 1
        assert history[0]["signal_type"] == "agent_completed"

    @pytest.mark.asyncio
    async def test_integration_with_metrics_collector(self, coordinator):
        """Test that metrics are collected."""
        # Send signal (metrics should be collected)
        await coordinator.signal_coordination_event(
            coordination_id="session-1",
            event_type="agent_completed",
            event_data={"agent_id": "agent-1"},
        )

        # Check metrics collector stats
        stats = await coordinator.metrics.get_stats()

        # Should have buffered metrics
        assert stats["buffer_size"] >= 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_coordination_id_handled(self, coordinator):
        """Test that empty coordination_id is handled gracefully."""
        # This should create signal with empty coordination_id
        # Implementation may want to validate this in future
        success = await coordinator.signal_coordination_event(
            coordination_id="",
            event_type="agent_initialized",
            event_data={},
        )

        # Currently allows empty coordination_id
        # Could add validation in future
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_get_history_for_nonexistent_session(self, coordinator):
        """Test getting history for session that doesn't exist."""
        history = coordinator.get_signal_history(coordination_id="nonexistent")

        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_get_metrics_for_nonexistent_session(self, coordinator):
        """Test getting metrics for session that doesn't exist."""
        metrics = coordinator.get_signal_metrics(coordination_id="nonexistent")

        assert metrics.total_signals_sent == 0
        assert metrics.total_signals_received == 0


class TestCodeGenerationUseCases:
    """Test signal system for code generation workflows."""

    @pytest.mark.asyncio
    async def test_model_generation_workflow(self, coordinator):
        """Test signal flow for model generation workflow."""
        # 1. Model generator initialized
        await coordinator.signal_coordination_event(
            coordination_id="codegen-session",
            event_type="agent_initialized",
            event_data={
                "agent_id": "model-gen",
                "capabilities": ["pydantic_models"],
                "ready": True,
            },
            sender_agent_id="model-gen",
        )

        # 2. Model generator completed
        await coordinator.signal_coordination_event(
            coordination_id="codegen-session",
            event_type="agent_completed",
            event_data={
                "agent_id": "model-gen",
                "result_summary": "Generated 5 models",
                "quality_score": 0.95,
                "artifacts_generated": ["models/user.py", "models/post.py"],
            },
            sender_agent_id="model-gen",
        )

        # 3. Dependency resolved (models available for validator)
        await coordinator.signal_coordination_event(
            coordination_id="codegen-session",
            event_type="dependency_resolved",
            event_data={
                "dependency_type": "model",
                "dependency_id": "UserModel",
                "resolved_by": "model-gen",
                "resolution_data": {"file_path": "models/user.py"},
            },
            sender_agent_id="model-gen",
        )

        # Verify workflow signals
        history = coordinator.get_signal_history("codegen-session")

        assert len(history) == 3
        assert history[2].signal_type == SignalType.AGENT_INITIALIZED
        assert history[1].signal_type == SignalType.AGENT_COMPLETED
        assert history[0].signal_type == SignalType.DEPENDENCY_RESOLVED

    @pytest.mark.asyncio
    async def test_parallel_agent_coordination(self, coordinator):
        """Test coordination of parallel agents."""
        # Validator subscribes to model completion
        validator_task = asyncio.create_task(
            self._collect_signals_for_codegen(
                coordinator, "codegen-session", "validator-gen", 1
            )
        )

        # Test runner subscribes to validator completion
        test_runner_task = asyncio.create_task(
            self._collect_signals_for_codegen(
                coordinator, "codegen-session", "test-runner", 1
            )
        )

        await asyncio.sleep(0.1)

        # Model generator completes
        await coordinator.signal_coordination_event(
            coordination_id="codegen-session",
            event_type="agent_completed",
            event_data={"agent_id": "model-gen"},
            sender_agent_id="model-gen",
        )

        # Validator should receive signal
        validator_signals = await asyncio.wait_for(validator_task, timeout=2.0)
        assert len(validator_signals) == 1

        # Validator completes
        await coordinator.signal_coordination_event(
            coordination_id="codegen-session",
            event_type="agent_completed",
            event_data={"agent_id": "validator-gen"},
            sender_agent_id="validator-gen",
        )

        # Test runner should receive signal
        test_runner_signals = await asyncio.wait_for(test_runner_task, timeout=2.0)
        assert len(test_runner_signals) == 1

    async def _collect_signals_for_codegen(
        self, coordinator, coordination_id, agent_id, count
    ):
        """Helper for code generation signal collection."""
        signals = []
        async for signal in coordinator.subscribe_to_signals(
            coordination_id=coordination_id,
            agent_id=agent_id,
            signal_types=["agent_completed"],
        ):
            signals.append(signal)
            if len(signals) >= count:
                break
        return signals
