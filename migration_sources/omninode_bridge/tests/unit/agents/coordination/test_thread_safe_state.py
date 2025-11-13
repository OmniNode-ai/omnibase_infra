"""
Comprehensive unit tests for ThreadSafeState.

Test Coverage:
- Basic operations (get/set/delete/update)
- Thread safety with concurrent access
- Version tracking and rollback
- Change history tracking
- Snapshot functionality with caching
- Performance targets validation
- Error handling

Performance Targets:
- get(): <1ms per operation
- set(): <2ms per operation
- snapshot(): <5ms for 1000 keys
"""

import threading
import time
from datetime import datetime
from typing import Any

import pytest

from omninode_bridge.agents.coordination import (
    AgentCoordinationState,
    CodeGenerationState,
    StateChangeRecord,
    StateKeyError,
    StateRollbackError,
    StateVersionError,
    ThreadSafeState,
)


class TestThreadSafeStateBasicOperations:
    """Test basic get/set/delete/update operations."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        state = ThreadSafeState[int]()

        state.set("counter", 42)
        assert state.get("counter") == 42

    def test_get_with_default(self):
        """Test get with default value."""
        state = ThreadSafeState[int]()

        assert state.get("nonexistent", default=0) == 0

    def test_get_none_returns_default(self):
        """Test that None values return default."""
        state = ThreadSafeState[Any]()

        state.set("value", None)
        # None value should return default
        result = state.get("value", default=42)
        assert result == 42

    def test_delete(self):
        """Test delete operation."""
        state = ThreadSafeState[str]()

        state.set("key", "value")
        assert state.delete("key") is True
        assert state.get("key") is None

    def test_delete_nonexistent_returns_false(self):
        """Test deleting non-existent key returns False."""
        state = ThreadSafeState[str]()

        assert state.delete("nonexistent") is False

    def test_update_batch(self):
        """Test batch update operation."""
        state = ThreadSafeState[int]()

        state.update({"a": 1, "b": 2, "c": 3})

        assert state.get("a") == 1
        assert state.get("b") == 2
        assert state.get("c") == 3

    def test_contains(self):
        """Test __contains__ operator."""
        state = ThreadSafeState[str]()

        state.set("key", "value")
        assert "key" in state
        assert "nonexistent" not in state

    def test_len(self):
        """Test __len__ operator."""
        state = ThreadSafeState[int]()

        assert len(state) == 0

        state.update({"a": 1, "b": 2, "c": 3})
        assert len(state) == 3

    def test_has(self):
        """Test has() method."""
        state = ThreadSafeState[str]()

        state.set("key", "value")
        assert state.has("key") is True
        assert state.has("nonexistent") is False

    def test_keys(self):
        """Test keys() method."""
        state = ThreadSafeState[int]()

        state.update({"a": 1, "b": 2, "c": 3})
        keys = state.keys()

        assert len(keys) == 3
        assert "a" in keys
        assert "b" in keys
        assert "c" in keys

    def test_clear(self):
        """Test clear operation."""
        state = ThreadSafeState[int]()

        state.update({"a": 1, "b": 2, "c": 3})
        assert len(state) == 3

        state.clear(changed_by="test")
        assert len(state) == 0

    def test_repr(self):
        """Test __repr__ method."""
        state = ThreadSafeState[int]()

        state.update({"a": 1, "b": 2})

        repr_str = repr(state)
        assert "ThreadSafeState" in repr_str
        assert "version=" in repr_str
        assert "keys=2" in repr_str


class TestThreadSafeStateDataIsolation:
    """Test data isolation through deep copy."""

    def test_get_returns_copy(self):
        """Test that get() returns a deep copy."""
        state = ThreadSafeState[dict[str, int]]()

        original = {"nested": 42}
        state.set("data", original)

        # Modify returned value
        retrieved = state.get("data")
        retrieved["nested"] = 99

        # Original state should be unchanged
        assert state.get("data")["nested"] == 42

    def test_set_stores_copy(self):
        """Test that set() stores a deep copy."""
        state = ThreadSafeState[dict[str, int]]()

        data = {"nested": 42}
        state.set("data", data)

        # Modify original
        data["nested"] = 99

        # State should have original value
        assert state.get("data")["nested"] == 42

    def test_update_stores_copies(self):
        """Test that update() stores deep copies."""
        state = ThreadSafeState[dict[str, int]]()

        data1 = {"nested": 1}
        data2 = {"nested": 2}
        state.update({"a": data1, "b": data2})

        # Modify originals
        data1["nested"] = 99
        data2["nested"] = 99

        # State should have original values
        assert state.get("a")["nested"] == 1
        assert state.get("b")["nested"] == 2


class TestThreadSafeStateVersioning:
    """Test version tracking and rollback."""

    def test_version_increments_on_set(self):
        """Test version increments on each set."""
        state = ThreadSafeState[int]()

        assert state.get_version() == 0

        state.set("key", 1)
        assert state.get_version() == 1

        state.set("key", 2)
        assert state.get_version() == 2

    def test_version_increments_on_update(self):
        """Test version increments once for batch update."""
        state = ThreadSafeState[int]()

        state.update({"a": 1, "b": 2, "c": 3})
        assert state.get_version() == 1  # Single increment

    def test_version_increments_on_delete(self):
        """Test version increments on delete."""
        state = ThreadSafeState[str]()

        state.set("key", "value")
        assert state.get_version() == 1

        state.delete("key")
        assert state.get_version() == 2

    def test_version_increments_on_clear(self):
        """Test version increments on clear."""
        state = ThreadSafeState[int]()

        state.update({"a": 1, "b": 2})
        initial_version = state.get_version()

        state.clear()
        assert state.get_version() == initial_version + 1

    def test_rollback_to_previous_version(self):
        """Test rollback to previous version."""
        state = ThreadSafeState[int]()

        state.set("counter", 1)
        state.set("counter", 2)
        state.set("counter", 3)

        state.rollback(1)  # Rollback to version 1
        assert state.get("counter") == 1
        assert state.get_version() == 1

    def test_rollback_multiple_keys(self):
        """Test rollback with multiple keys."""
        state = ThreadSafeState[int]()

        state.set("a", 1)
        state.set("b", 2)
        version = state.get_version()

        state.set("a", 10)
        state.set("b", 20)

        state.rollback(version)
        assert state.get("a") == 1
        assert state.get("b") == 2

    def test_rollback_restores_deleted_keys(self):
        """Test rollback restores deleted keys."""
        state = ThreadSafeState[str]()

        state.set("key", "value")
        version = state.get_version()

        state.delete("key")

        state.rollback(version)
        assert state.get("key") == "value"

    def test_rollback_removes_new_keys(self):
        """Test rollback removes keys that didn't exist."""
        state = ThreadSafeState[int]()

        state.set("a", 1)
        version = state.get_version()

        state.set("b", 2)  # New key
        assert "b" in state

        state.rollback(version)
        assert "b" not in state
        assert state.get("a") == 1

    def test_rollback_invalid_version_raises_error(self):
        """Test rollback to invalid version raises error."""
        state = ThreadSafeState[int]()

        state.set("counter", 1)

        with pytest.raises(StateVersionError):
            state.rollback(10)  # Future version

    def test_rollback_negative_version_raises_error(self):
        """Test rollback to negative version raises error."""
        state = ThreadSafeState[int]()

        state.set("counter", 1)

        with pytest.raises(StateVersionError):
            state.rollback(-1)


class TestThreadSafeStateHistory:
    """Test change history tracking."""

    def test_history_records_changes(self):
        """Test history records all changes."""
        state = ThreadSafeState[int](max_history_size=100)

        state.set("counter", 1, changed_by="agent-1")
        state.set("counter", 2, changed_by="agent-2")

        history = state.get_history()
        assert len(history) == 2
        assert history[0].changed_by == "agent-2"  # Most recent first
        assert history[1].changed_by == "agent-1"

    def test_history_filters_by_key(self):
        """Test history filtering by key."""
        state = ThreadSafeState[Any]()

        state.set("counter", 1)
        state.set("status", "active")
        state.set("counter", 2)

        counter_history = state.get_history(key="counter")
        assert len(counter_history) == 2
        assert all(record.key == "counter" for record in counter_history)

    def test_history_limits_results(self):
        """Test history limit parameter."""
        state = ThreadSafeState[int]()

        for i in range(10):
            state.set(f"key_{i}", i)

        history = state.get_history(limit=5)
        assert len(history) == 5

    def test_history_limits_size(self):
        """Test history size limiting."""
        state = ThreadSafeState[int](max_history_size=10)

        # Add 20 changes
        for i in range(20):
            state.set(f"key_{i}", i)

        # Should only keep last 10
        history = state.get_history()
        assert len(history) <= 10

    def test_history_records_operation_types(self):
        """Test history records operation types."""
        state = ThreadSafeState[int]()

        state.set("key", 1)
        state.update({"key": 2})
        state.delete("key")

        history = state.get_history()
        operations = [record.operation for record in history]

        assert "set" in operations
        assert "update" in operations
        assert "delete" in operations

    def test_history_records_old_and_new_values(self):
        """Test history records old and new values."""
        state = ThreadSafeState[int]()

        state.set("counter", 1)
        state.set("counter", 2)

        history = state.get_history(key="counter")
        # Most recent first
        assert history[0].old_value == 1
        assert history[0].new_value == 2
        assert history[1].old_value is None
        assert history[1].new_value == 1

    def test_clear_history(self):
        """Test clearing history."""
        state = ThreadSafeState[int]()

        state.set("key", 1)
        state.set("key", 2)

        assert len(state.get_history()) == 2

        state.clear_history()
        assert len(state.get_history()) == 0


class TestThreadSafeStateSnapshot:
    """Test snapshot functionality."""

    def test_snapshot_creates_immutable_copy(self):
        """Test snapshot is immutable."""
        state = ThreadSafeState[int]()

        state.set("counter", 1)
        snapshot = state.snapshot()

        # Modify state
        state.set("counter", 2)

        # Snapshot should be unchanged
        assert snapshot["counter"] == 1
        assert state.get("counter") == 2

    def test_snapshot_caching(self):
        """Test snapshot caching optimization."""
        state = ThreadSafeState[int]()

        state.set("counter", 1)

        snapshot1 = state.snapshot()
        snapshot2 = state.snapshot()

        # Should return cached snapshot (same values)
        assert snapshot1 == snapshot2
        assert snapshot1["counter"] == snapshot2["counter"]

        metrics = state.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 1

    def test_snapshot_cache_invalidation(self):
        """Test snapshot cache is invalidated on state change."""
        state = ThreadSafeState[int]()

        state.set("counter", 1)
        snapshot1 = state.snapshot()

        # Modify state
        state.set("counter", 2)

        # New snapshot should be different
        snapshot2 = state.snapshot()
        assert snapshot1 is not snapshot2
        assert snapshot1["counter"] == 1
        assert snapshot2["counter"] == 2

    def test_snapshot_with_disabled_caching(self):
        """Test snapshot without caching."""
        state = ThreadSafeState[int](enable_snapshots=False)

        state.set("counter", 1)

        snapshot1 = state.snapshot()
        snapshot2 = state.snapshot()

        # Should be same values but no cache hits
        assert snapshot1 == snapshot2

        metrics = state.get_metrics()
        assert metrics["cache_hits"] == 0  # No caching

    def test_snapshot_with_complex_data(self):
        """Test snapshot with nested data structures."""
        state = ThreadSafeState[dict[str, Any]]()

        complex_data = {"nested": {"level2": {"level3": 42}}, "list": [1, 2, 3]}
        state.set("data", complex_data)

        snapshot = state.snapshot()

        # Modify nested structure
        state.get("data")["nested"]["level2"]["level3"] = 99

        # Snapshot should be unchanged
        assert snapshot["data"]["nested"]["level2"]["level3"] == 42


class TestThreadSafeStateConcurrency:
    """Test thread safety under concurrent access."""

    def test_concurrent_writes_are_thread_safe(self):
        """Test concurrent writes don't corrupt state."""
        state = ThreadSafeState[int]()
        state.set("counter", 0)

        def increment_counter(iterations: int):
            for _ in range(iterations):
                current = state.get("counter", default=0)
                state.set("counter", current + 1)

        # Run 10 threads, each incrementing 100 times
        threads = [
            threading.Thread(target=increment_counter, args=(100,)) for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final value should be 1000 (10 threads * 100 increments)
        final = state.get("counter")
        assert final == 1000

    def test_concurrent_snapshots_are_consistent(self):
        """Test snapshots are consistent during concurrent modifications."""
        state = ThreadSafeState[int](initial_state={f"key_{i}": i for i in range(100)})

        snapshots = []

        def take_snapshots():
            for _ in range(10):
                snapshots.append(state.snapshot())

        def modify_state():
            for i in range(10):
                state.set(f"key_{i}", i * 2)

        # Run snapshot and modify threads concurrently
        t1 = threading.Thread(target=take_snapshots)
        t2 = threading.Thread(target=modify_state)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # All snapshots should be internally consistent
        for snapshot in snapshots:
            assert len(snapshot) == 100  # No partial snapshots

    def test_concurrent_mixed_operations(self):
        """Test concurrent mixed read/write operations."""
        state = ThreadSafeState[int](initial_state={"shared": 0})

        def reader(iterations: int):
            for _ in range(iterations):
                _ = state.get("shared")
                _ = state.snapshot()

        def writer(iterations: int):
            for i in range(iterations):
                state.set("shared", i)

        # Run multiple readers and writers
        readers = [threading.Thread(target=reader, args=(100,)) for _ in range(5)]
        writers = [threading.Thread(target=writer, args=(50,)) for _ in range(2)]

        all_threads = readers + writers
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        # Should complete without deadlock or corruption
        assert state.get("shared") is not None


class TestThreadSafeStatePerformance:
    """Test performance targets."""

    @pytest.mark.performance
    def test_get_performance_target(self):
        """Test get() meets <1ms target."""
        state = ThreadSafeState[int](initial_state={"key": 42})

        iterations = 10000

        start = time.perf_counter()
        for _ in range(iterations):
            _ = state.get("key")
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 1.0, f"get() took {elapsed:.3f}ms (target: <1ms)"

    @pytest.mark.performance
    def test_set_performance_target(self):
        """Test set() meets <2ms target."""
        state = ThreadSafeState[int]()

        iterations = 10000

        start = time.perf_counter()
        for i in range(iterations):
            state.set(f"key_{i % 100}", i)  # Reuse 100 keys
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 2.0, f"set() took {elapsed:.3f}ms (target: <2ms)"

    @pytest.mark.performance
    def test_snapshot_performance_target(self):
        """Test snapshot() meets <5ms target for 1000 keys."""
        state = ThreadSafeState[int](initial_state={f"key_{i}": i for i in range(1000)})

        start = time.perf_counter()
        _ = state.snapshot()
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 5.0, f"snapshot() took {elapsed:.3f}ms (target: <5ms)"

    @pytest.mark.performance
    def test_update_performance(self):
        """Test batch update performance."""
        state = ThreadSafeState[int]()

        updates = {f"key_{i}": i for i in range(100)}

        start = time.perf_counter()
        state.update(updates)
        elapsed = (time.perf_counter() - start) * 1000

        # Should be faster than 100 individual sets
        assert elapsed < 100, f"update() took {elapsed:.3f}ms for 100 keys"


class TestThreadSafeStateErrorHandling:
    """Test error handling."""

    def test_get_required_raises_on_missing_key(self):
        """Test get_required() raises StateKeyError."""
        state = ThreadSafeState[int]()

        with pytest.raises(StateKeyError) as exc_info:
            state.get_required("nonexistent")

        assert exc_info.value.key == "nonexistent"
        assert isinstance(exc_info.value.available_keys, list)

    def test_get_required_returns_value_when_present(self):
        """Test get_required() returns value when key exists."""
        state = ThreadSafeState[int]()

        state.set("key", 42)
        value = state.get_required("key")

        assert value == 42

    def test_rollback_to_future_version_raises_error(self):
        """Test rollback to future version raises error."""
        state = ThreadSafeState[int]()
        state.set("counter", 1)

        with pytest.raises(StateVersionError) as exc_info:
            state.rollback(100)  # Future version

        assert exc_info.value.current_version == 1
        assert exc_info.value.target_version == 100

    def test_rollback_without_history_raises_error(self):
        """Test rollback without sufficient history raises error."""
        state = ThreadSafeState[int](max_history_size=5)

        # Create more changes than history can hold
        for i in range(20):
            state.set("counter", i)

        # Try to rollback to version 1 (beyond history)
        with pytest.raises(StateRollbackError) as exc_info:
            state.rollback(1)

        assert exc_info.value.target_version == 1


class TestThreadSafeStateMetrics:
    """Test performance metrics collection."""

    def test_metrics_track_operations(self):
        """Test metrics track all operations."""
        state = ThreadSafeState[int]()

        state.set("key1", 1)
        state.set("key2", 2)
        _ = state.get("key1")
        _ = state.get("key2")
        _ = state.snapshot()

        metrics = state.get_metrics()

        assert metrics["set_count"] == 2
        assert metrics["get_count"] == 2
        assert metrics["snapshot_count"] == 1

    def test_metrics_track_cache_performance(self):
        """Test metrics track cache hits/misses."""
        state = ThreadSafeState[int]()

        state.set("key", 1)

        _ = state.snapshot()  # Miss
        _ = state.snapshot()  # Hit

        metrics = state.get_metrics()

        assert metrics["cache_misses"] == 1
        assert metrics["cache_hits"] == 1


class TestStateChangeRecord:
    """Test StateChangeRecord model."""

    def test_state_change_record_is_immutable(self):
        """Test StateChangeRecord is immutable."""
        record = StateChangeRecord(
            timestamp=datetime.utcnow(),
            version=1,
            key="test",
            operation="set",
            old_value=None,
            new_value=42,
            changed_by="agent-1",
        )

        with pytest.raises(Exception):  # ValidationError or similar
            record.version = 2  # Should raise error

    def test_state_change_record_validates_operation(self):
        """Test StateChangeRecord validates operation type."""
        with pytest.raises(Exception):  # ValidationError
            StateChangeRecord(
                timestamp=datetime.utcnow(),
                version=1,
                key="test",
                operation="invalid_op",  # Invalid operation
                old_value=None,
                new_value=42,
            )


class TestSpecializedStates:
    """Test specialized state models."""

    def test_agent_coordination_state(self):
        """Test AgentCoordinationState model."""
        state_data = AgentCoordinationState(
            session_id="test-session",
            agent_count=3,
            completed_agents=["agent-1", "agent-2"],
            shared_context={"key": "value"},
            generated_files=["file1.py", "file2.py"],
            current_phase="generation",
        )

        assert state_data.session_id == "test-session"
        assert state_data.agent_count == 3
        assert len(state_data.completed_agents) == 2
        assert state_data.current_phase == "generation"

    def test_code_generation_state(self):
        """Test CodeGenerationState model."""
        state_data = CodeGenerationState(
            workflow_id="workflow-123",
            contracts_parsed={"UserContract": {}},
            generated_models={"User": "models/user.py"},
            generated_validators={"UserValidator": "validators/user.py"},
            generated_tests={"test_user": "tests/test_user.py"},
            type_registry={"UserID": "str"},
            import_statements=["from typing import Optional"],
            quality_metrics={"coverage": 0.95},
        )

        assert state_data.workflow_id == "workflow-123"
        assert "UserContract" in state_data.contracts_parsed
        assert "User" in state_data.generated_models
        assert state_data.quality_metrics["coverage"] == 0.95

    def test_specialized_state_in_thread_safe_state(self):
        """Test using specialized state with ThreadSafeState."""
        state = ThreadSafeState[dict[str, Any]](
            initial_state=AgentCoordinationState(
                session_id="test", agent_count=2
            ).model_dump()
        )

        session_id = state.get("session_id")
        assert session_id == "test"

        agent_count = state.get("agent_count")
        assert agent_count == 2


class TestThreadSafeStateInitialization:
    """Test initialization and configuration."""

    def test_init_with_initial_state(self):
        """Test initialization with initial state."""
        initial = {"key1": 1, "key2": 2}
        state = ThreadSafeState[int](initial_state=initial)

        assert state.get("key1") == 1
        assert state.get("key2") == 2

    def test_init_without_initial_state(self):
        """Test initialization without initial state."""
        state = ThreadSafeState[int]()

        assert len(state) == 0
        assert state.get_version() == 0

    def test_init_with_custom_history_size(self):
        """Test initialization with custom history size."""
        state = ThreadSafeState[int](max_history_size=50)

        # Add 100 changes
        for i in range(100):
            state.set("key", i)

        # Should only keep 50
        history = state.get_history()
        assert len(history) <= 50

    def test_init_with_disabled_snapshots(self):
        """Test initialization with snapshots disabled."""
        state = ThreadSafeState[int](enable_snapshots=False)

        state.set("key", 1)

        snapshot1 = state.snapshot()
        snapshot2 = state.snapshot()

        # Should be different objects (no caching)
        assert snapshot1 is not snapshot2

        metrics = state.get_metrics()
        assert metrics["cache_hits"] == 0
