# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for NodeContractRegistryReducer.

Tests cover:
- Contract registration event handling
- Contract deregistration event handling
- Heartbeat event handling (last_seen_at updates)
- Runtime tick staleness computation
- Idempotency/dedupe behavior
- Topic extraction from contract_yaml

Related:
    - NodeContractRegistryReducer: Declarative reducer node
    - ContractRegistryReducer: Pure reducer implementation
    - ModelContractRegistryState: Immutable state model
    - OMN-1653: Contract Registry Reducer implementation ticket
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import yaml
from pydantic import ValidationError

from omnibase_core.enums import EnumDeregistrationReason
from omnibase_core.models.events import (
    ModelContractDeregisteredEvent,
    ModelContractRegisteredEvent,
    ModelNodeHeartbeatEvent,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.nodes.contract_registry_reducer.models.model_contract_registry_state import (
    ModelContractRegistryState,
)
from omnibase_infra.nodes.contract_registry_reducer.reducer import (
    STALENESS_THRESHOLD,
    ContractRegistryReducer,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reducer() -> ContractRegistryReducer:
    """Create a fresh ContractRegistryReducer instance."""
    return ContractRegistryReducer()


@pytest.fixture
def initial_state() -> ModelContractRegistryState:
    """Create an initial contract registry state."""
    return ModelContractRegistryState()


@pytest.fixture
def sample_version() -> ModelSemVer:
    """Create a sample semantic version."""
    return ModelSemVer(major=1, minor=0, patch=0)


@pytest.fixture
def sample_contract_yaml_dict() -> dict:
    """Create a sample contract_yaml dict with topics."""
    return {
        "name": "test-reducer",
        "version": {"major": 1, "minor": 0, "patch": 0},
        "type": "reducer",
        "consumed_events": [
            {"topic": "onex.evt.platform.test-event.v1", "event_type": "TestEvent"},
        ],
        "published_events": [
            {"topic": "onex.evt.platform.output-event.v1", "event_type": "OutputEvent"},
        ],
    }


@pytest.fixture
def sample_contract_yaml(sample_contract_yaml_dict: dict) -> str:
    """Create a sample contract_yaml as YAML string."""
    return yaml.dump(sample_contract_yaml_dict)


@pytest.fixture
def contract_registered_event(
    sample_version: ModelSemVer, sample_contract_yaml: str
) -> ModelContractRegisteredEvent:
    """Create a sample contract registered event."""
    return ModelContractRegisteredEvent(
        event_id=uuid4(),
        correlation_id=uuid4(),
        timestamp=datetime.now(UTC),
        source_node_id=uuid4(),
        event_type="onex.evt.platform.contract-registered.v1",
        node_name="test-reducer",
        node_version=sample_version,
        contract_hash="abc123",
        contract_yaml=sample_contract_yaml,
    )


@pytest.fixture
def contract_deregistered_event(
    sample_version: ModelSemVer,
) -> ModelContractDeregisteredEvent:
    """Create a sample contract deregistered event."""
    return ModelContractDeregisteredEvent(
        event_id=uuid4(),
        correlation_id=uuid4(),
        timestamp=datetime.now(UTC),
        source_node_id=uuid4(),
        event_type="onex.evt.platform.contract-deregistered.v1",
        node_name="test-reducer",
        node_version=sample_version,
        reason=EnumDeregistrationReason.SHUTDOWN,
    )


@pytest.fixture
def heartbeat_event(sample_version: ModelSemVer) -> ModelNodeHeartbeatEvent:
    """Create a sample node heartbeat event."""
    return ModelNodeHeartbeatEvent(
        event_id=uuid4(),
        correlation_id=uuid4(),
        timestamp=datetime.now(UTC),
        source_node_id=uuid4(),
        event_type="onex.evt.platform.node-heartbeat.v1",
        node_name="test-reducer",
        node_version=sample_version,
        sequence_number=1,
        uptime_seconds=60.0,
        contract_hash="abc123",
    )


@pytest.fixture
def runtime_tick_event() -> ModelRuntimeTick:
    """Create a sample runtime tick event."""
    now = datetime.now(UTC)
    return ModelRuntimeTick(
        now=now,
        tick_id=uuid4(),
        sequence_number=1,
        scheduled_at=now,
        correlation_id=uuid4(),
        scheduler_id="test-scheduler",
        tick_interval_ms=1000,
    )


def make_event_metadata(
    topic: str = "test.topic", partition: int = 0, offset: int = 1
) -> dict[str, object]:
    """Create event metadata dict for Kafka position tracking."""
    return {"topic": topic, "partition": partition, "offset": offset}


# =============================================================================
# Test: Idempotency - Duplicate Event Rejection
# =============================================================================


@pytest.mark.integration
class TestContractRegistryReducerIdempotency:
    """Test dedupe behavior for duplicate events."""

    def test_duplicate_event_returns_noop(
        self,
        reducer: ContractRegistryReducer,
        contract_registered_event: ModelContractRegisteredEvent,
    ) -> None:
        """Reducer should skip events already processed (same topic/partition/offset)."""
        # First, process an event to set the position
        state = ModelContractRegistryState()
        metadata = make_event_metadata(topic="test.topic", partition=0, offset=100)

        result1 = reducer.reduce(state, contract_registered_event, metadata)
        assert result1.items_processed == 1

        # Now try to process at same position - should be NOOP
        result2 = reducer.reduce(result1.result, contract_registered_event, metadata)
        assert result2.items_processed == 0
        assert len(result2.intents) == 0

    def test_new_event_is_processed(
        self,
        reducer: ContractRegistryReducer,
        contract_registered_event: ModelContractRegisteredEvent,
    ) -> None:
        """Reducer should process events with higher offset."""
        state = ModelContractRegistryState()

        # Process first event
        metadata1 = make_event_metadata(offset=100)
        result1 = reducer.reduce(state, contract_registered_event, metadata1)
        assert result1.items_processed == 1

        # Process new event at higher offset
        metadata2 = make_event_metadata(offset=101)
        result2 = reducer.reduce(result1.result, contract_registered_event, metadata2)
        assert result2.items_processed == 1
        assert result2.result.contracts_processed == 2


# =============================================================================
# Test: Contract Registration Event Handling
# =============================================================================


@pytest.mark.integration
class TestContractRegistryReducerRegistration:
    """Test contract registration event handling."""

    def test_contract_registered_emits_upsert_intent(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        contract_registered_event: ModelContractRegisteredEvent,
    ) -> None:
        """Registration event should emit postgres.upsert_contract intent."""
        result = reducer.reduce(
            initial_state, contract_registered_event, make_event_metadata()
        )

        # Should have at least upsert intent
        assert len(result.intents) >= 1

        upsert_intents = [
            i
            for i in result.intents
            if i.payload.intent_type == "postgres.upsert_contract"
        ]
        assert len(upsert_intents) == 1

        payload = upsert_intents[0].payload
        assert payload.node_name == "test-reducer"
        assert payload.version_major == 1
        assert payload.is_active is True

    def test_contract_registered_extracts_topics(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        contract_registered_event: ModelContractRegisteredEvent,
    ) -> None:
        """Registration event should extract topics from contract_yaml."""
        result = reducer.reduce(
            initial_state, contract_registered_event, make_event_metadata()
        )

        topic_intents = [
            i
            for i in result.intents
            if i.payload.intent_type == "postgres.update_topic"
        ]

        # Should have 2 topic intents (1 subscribe, 1 publish)
        assert len(topic_intents) == 2

        directions = {i.payload.direction for i in topic_intents}
        assert directions == {"subscribe", "publish"}

    def test_contract_registered_increments_counter(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        contract_registered_event: ModelContractRegisteredEvent,
    ) -> None:
        """State should track contracts_processed count."""
        assert initial_state.contracts_processed == 0

        result = reducer.reduce(
            initial_state, contract_registered_event, make_event_metadata()
        )

        assert result.result.contracts_processed == 1


# =============================================================================
# Test: Contract Deregistration Event Handling
# =============================================================================


@pytest.mark.integration
class TestContractRegistryReducerDeregistration:
    """Test contract deregistration event handling."""

    def test_contract_deregistered_emits_deactivate_intent(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        contract_deregistered_event: ModelContractDeregisteredEvent,
    ) -> None:
        """Deregistration event should emit deactivate and cleanup intents."""
        result = reducer.reduce(
            initial_state, contract_deregistered_event, make_event_metadata()
        )

        # Should emit 2 intents: deactivate + cleanup
        assert len(result.intents) == 2

        # Intent 1: Deactivate contract
        deactivate_payload = result.intents[0].payload
        assert deactivate_payload.intent_type == "postgres.deactivate_contract"
        assert deactivate_payload.node_name == "test-reducer"
        assert deactivate_payload.reason == "shutdown"

        # Intent 2: Cleanup topic references
        cleanup_payload = result.intents[1].payload
        assert cleanup_payload.intent_type == "postgres.cleanup_topic_references"
        assert cleanup_payload.node_name == "test-reducer"
        assert cleanup_payload.contract_id == "test-reducer:1.0.0"

    def test_contract_deregistered_increments_counter(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        contract_deregistered_event: ModelContractDeregisteredEvent,
    ) -> None:
        """State should track deregistrations_processed count."""
        result = reducer.reduce(
            initial_state, contract_deregistered_event, make_event_metadata()
        )

        assert result.result.deregistrations_processed == 1


# =============================================================================
# Test: Heartbeat Event Handling
# =============================================================================


@pytest.mark.integration
class TestContractRegistryReducerHeartbeat:
    """Test heartbeat event handling."""

    def test_heartbeat_emits_update_heartbeat_intent(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        heartbeat_event: ModelNodeHeartbeatEvent,
    ) -> None:
        """Heartbeat should emit postgres.update_heartbeat intent."""
        result = reducer.reduce(initial_state, heartbeat_event, make_event_metadata())

        assert len(result.intents) == 1

        payload = result.intents[0].payload
        assert payload.intent_type == "postgres.update_heartbeat"
        assert payload.node_name == "test-reducer"
        assert payload.uptime_seconds == 60.0

    def test_heartbeat_increments_counter(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        heartbeat_event: ModelNodeHeartbeatEvent,
    ) -> None:
        """State should track heartbeats_processed count."""
        result = reducer.reduce(initial_state, heartbeat_event, make_event_metadata())

        assert result.result.heartbeats_processed == 1


# =============================================================================
# Test: Staleness Computation on Runtime Tick
# =============================================================================


@pytest.mark.integration
class TestContractRegistryReducerStaleness:
    """Test staleness computation on runtime tick."""

    def test_runtime_tick_emits_mark_stale_intent(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        runtime_tick_event: ModelRuntimeTick,
    ) -> None:
        """Runtime tick should emit postgres.mark_stale intent."""
        result = reducer.reduce(
            initial_state, runtime_tick_event, make_event_metadata()
        )

        assert len(result.intents) == 1

        payload = result.intents[0].payload
        assert payload.intent_type == "postgres.mark_stale"
        assert payload.stale_cutoff is not None
        assert payload.checked_at is not None

    def test_staleness_threshold_is_five_minutes(self) -> None:
        """Staleness threshold should be 5 minutes."""
        assert timedelta(minutes=5) == STALENESS_THRESHOLD

    def test_runtime_tick_updates_staleness_check_timestamp(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        runtime_tick_event: ModelRuntimeTick,
    ) -> None:
        """Runtime tick should update last_staleness_check_at in state."""
        assert initial_state.last_staleness_check_at is None

        result = reducer.reduce(
            initial_state, runtime_tick_event, make_event_metadata()
        )

        assert result.result.last_staleness_check_at is not None
        # Should match the tick's now timestamp
        assert result.result.last_staleness_check_at == runtime_tick_event.now


# =============================================================================
# Test: State Model Behavior
# =============================================================================


@pytest.mark.integration
class TestContractRegistryState:
    """Test state model behavior."""

    def test_state_is_immutable(
        self, initial_state: ModelContractRegistryState
    ) -> None:
        """State model should be frozen (immutable)."""
        with pytest.raises(ValidationError):
            initial_state.contracts_processed = 10  # type: ignore[misc]

    def test_state_transition_returns_new_instance(
        self, initial_state: ModelContractRegistryState
    ) -> None:
        """State transitions should return new instances."""
        new_state = initial_state.with_contract_registered()

        assert new_state is not initial_state
        assert new_state.contracts_processed == 1
        assert initial_state.contracts_processed == 0

    def test_duplicate_detection(
        self, initial_state: ModelContractRegistryState
    ) -> None:
        """State should correctly detect duplicate events."""
        # Initial state - no duplicates
        assert not initial_state.is_duplicate_event("topic", 0, 100)

        # Update position
        new_state = initial_state.with_event_processed(
            event_id=uuid4(),
            topic="topic",
            partition=0,
            offset=100,
        )

        # Same position should be duplicate
        assert new_state.is_duplicate_event("topic", 0, 100)

        # Lower offset should be duplicate
        assert new_state.is_duplicate_event("topic", 0, 99)

        # Higher offset should NOT be duplicate
        assert not new_state.is_duplicate_event("topic", 0, 101)

        # Different partition should NOT be duplicate
        assert not new_state.is_duplicate_event("topic", 1, 50)


# =============================================================================
# Test: Malformed YAML Handling
# =============================================================================


@pytest.mark.integration
class TestContractRegistryReducerMalformedYaml:
    """Test graceful handling of malformed contract_yaml.

    The reducer should skip topic extraction when contract_yaml cannot be parsed,
    but should still emit the upsert_contract intent and process the event.

    Related:
        - OMN-1653: Contract Registry Reducer implementation
        - reducer.py _build_topic_update_intents() YAML parse error handling
    """

    def test_malformed_yaml_skips_topic_extraction(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        sample_version: ModelSemVer,
    ) -> None:
        """Malformed YAML should skip topic extraction but still process event."""
        # Create event with malformed YAML (unclosed bracket)
        event = ModelContractRegisteredEvent(
            event_id=uuid4(),
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            source_node_id=uuid4(),
            event_type="onex.evt.platform.contract-registered.v1",
            node_name="test-reducer",
            node_version=sample_version,
            contract_hash="abc123",
            contract_yaml="invalid: yaml: [unclosed",  # Malformed YAML
        )

        result = reducer.reduce(initial_state, event, make_event_metadata())

        # Should still process event
        assert result.items_processed == 1
        assert result.result.contracts_processed == 1

        # Should have only upsert intent (no topic intents due to parse failure)
        assert len(result.intents) == 1
        assert result.intents[0].payload.intent_type == "postgres.upsert_contract"

    def test_malformed_yaml_colon_in_value(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        sample_version: ModelSemVer,
    ) -> None:
        """YAML with unquoted colon in value should skip topic extraction."""
        event = ModelContractRegisteredEvent(
            event_id=uuid4(),
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            source_node_id=uuid4(),
            event_type="onex.evt.platform.contract-registered.v1",
            node_name="test-reducer",
            node_version=sample_version,
            contract_hash="abc123",
            contract_yaml="key: value: with: extra: colons:",  # Invalid YAML
        )

        result = reducer.reduce(initial_state, event, make_event_metadata())

        assert result.items_processed == 1
        assert result.result.contracts_processed == 1
        assert len(result.intents) == 1
        assert result.intents[0].payload.intent_type == "postgres.upsert_contract"

    def test_non_dict_yaml_skips_topic_extraction(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        sample_version: ModelSemVer,
    ) -> None:
        """Valid YAML that parses to non-dict should skip topic extraction."""
        event = ModelContractRegisteredEvent(
            event_id=uuid4(),
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            source_node_id=uuid4(),
            event_type="onex.evt.platform.contract-registered.v1",
            node_name="test-reducer",
            node_version=sample_version,
            contract_hash="abc123",
            contract_yaml="- item1\n- item2\n- item3",  # Valid YAML but a list
        )

        result = reducer.reduce(initial_state, event, make_event_metadata())

        # Should still process event
        assert result.items_processed == 1
        assert result.result.contracts_processed == 1

        # Should have only upsert intent (topic extraction skipped for non-dict)
        assert len(result.intents) == 1
        assert result.intents[0].payload.intent_type == "postgres.upsert_contract"

    def test_empty_yaml_skips_topic_extraction(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        sample_version: ModelSemVer,
    ) -> None:
        """Empty string YAML should skip topic extraction."""
        event = ModelContractRegisteredEvent(
            event_id=uuid4(),
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            source_node_id=uuid4(),
            event_type="onex.evt.platform.contract-registered.v1",
            node_name="test-reducer",
            node_version=sample_version,
            contract_hash="abc123",
            contract_yaml="",  # Empty string
        )

        result = reducer.reduce(initial_state, event, make_event_metadata())

        # Should still process event
        assert result.items_processed == 1
        assert result.result.contracts_processed == 1

        # Empty string parses to None which is not a dict
        assert len(result.intents) == 1
        assert result.intents[0].payload.intent_type == "postgres.upsert_contract"

    def test_yaml_without_topics_no_topic_intents(
        self,
        reducer: ContractRegistryReducer,
        initial_state: ModelContractRegistryState,
        sample_version: ModelSemVer,
    ) -> None:
        """Valid YAML without consumed_events/published_events has no topic intents."""
        event = ModelContractRegisteredEvent(
            event_id=uuid4(),
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            source_node_id=uuid4(),
            event_type="onex.evt.platform.contract-registered.v1",
            node_name="test-reducer",
            node_version=sample_version,
            contract_hash="abc123",
            contract_yaml="name: test\nversion: 1.0.0",  # Valid but no topics
        )

        result = reducer.reduce(initial_state, event, make_event_metadata())

        # Should still process event
        assert result.items_processed == 1
        assert result.result.contracts_processed == 1

        # Should have only upsert intent (no topics defined in contract)
        assert len(result.intents) == 1
        assert result.intents[0].payload.intent_type == "postgres.upsert_contract"
