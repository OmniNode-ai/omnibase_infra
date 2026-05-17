# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for data_provenance flow through contract registry reducer.

Validates:
- reducer extracts provenance from event_metadata and passes it to upsert payload
- known provenance values are preserved; absent/unrecognised values fall back to "unknown"
- router extracts provenance from envelope metadata tags into event_metadata

Related Tickets:
    - OMN-11201: data_provenance column migration and handler updates
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumReductionType, EnumStreamingMode
from omnibase_core.models.events import ModelContractRegisteredEvent
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.nodes import ModelReducerOutput
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_contract_registry_reducer.contract_registration_event_router import (
    ContractRegistrationEventRouter,
)
from omnibase_infra.nodes.node_contract_registry_reducer.models.model_contract_registry_state import (
    ModelContractRegistryState,
)
from omnibase_infra.nodes.node_contract_registry_reducer.models.model_payload_upsert_contract import (
    ModelPayloadUpsertContract,
)
from omnibase_infra.nodes.node_contract_registry_reducer.reducer import (
    ContractRegistryReducer,
)

TEST_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)

VALID_PROVENANCE_VALUES = [
    "demo_seeded",
    "demo_projected_shortcut",
    "measured",
    "estimated",
    "unknown",
]


# =============================================================================
# Helpers
# =============================================================================


def _make_registered_event() -> ModelContractRegisteredEvent:
    return ModelContractRegisteredEvent(
        event_id=uuid4(),
        correlation_id=uuid4(),
        timestamp=TEST_NOW,
        source_node_id=uuid4(),
        node_name="test-node",
        node_version=ModelSemVer(major=1, minor=0, patch=0),
        contract_hash="abc123",
        contract_yaml="name: test-node\nversion: 1.0.0",
    )


def _make_message(
    event: ModelContractRegisteredEvent,
    provenance: str | None = None,
) -> ModelEventMessage:
    tags: dict[str, str] = {}
    if provenance is not None:
        tags["data_provenance"] = provenance

    envelope = ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=event.model_dump(),
        envelope_timestamp=TEST_NOW,
        correlation_id=event.correlation_id,
        source="test",
        metadata={"tags": tags},  # type: ignore[arg-type]
    )
    payload_bytes = json.dumps(envelope.model_dump(mode="json")).encode("utf-8")
    return ModelEventMessage(
        topic="test.contract.events",
        key=None,
        value=payload_bytes,
        headers=ModelEventHeaders(
            source="test",
            event_type="contract-registered",
            correlation_id=envelope.correlation_id,
            timestamp=TEST_NOW,
        ),
        offset="1",
        partition=0,
    )


def _make_router(mock_reducer: MagicMock) -> ContractRegistrationEventRouter:
    container = MagicMock()
    upsert_handler = MagicMock()
    upsert_handler.handle = AsyncMock(return_value=MagicMock(success=True))
    return ContractRegistrationEventRouter(
        container=container,
        reducer=mock_reducer,
        effect_handlers={"postgres.upsert_contract": upsert_handler},  # type: ignore[arg-type]
        event_bus=None,
        tick_interval_seconds=60,
    )


def _default_reducer_output() -> ModelReducerOutput[ModelContractRegistryState]:
    return ModelReducerOutput(
        result=ModelContractRegistryState(),
        operation_id=uuid4(),
        reduction_type=EnumReductionType.MERGE,
        processing_time_ms=1.0,
        items_processed=1,
        conflicts_resolved=0,
        streaming_mode=EnumStreamingMode.BATCH,
        batches_processed=1,
        intents=(),
    )


# =============================================================================
# Reducer unit tests (pure function)
# =============================================================================


class TestReducerProvenanceExtraction:
    """The reducer reads data_provenance from event_metadata and sets it on the upsert payload."""

    def _call_on_contract_registered(
        self,
        reducer: ContractRegistryReducer,
        event: ModelContractRegisteredEvent,
        data_provenance: str,
    ) -> Any:
        state = ModelContractRegistryState()
        metadata = {
            "topic": "test.topic",
            "partition": 0,
            "offset": 1,
            "data_provenance": data_provenance,
        }
        return reducer.reduce(state, event, metadata)

    @pytest.mark.unit
    @pytest.mark.parametrize("provenance", VALID_PROVENANCE_VALUES)
    def test_valid_provenance_preserved(self, provenance: str) -> None:
        """All valid provenance values are propagated unchanged."""
        reducer = ContractRegistryReducer()
        event = _make_registered_event()
        output = self._call_on_contract_registered(reducer, event, provenance)

        upsert_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadUpsertContract)
        ]
        assert len(upsert_intents) == 1
        assert upsert_intents[0].payload.data_provenance == provenance  # type: ignore[union-attr]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "bad_value",
        ["", "UNKNOWN", "random_string", "MEASURED", None],
    )
    def test_invalid_provenance_falls_back_to_unknown(
        self, bad_value: str | None
    ) -> None:
        """Absent or unrecognised provenance values collapse to 'unknown'."""
        reducer = ContractRegistryReducer()
        event = _make_registered_event()
        state = ModelContractRegistryState()
        metadata: dict[str, Any] = {
            "topic": "test.topic",
            "partition": 0,
            "offset": 1,
        }
        if bad_value is not None:
            metadata["data_provenance"] = bad_value

        output = reducer.reduce(state, event, metadata)

        upsert_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadUpsertContract)
        ]
        assert len(upsert_intents) == 1
        assert upsert_intents[0].payload.data_provenance == "unknown"  # type: ignore[union-attr]

    @pytest.mark.unit
    def test_absent_provenance_key_falls_back_to_unknown(self) -> None:
        """When data_provenance key is missing entirely from metadata, default is 'unknown'."""
        reducer = ContractRegistryReducer()
        event = _make_registered_event()
        state = ModelContractRegistryState()
        metadata: dict[str, Any] = {"topic": "test.topic", "partition": 0, "offset": 1}

        output = reducer.reduce(state, event, metadata)

        upsert_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadUpsertContract)
        ]
        assert len(upsert_intents) == 1
        assert upsert_intents[0].payload.data_provenance == "unknown"  # type: ignore[union-attr]


# =============================================================================
# Router unit tests — provenance extracted from envelope metadata tags
# =============================================================================


class TestRouterProvenanceExtraction:
    """The router extracts data_provenance from envelope metadata tags."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    @pytest.mark.parametrize("provenance", VALID_PROVENANCE_VALUES)
    async def test_router_passes_valid_provenance_to_reducer(
        self, provenance: str
    ) -> None:
        """A valid data_provenance tag in the envelope is forwarded to reducer.reduce()."""
        mock_reducer = MagicMock(spec=ContractRegistryReducer)
        mock_reducer.reduce.return_value = _default_reducer_output()

        router = _make_router(mock_reducer)
        event = _make_registered_event()
        message = _make_message(event, provenance=provenance)

        await router.handle_message(message)

        mock_reducer.reduce.assert_called_once()
        _, _, event_metadata = mock_reducer.reduce.call_args[0]
        assert event_metadata.get("data_provenance") == provenance

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_router_falls_back_to_unknown_when_tag_absent(self) -> None:
        """When data_provenance tag is absent in the envelope, 'unknown' is forwarded."""
        mock_reducer = MagicMock(spec=ContractRegistryReducer)
        mock_reducer.reduce.return_value = _default_reducer_output()

        router = _make_router(mock_reducer)
        event = _make_registered_event()
        message = _make_message(event, provenance=None)

        await router.handle_message(message)

        mock_reducer.reduce.assert_called_once()
        _, _, event_metadata = mock_reducer.reduce.call_args[0]
        assert event_metadata.get("data_provenance") == "unknown"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_router_falls_back_to_unknown_for_invalid_tag(self) -> None:
        """An unrecognised data_provenance tag value is normalised to 'unknown'."""
        mock_reducer = MagicMock(spec=ContractRegistryReducer)
        mock_reducer.reduce.return_value = _default_reducer_output()

        router = _make_router(mock_reducer)
        event = _make_registered_event()
        message = _make_message(event, provenance="INVALID_VALUE")

        await router.handle_message(message)

        mock_reducer.reduce.assert_called_once()
        _, _, event_metadata = mock_reducer.reduce.call_args[0]
        assert event_metadata.get("data_provenance") == "unknown"


# =============================================================================
# Projection model tests
# =============================================================================


class TestProjectionModelDefaults:
    """Projection models carry data_provenance with a default of 'unknown'."""

    @pytest.mark.unit
    def test_contract_projection_default_provenance(self) -> None:
        from datetime import UTC, datetime

        from omnibase_infra.models.projection.model_contract_projection import (
            ModelContractProjection,
        )

        now = datetime.now(UTC)
        proj = ModelContractProjection(
            contract_id="test-node:1.0.0",
            node_name="test-node",
            version_major=1,
            version_minor=0,
            version_patch=0,
            contract_hash="abc",
            contract_yaml="name: test-node",
            registered_at=now,
            last_seen_at=now,
            is_active=True,
        )
        assert proj.data_provenance == "unknown"

    @pytest.mark.unit
    def test_contract_projection_explicit_provenance(self) -> None:
        from datetime import UTC, datetime

        from omnibase_infra.models.projection.model_contract_projection import (
            ModelContractProjection,
        )

        now = datetime.now(UTC)
        proj = ModelContractProjection(
            contract_id="test-node:1.0.0",
            node_name="test-node",
            version_major=1,
            version_minor=0,
            version_patch=0,
            contract_hash="abc",
            contract_yaml="name: test-node",
            registered_at=now,
            last_seen_at=now,
            is_active=True,
            data_provenance="measured",
        )
        assert proj.data_provenance == "measured"

    @pytest.mark.unit
    def test_topic_projection_default_provenance(self) -> None:
        from datetime import UTC, datetime

        from omnibase_infra.models.projection.model_topic_projection import (
            ModelTopicProjection,
        )

        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="onex.evt.test.v1",
            direction="publish",
            first_seen_at=now,
            last_seen_at=now,
        )
        assert proj.data_provenance == "unknown"

    @pytest.mark.unit
    def test_registration_projection_default_provenance(self) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from omnibase_core.enums import EnumNodeKind
        from omnibase_infra.enums import EnumRegistrationState
        from omnibase_infra.models.projection.model_registration_projection import (
            ModelRegistrationProjection,
        )

        now = datetime.now(UTC)
        proj = ModelRegistrationProjection(
            entity_id=uuid4(),
            current_state=EnumRegistrationState.ACTIVE,
            node_type=EnumNodeKind.EFFECT,
            last_applied_event_id=uuid4(),
            registered_at=now,
            updated_at=now,
        )
        assert proj.data_provenance == "unknown"


__all__: list[str] = [
    "TestReducerProvenanceExtraction",
    "TestRouterProvenanceExtraction",
    "TestProjectionModelDefaults",
]
