# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for OMN-2314: IntentEffectConsulRegister catalog change emission.

Tests verify that:
1. ModelTopicCatalogChanged is emitted when the topic index delta is non-empty.
2. CAS version increment is called before event emission.
3. catalog_version=-1 (CAS failure) is handled gracefully (D3).
4. No emission when delta is empty.
5. No emission when catalog_service or event_bus is None.

Related:
    - OMN-2314: Topic Catalog change notification emission + CAS versioning
    - IntentEffectConsulRegister: Implementation under test
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import InfraUnavailableError, RuntimeHostError
from omnibase_infra.handlers.models.consul.model_consul_register_payload import (
    ModelConsulRegisterPayload,
)
from omnibase_infra.models.registration import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.runtime.intent_effects.intent_effect_consul_register import (
    IntentEffectConsulRegister,
)

pytestmark = [pytest.mark.unit]


def _make_consul_register_payload(
    topics_added: frozenset[str] = frozenset(),
    topics_removed: frozenset[str] = frozenset(),
) -> MagicMock:
    """Build a mock ModelHandlerOutput whose nested payload carries a register delta.

    Uses a real ModelConsulRegisterPayload instance so that isinstance checks in
    the production extraction path exercise the actual model, not a mock spec.
    """
    data = ModelConsulRegisterPayload(
        registered=True,
        name="onex-test-service",
        consul_service_id="onex-test-service-id",
        topics_added=topics_added,
        topics_removed=topics_removed,
    )

    payload_mock = MagicMock()
    payload_mock.data = data

    result_mock = MagicMock()
    result_mock.payload = payload_mock

    output_mock = MagicMock()
    output_mock.result = result_mock

    return output_mock


def _make_base_consul_payload(
    node_id: str | None = "test-node-1",
    event_bus_config: ModelNodeEventBusConfig | None = None,
) -> ModelPayloadConsulRegister:
    """Build a minimal ModelPayloadConsulRegister for testing."""
    return ModelPayloadConsulRegister(
        correlation_id=uuid4(),
        service_id="onex-effect-test-1",
        service_name="onex-effect",
        tags=["onex"],
        node_id=node_id,
        event_bus_config=event_bus_config,
    )


@pytest.mark.unit
class TestCatalogChangeEmission:
    """Tests for ModelTopicCatalogChanged emission in IntentEffectConsulRegister."""

    @pytest.fixture
    def mock_handler_with_delta(self) -> MagicMock:
        """Consul handler whose execute() returns a handler output with non-empty delta."""
        output = _make_consul_register_payload(
            topics_added=frozenset(["onex.evt.topic-a.v1"]),
            topics_removed=frozenset(),
        )
        handler = MagicMock()
        handler.execute = AsyncMock(return_value=output)
        return handler

    @pytest.fixture
    def mock_handler_no_delta(self) -> MagicMock:
        """Consul handler whose execute() returns a handler output with empty delta."""
        output = _make_consul_register_payload(
            topics_added=frozenset(),
            topics_removed=frozenset(),
        )
        handler = MagicMock()
        handler.execute = AsyncMock(return_value=output)
        return handler

    @pytest.fixture
    def mock_catalog_service(self) -> MagicMock:
        """ServiceTopicCatalog with successful CAS increment (returns version 5)."""
        svc = MagicMock()
        svc.increment_version = AsyncMock(return_value=5)
        return svc

    @pytest.fixture
    def mock_event_bus(self) -> MagicMock:
        """Mock event bus that captures publish_envelope calls."""
        bus = MagicMock()
        bus.publish_envelope = AsyncMock()
        return bus

    @pytest.mark.asyncio
    async def test_emit_on_nonempty_delta(
        self,
        mock_handler_with_delta: MagicMock,
        mock_catalog_service: MagicMock,
        mock_event_bus: MagicMock,
    ) -> None:
        """ModelTopicCatalogChanged is emitted when topics_added is non-empty."""
        from omnibase_infra.models.catalog.model_topic_catalog_changed import (
            ModelTopicCatalogChanged,
        )
        from omnibase_infra.topics.platform_topic_suffixes import (
            SUFFIX_TOPIC_CATALOG_CHANGED,
        )

        effect = IntentEffectConsulRegister(
            consul_handler=mock_handler_with_delta,
            catalog_service=mock_catalog_service,
            event_bus=mock_event_bus,
        )

        payload = _make_base_consul_payload()
        await effect.execute(payload, correlation_id=uuid4())

        # CAS increment must have been called
        mock_catalog_service.increment_version.assert_awaited_once()

        # Event bus must have been called with ModelTopicCatalogChanged
        mock_event_bus.publish_envelope.assert_awaited_once()
        call_args = mock_event_bus.publish_envelope.call_args

        # publish_envelope(envelope, topic) — positional args
        emitted_envelope = call_args.args[0]
        topic_arg = call_args.args[1]

        assert topic_arg == SUFFIX_TOPIC_CATALOG_CHANGED

        # The envelope payload should be a ModelTopicCatalogChanged
        changed_event = emitted_envelope.payload
        assert isinstance(changed_event, ModelTopicCatalogChanged)
        assert "onex.evt.topic-a.v1" in changed_event.topics_added
        assert changed_event.catalog_version == 5  # returned by mock CAS increment
        assert changed_event.trigger_reason == "registration"

    @pytest.mark.asyncio
    async def test_no_emit_when_delta_empty(
        self,
        mock_handler_no_delta: MagicMock,
        mock_catalog_service: MagicMock,
        mock_event_bus: MagicMock,
    ) -> None:
        """When delta is empty, no CAS increment and no event emission."""
        effect = IntentEffectConsulRegister(
            consul_handler=mock_handler_no_delta,
            catalog_service=mock_catalog_service,
            event_bus=mock_event_bus,
        )

        payload = _make_base_consul_payload()
        await effect.execute(payload, correlation_id=uuid4())

        mock_catalog_service.increment_version.assert_not_awaited()
        mock_event_bus.publish_envelope.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_emit_when_catalog_service_none(
        self,
        mock_handler_with_delta: MagicMock,
        mock_event_bus: MagicMock,
    ) -> None:
        """When catalog_service is None, no emission even if delta is non-empty."""
        effect = IntentEffectConsulRegister(
            consul_handler=mock_handler_with_delta,
            catalog_service=None,
            event_bus=mock_event_bus,
        )

        payload = _make_base_consul_payload()
        await effect.execute(payload, correlation_id=uuid4())

        mock_event_bus.publish_envelope.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_emit_when_event_bus_none(
        self,
        mock_handler_with_delta: MagicMock,
        mock_catalog_service: MagicMock,
    ) -> None:
        """When event_bus is None, no emission even if delta is non-empty.

        Joint guard invariant: _maybe_emit_catalog_changed() requires BOTH
        catalog_service and event_bus to be non-None before any catalog change
        notification occurs (see IntentEffectConsulRegister.__init__ docstring).
        This test exercises the event_bus=None branch; the catalog_service=None
        branch is covered by test_no_emit_when_catalog_service_none.  Neither
        CAS increment nor event emission should fire when either guard is absent.
        """
        effect = IntentEffectConsulRegister(
            consul_handler=mock_handler_with_delta,
            catalog_service=mock_catalog_service,
            event_bus=None,
        )

        payload = _make_base_consul_payload()
        await effect.execute(payload, correlation_id=uuid4())

        mock_catalog_service.increment_version.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cas_failure_catalog_version_minus_one(
        self,
        mock_handler_with_delta: MagicMock,
        mock_event_bus: MagicMock,
    ) -> None:
        """When CAS increment returns -1, catalog_version=0 (max(-1,0)=0) in event (D3)."""
        from omnibase_infra.models.catalog.model_topic_catalog_changed import (
            ModelTopicCatalogChanged,
        )

        # Simulate exhausted CAS retries
        failing_catalog = MagicMock()
        failing_catalog.increment_version = AsyncMock(return_value=-1)

        effect = IntentEffectConsulRegister(
            consul_handler=mock_handler_with_delta,
            catalog_service=failing_catalog,
            event_bus=mock_event_bus,
        )

        payload = _make_base_consul_payload()
        await effect.execute(payload, correlation_id=uuid4())

        # Event should still be emitted but with version 0 (max(-1,0))
        mock_event_bus.publish_envelope.assert_awaited_once()
        call_args = mock_event_bus.publish_envelope.call_args
        changed_event = call_args.args[0].payload
        assert isinstance(changed_event, ModelTopicCatalogChanged)
        assert changed_event.catalog_version == 0
        assert changed_event.cas_failure is True

    @pytest.mark.asyncio
    async def test_trigger_reason_registration_for_added_only(
        self,
        mock_catalog_service: MagicMock,
        mock_event_bus: MagicMock,
    ) -> None:
        """trigger_reason='registration' when only topics are added."""
        from omnibase_infra.models.catalog.model_topic_catalog_changed import (
            ModelTopicCatalogChanged,
        )

        output = _make_consul_register_payload(
            topics_added=frozenset(["onex.evt.topic-x.v1"]),
            topics_removed=frozenset(),
        )
        handler = MagicMock()
        handler.execute = AsyncMock(return_value=output)

        effect = IntentEffectConsulRegister(
            consul_handler=handler,
            catalog_service=mock_catalog_service,
            event_bus=mock_event_bus,
        )

        await effect.execute(_make_base_consul_payload(), correlation_id=uuid4())

        call_args = mock_event_bus.publish_envelope.call_args
        changed_event = call_args.args[0].payload
        assert isinstance(changed_event, ModelTopicCatalogChanged)
        assert changed_event.trigger_reason == "registration"

    @pytest.mark.asyncio
    async def test_trigger_reason_deregistration_for_removed_only(
        self,
        mock_catalog_service: MagicMock,
        mock_event_bus: MagicMock,
    ) -> None:
        """trigger_reason='deregistration' when only topics are removed."""
        from omnibase_infra.models.catalog.model_topic_catalog_changed import (
            ModelTopicCatalogChanged,
        )

        output = _make_consul_register_payload(
            topics_added=frozenset(),
            topics_removed=frozenset(["onex.evt.topic-old.v1"]),
        )
        handler = MagicMock()
        handler.execute = AsyncMock(return_value=output)

        effect = IntentEffectConsulRegister(
            consul_handler=handler,
            catalog_service=mock_catalog_service,
            event_bus=mock_event_bus,
        )

        await effect.execute(_make_base_consul_payload(), correlation_id=uuid4())

        call_args = mock_event_bus.publish_envelope.call_args
        changed_event = call_args.args[0].payload
        assert isinstance(changed_event, ModelTopicCatalogChanged)
        assert changed_event.trigger_reason == "deregistration"

    @pytest.mark.asyncio
    async def test_trigger_reason_capability_change_for_both(
        self,
        mock_catalog_service: MagicMock,
        mock_event_bus: MagicMock,
    ) -> None:
        """trigger_reason='capability_change' when both added and removed are non-empty."""
        from omnibase_infra.models.catalog.model_topic_catalog_changed import (
            ModelTopicCatalogChanged,
        )

        output = _make_consul_register_payload(
            topics_added=frozenset(["onex.evt.topic-new.v1"]),
            topics_removed=frozenset(["onex.evt.topic-old.v1"]),
        )
        handler = MagicMock()
        handler.execute = AsyncMock(return_value=output)

        effect = IntentEffectConsulRegister(
            consul_handler=handler,
            catalog_service=mock_catalog_service,
            event_bus=mock_event_bus,
        )

        await effect.execute(_make_base_consul_payload(), correlation_id=uuid4())

        call_args = mock_event_bus.publish_envelope.call_args
        changed_event = call_args.args[0].payload
        assert isinstance(changed_event, ModelTopicCatalogChanged)
        assert changed_event.trigger_reason == "capability_change"

    @pytest.mark.asyncio
    async def test_delta_tuples_sorted_alphabetically(
        self,
        mock_catalog_service: MagicMock,
        mock_event_bus: MagicMock,
    ) -> None:
        """topics_added and topics_removed in the emitted event are sorted (D7)."""
        from omnibase_infra.models.catalog.model_topic_catalog_changed import (
            ModelTopicCatalogChanged,
        )

        added = frozenset(
            ["onex.evt.topic-z.v1", "onex.evt.topic-a.v1", "onex.evt.topic-m.v1"]
        )
        output = _make_consul_register_payload(
            topics_added=added,
            topics_removed=frozenset(),
        )
        handler = MagicMock()
        handler.execute = AsyncMock(return_value=output)

        effect = IntentEffectConsulRegister(
            consul_handler=handler,
            catalog_service=mock_catalog_service,
            event_bus=mock_event_bus,
        )

        await effect.execute(_make_base_consul_payload(), correlation_id=uuid4())

        call_args = mock_event_bus.publish_envelope.call_args
        changed_event = call_args.args[0].payload
        assert isinstance(changed_event, ModelTopicCatalogChanged)
        # ModelTopicCatalogChanged.sort_delta_tuples validator ensures ordering
        assert list(changed_event.topics_added) == sorted(changed_event.topics_added)

    @pytest.mark.asyncio
    async def test_emit_failure_does_not_propagate(
        self,
        mock_handler_with_delta: MagicMock,
        mock_catalog_service: MagicMock,
    ) -> None:
        """Infrastructure emit failures are caught; Consul registration still succeeds.

        Only RuntimeHostError subclasses (e.g. InfraUnavailableError) are
        swallowed.  Programming errors such as pydantic.ValidationError are
        intentionally not caught and will propagate (see
        test_programming_error_in_emit_propagates).
        """
        failing_bus = MagicMock()
        failing_bus.publish_envelope = AsyncMock(
            side_effect=InfraUnavailableError("Kafka unavailable")
        )

        effect = IntentEffectConsulRegister(
            consul_handler=mock_handler_with_delta,
            catalog_service=mock_catalog_service,
            event_bus=failing_bus,
        )

        payload = _make_base_consul_payload()
        # Should NOT raise — infrastructure emission failure is best-effort
        await effect.execute(payload, correlation_id=uuid4())

    @pytest.mark.asyncio
    async def test_programming_error_in_emit_propagates(
        self,
        mock_handler_with_delta: MagicMock,
        mock_catalog_service: MagicMock,
    ) -> None:
        """Non-infrastructure errors from event construction are not swallowed.

        The broad ``except Exception`` was narrowed to ``except RuntimeHostError``
        so that programming errors (e.g. a bug in event construction logic that
        raises a ``TypeError`` or ``pydantic.ValidationError``) surface immediately
        rather than being silently logged and discarded.

        A ``TypeError`` is used here as a stand-in for any non-infrastructure
        exception (``TypeError`` is not a subclass of ``RuntimeHostError``).
        The outer ``execute()`` handler will re-wrap it in ``RuntimeHostError``.
        """
        failing_bus = MagicMock()
        # TypeError is not a RuntimeHostError subclass — it represents a
        # programming error (e.g. wrong type passed to publish_envelope).
        failing_bus.publish_envelope = AsyncMock(
            side_effect=TypeError("programming error: not a RuntimeHostError")
        )

        effect = IntentEffectConsulRegister(
            consul_handler=mock_handler_with_delta,
            catalog_service=mock_catalog_service,
            event_bus=failing_bus,
        )

        payload = _make_base_consul_payload()
        # TypeError must not be silently swallowed; it propagates out wrapped
        # in RuntimeHostError by the outer execute() exception handler.
        with pytest.raises(RuntimeHostError):
            await effect.execute(payload, correlation_id=uuid4())

    @pytest.mark.asyncio
    async def test_trigger_node_id_propagated(
        self,
        mock_handler_with_delta: MagicMock,
        mock_catalog_service: MagicMock,
        mock_event_bus: MagicMock,
    ) -> None:
        """trigger_node_id in the emitted event matches payload.node_id."""
        from omnibase_infra.models.catalog.model_topic_catalog_changed import (
            ModelTopicCatalogChanged,
        )

        effect = IntentEffectConsulRegister(
            consul_handler=mock_handler_with_delta,
            catalog_service=mock_catalog_service,
            event_bus=mock_event_bus,
        )

        node_id = "my-special-node-42"
        payload = _make_base_consul_payload(node_id=node_id)
        await effect.execute(payload, correlation_id=uuid4())

        call_args = mock_event_bus.publish_envelope.call_args
        changed_event = call_args.args[0].payload
        assert isinstance(changed_event, ModelTopicCatalogChanged)
        assert changed_event.trigger_node_id == node_id

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_to_changed_event(
        self,
        mock_handler_with_delta: MagicMock,
        mock_catalog_service: MagicMock,
        mock_event_bus: MagicMock,
    ) -> None:
        """The emitted ModelTopicCatalogChanged carries the same correlation_id."""
        from omnibase_infra.models.catalog.model_topic_catalog_changed import (
            ModelTopicCatalogChanged,
        )

        effect = IntentEffectConsulRegister(
            consul_handler=mock_handler_with_delta,
            catalog_service=mock_catalog_service,
            event_bus=mock_event_bus,
        )

        specific_correlation_id = uuid4()
        payload = _make_base_consul_payload()
        await effect.execute(payload, correlation_id=specific_correlation_id)

        call_args = mock_event_bus.publish_envelope.call_args
        changed_event = call_args.args[0].payload
        assert isinstance(changed_event, ModelTopicCatalogChanged)
        assert changed_event.correlation_id == specific_correlation_id
