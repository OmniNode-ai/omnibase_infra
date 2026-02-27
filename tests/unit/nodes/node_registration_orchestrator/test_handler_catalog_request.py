# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerCatalogRequest (OMN-2923).

Tests validate:
- Happy path: handler returns ModelTopicCatalogResponse with onex.evt.* topics
- Cold-start: no nodes registered -> empty topics + cold registry warning
- Topic filter: only onex.evt.* topics included, others excluded
- Correlation ID: echoed from payload (not envelope)
- Multiple nodes: topics unioned and deduplicated across all registered nodes
- Handler properties: handler_id, category, message_types, node_kind, handler_type

Related Tickets:
    - OMN-2923: Catalog responder for topic-catalog-request.v1
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.models.registration.model_topic_catalog_request import (
    ModelTopicCatalogRequest,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_catalog_request import (
    _WARNING_COLD_REGISTRY,
    HandlerCatalogRequest,
)
from omnibase_infra.nodes.node_registration_orchestrator.services.service_introspection_topic_store import (
    ServiceIntrospectionTopicStore,
)

TEST_NOW = datetime(2026, 2, 27, 12, 0, 0, tzinfo=UTC)


def _make_store() -> ServiceIntrospectionTopicStore:
    """Create a fresh ServiceIntrospectionTopicStore."""
    return ServiceIntrospectionTopicStore()


def _make_request_envelope(
    correlation_id: UUID | None = None,
    requester: str | None = "test-client",
    store: ServiceIntrospectionTopicStore | None = None,
) -> tuple[
    ModelEventEnvelope[ModelTopicCatalogRequest], ServiceIntrospectionTopicStore
]:
    """Create a test request envelope and store."""
    if correlation_id is None:
        correlation_id = uuid4()
    if store is None:
        store = _make_store()
    payload = ModelTopicCatalogRequest(
        correlation_id=correlation_id,
        requested_at=TEST_NOW,
        requester=requester,
    )
    envelope: ModelEventEnvelope[ModelTopicCatalogRequest] = ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=payload,
        envelope_timestamp=TEST_NOW,
        correlation_id=correlation_id,
        source="test",
    )
    return envelope, store


@pytest.mark.unit
class TestHandlerCatalogRequestProperties:
    """Verify handler property values."""

    def test_handler_id(self) -> None:
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        assert handler.handler_id == "handler-catalog-request"

    def test_category(self) -> None:
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        assert handler.category == EnumMessageCategory.COMMAND

    def test_message_types(self) -> None:
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        assert "ModelTopicCatalogRequest" in handler.message_types

    def test_node_kind(self) -> None:
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        assert handler.node_kind == EnumNodeKind.ORCHESTRATOR

    def test_handler_type(self) -> None:
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self) -> None:
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        assert handler.handler_category == EnumHandlerTypeCategory.COMPUTE


@pytest.mark.unit
class TestHandlerCatalogRequestColdStart:
    """Verify cold-start behavior when no nodes have registered."""

    @pytest.mark.asyncio
    async def test_cold_start_returns_empty_topics(self) -> None:
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        envelope, _ = _make_request_envelope(store=store)

        output = await handler.handle(envelope)

        assert len(output.events) == 1
        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)
        assert len(response.topics) == 0
        assert _WARNING_COLD_REGISTRY in response.warnings

    @pytest.mark.asyncio
    async def test_cold_start_echoes_correlation_id(self) -> None:
        corr_id = uuid4()
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        envelope, _ = _make_request_envelope(correlation_id=corr_id, store=store)

        output = await handler.handle(envelope)

        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)
        assert response.correlation_id == corr_id


@pytest.mark.unit
class TestHandlerCatalogRequestTopicFiltering:
    """Verify only onex.evt.* topics are included."""

    @pytest.mark.asyncio
    async def test_filters_to_evt_topics_only(self) -> None:
        store = _make_store()
        await store.update_node(
            "node-001",
            [
                "onex.evt.platform.node-registration.v1",
                "onex.evt.omniintelligence.intent-classified.v1",
                "onex.cmd.platform.request-introspection.v1",  # should be excluded
                "onex.intent.platform.runtime-tick.v1",  # should be excluded
            ],
        )
        handler = HandlerCatalogRequest(topic_store=store)
        envelope, _ = _make_request_envelope(store=store)

        output = await handler.handle(envelope)

        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)
        topic_names = {e.topic_name for e in response.topics}
        assert "onex.evt.platform.node-registration.v1" in topic_names
        assert "onex.evt.omniintelligence.intent-classified.v1" in topic_names
        assert "onex.cmd.platform.request-introspection.v1" not in topic_names
        assert "onex.intent.platform.runtime-tick.v1" not in topic_names

    @pytest.mark.asyncio
    async def test_no_cold_warning_when_nodes_registered(self) -> None:
        store = _make_store()
        await store.update_node("node-001", ["onex.evt.platform.foo.v1"])
        handler = HandlerCatalogRequest(topic_store=store)
        envelope, _ = _make_request_envelope(store=store)

        output = await handler.handle(envelope)

        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)
        assert _WARNING_COLD_REGISTRY not in response.warnings


@pytest.mark.unit
class TestHandlerCatalogRequestMultiNode:
    """Verify topic union across multiple registered nodes."""

    @pytest.mark.asyncio
    async def test_unions_topics_from_multiple_nodes(self) -> None:
        store = _make_store()
        await store.update_node("node-001", ["onex.evt.platform.foo.v1"])
        await store.update_node(
            "node-002",
            [
                "onex.evt.platform.bar.v1",
                "onex.evt.omniintelligence.pattern-stored.v1",
            ],
        )
        await store.update_node("node-003", ["onex.evt.platform.foo.v1"])  # duplicate

        handler = HandlerCatalogRequest(topic_store=store)
        envelope, _ = _make_request_envelope(store=store)

        output = await handler.handle(envelope)

        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)
        topic_names = {e.topic_name for e in response.topics}
        # Deduplicated: foo.v1 appears in node-001 and node-003 but listed once
        assert "onex.evt.platform.foo.v1" in topic_names
        assert "onex.evt.platform.bar.v1" in topic_names
        assert "onex.evt.omniintelligence.pattern-stored.v1" in topic_names
        # Exactly 3 unique evt topics
        assert len(topic_names) == 3

    @pytest.mark.asyncio
    async def test_node_count_in_response(self) -> None:
        store = _make_store()
        await store.update_node("node-001", ["onex.evt.platform.foo.v1"])
        await store.update_node("node-002", ["onex.evt.platform.bar.v1"])

        handler = HandlerCatalogRequest(topic_store=store)
        envelope, _ = _make_request_envelope(store=store)

        output = await handler.handle(envelope)

        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)
        assert response.node_count == 2

    @pytest.mark.asyncio
    async def test_topic_names_sorted(self) -> None:
        store = _make_store()
        await store.update_node(
            "node-001",
            [
                "onex.evt.platform.zzz.v1",
                "onex.evt.platform.aaa.v1",
                "onex.evt.omnimemory.doc-changed.v1",
            ],
        )
        handler = HandlerCatalogRequest(topic_store=store)
        envelope, _ = _make_request_envelope(store=store)

        output = await handler.handle(envelope)

        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)
        names = [e.topic_name for e in response.topics]
        assert names == sorted(names)


@pytest.mark.unit
class TestHandlerCatalogRequestOutput:
    """Verify output structure."""

    @pytest.mark.asyncio
    async def test_output_has_exactly_one_event(self) -> None:
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        envelope, _ = _make_request_envelope(store=store)

        output = await handler.handle(envelope)

        assert len(output.events) == 1
        assert len(output.intents) == 0
        assert len(output.projections) == 0

    @pytest.mark.asyncio
    async def test_output_event_is_topic_catalog_response(self) -> None:
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        envelope, _ = _make_request_envelope(store=store)

        output = await handler.handle(envelope)

        assert isinstance(output.events[0], ModelTopicCatalogResponse)

    @pytest.mark.asyncio
    async def test_handler_id_in_output(self) -> None:
        store = _make_store()
        handler = HandlerCatalogRequest(topic_store=store)
        envelope, _ = _make_request_envelope(store=store)

        output = await handler.handle(envelope)

        assert output.handler_id == "handler-catalog-request"


@pytest.mark.unit
class TestServiceIntrospectionTopicStore:
    """Unit tests for ServiceIntrospectionTopicStore."""

    @pytest.mark.asyncio
    async def test_empty_store_returns_no_evt_topics(self) -> None:
        store = _make_store()
        topics = await store.get_evt_topics()
        assert topics == []

    @pytest.mark.asyncio
    async def test_update_and_retrieve(self) -> None:
        store = _make_store()
        await store.update_node(
            "node-001",
            ["onex.evt.platform.foo.v1", "onex.cmd.platform.bar.v1"],
        )
        topics = await store.get_evt_topics()
        assert "onex.evt.platform.foo.v1" in topics
        assert "onex.cmd.platform.bar.v1" not in topics

    @pytest.mark.asyncio
    async def test_update_replaces_previous_entry(self) -> None:
        store = _make_store()
        await store.update_node("node-001", ["onex.evt.platform.old.v1"])
        await store.update_node("node-001", ["onex.evt.platform.new.v1"])
        topics = await store.get_evt_topics()
        assert "onex.evt.platform.new.v1" in topics
        assert "onex.evt.platform.old.v1" not in topics

    @pytest.mark.asyncio
    async def test_node_count(self) -> None:
        store = _make_store()
        assert await store.get_node_count() == 0
        await store.update_node("node-001", ["onex.evt.platform.foo.v1"])
        await store.update_node("node-002", [])
        assert await store.get_node_count() == 2

    @pytest.mark.asyncio
    async def test_nodes_missing_event_bus(self) -> None:
        store = _make_store()
        await store.update_node("node-001", ["onex.evt.platform.foo.v1"])
        await store.update_node("node-002", [])  # missing event_bus
        count = await store.count_nodes_missing_event_bus()
        assert count == 1

    @pytest.mark.asyncio
    async def test_snapshot_atomic(self) -> None:
        store = _make_store()
        await store.update_node("node-001", ["onex.evt.platform.foo.v1"])
        await store.update_node("node-002", [])

        topics, node_count, missing = await store.snapshot()

        assert topics == ["onex.evt.platform.foo.v1"]
        assert node_count == 2
        assert missing == 1

    @pytest.mark.asyncio
    async def test_sorted_output(self) -> None:
        store = _make_store()
        await store.update_node(
            "node-001",
            ["onex.evt.platform.zzz.v1", "onex.evt.platform.aaa.v1"],
        )
        topics = await store.get_evt_topics()
        assert topics == sorted(topics)
