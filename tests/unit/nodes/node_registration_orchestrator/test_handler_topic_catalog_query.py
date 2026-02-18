# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerTopicCatalogQuery.

Tests validate:
- Happy path: handler returns ModelTopicCatalogResponse with catalog data
- Malformed query payload: returns empty response with invalid_query_payload warning
- Consul unavailable: ServiceTopicCatalog returns warnings, handler passes them through
- Unexpected exception from catalog service: returns empty response with internal_error warning
- Handler properties: handler_id, category, message_types, node_kind, handler_type

Related Tickets:
    - OMN-2313: Topic Catalog: query handler + dispatcher + contract wiring
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumMessageCategory, EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.catalog.model_topic_catalog_query import (
    ModelTopicCatalogQuery,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_topic_catalog_query import (
    HandlerTopicCatalogQuery,
)

TEST_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def _make_catalog_service(
    response: ModelTopicCatalogResponse | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock ServiceTopicCatalog."""
    service = MagicMock()
    if side_effect is not None:
        service.build_catalog = AsyncMock(side_effect=side_effect)
    else:
        if response is None:
            response = _empty_catalog_response(uuid4())
        service.build_catalog = AsyncMock(return_value=response)
    return service


def _empty_catalog_response(
    correlation_id: object,
    warnings: tuple[str, ...] = (),
) -> ModelTopicCatalogResponse:
    """Create an empty catalog response for testing."""
    from uuid import UUID

    return ModelTopicCatalogResponse(
        correlation_id=correlation_id if isinstance(correlation_id, UUID) else uuid4(),
        topics=(),
        catalog_version=0,
        node_count=0,
        generated_at=TEST_NOW,
        warnings=warnings,
    )


def _make_query_envelope(
    client_id: str = "test-client",
    include_inactive: bool = False,
    topic_pattern: str | None = None,
    correlation_id: object = None,
) -> ModelEventEnvelope[ModelTopicCatalogQuery]:
    """Create a valid query envelope for testing."""
    corr_id = correlation_id or uuid4()
    payload = ModelTopicCatalogQuery(
        correlation_id=corr_id,
        client_id=client_id,
        include_inactive=include_inactive,
        topic_pattern=topic_pattern,
    )
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=payload,
        envelope_timestamp=TEST_NOW,
        correlation_id=corr_id,
        source="test",
    )


# ---------------------------------------------------------------------------
# Handler property tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_handler_properties() -> None:
    """Handler exposes correct property values."""
    service = _make_catalog_service()
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    assert handler.handler_id == "handler-topic-catalog-query"
    assert handler.category == EnumMessageCategory.COMMAND
    assert "ModelTopicCatalogQuery" in handler.message_types
    assert handler.node_kind == EnumNodeKind.ORCHESTRATOR
    assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
    assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_handler_happy_path_returns_response() -> None:
    """Happy path: handler calls build_catalog and returns response as event."""
    correlation_id = uuid4()
    catalog_response = ModelTopicCatalogResponse(
        correlation_id=correlation_id,
        topics=(),
        catalog_version=3,
        node_count=5,
        generated_at=TEST_NOW,
        warnings=(),
    )
    service = _make_catalog_service(response=catalog_response)
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    envelope = _make_query_envelope(
        client_id="dashboard-service",
        correlation_id=correlation_id,
    )
    output = await handler.handle(envelope)

    assert len(output.events) == 1
    response_event = output.events[0]
    assert isinstance(response_event, ModelTopicCatalogResponse)
    assert response_event.catalog_version == 3
    assert response_event.node_count == 5
    assert response_event.warnings == ()


@pytest.mark.unit
async def test_handler_passes_filters_to_catalog_service() -> None:
    """Handler passes include_inactive and topic_pattern to catalog service."""
    service = _make_catalog_service()
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    envelope = _make_query_envelope(
        include_inactive=True,
        topic_pattern="onex.evt.*",
    )
    await handler.handle(envelope)

    service.build_catalog.assert_awaited_once()
    call_kwargs = service.build_catalog.call_args.kwargs
    assert call_kwargs["include_inactive"] is True
    assert call_kwargs["topic_pattern"] == "onex.evt.*"


@pytest.mark.unit
async def test_handler_always_returns_exactly_one_event() -> None:
    """Handler always returns exactly one event regardless of outcome."""
    service = _make_catalog_service()
    handler = HandlerTopicCatalogQuery(catalog_service=service)
    envelope = _make_query_envelope()

    output = await handler.handle(envelope)

    assert len(output.events) == 1
    assert output.result is None
    assert len(output.intents) == 0
    assert len(output.projections) == 0


# ---------------------------------------------------------------------------
# Malformed query payload
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_handler_malformed_payload_returns_warning() -> None:
    """Handler with wrong payload type returns invalid_query_payload warning."""
    service = _make_catalog_service()
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    # Create envelope with wrong payload type (not ModelTopicCatalogQuery)
    wrong_payload = MagicMock(spec=[])  # not a ModelTopicCatalogQuery
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=wrong_payload,
        envelope_timestamp=TEST_NOW,
        correlation_id=uuid4(),
        source="test",
    )

    output = await handler.handle(envelope)

    assert len(output.events) == 1
    response = output.events[0]
    assert isinstance(response, ModelTopicCatalogResponse)
    assert "invalid_query_payload" in response.warnings
    assert len(response.topics) == 0

    # Catalog service should NOT be called for malformed payloads
    service.build_catalog.assert_not_awaited()


# ---------------------------------------------------------------------------
# Consul unavailable
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_handler_consul_unavailable_returns_warning() -> None:
    """When Consul is unavailable, ServiceTopicCatalog returns no_consul_handler warning."""
    correlation_id = uuid4()
    consul_unavailable_response = _empty_catalog_response(
        correlation_id=correlation_id,
        warnings=("no_consul_handler",),
    )
    service = _make_catalog_service(response=consul_unavailable_response)
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    envelope = _make_query_envelope(correlation_id=correlation_id)
    output = await handler.handle(envelope)

    assert len(output.events) == 1
    response = output.events[0]
    assert isinstance(response, ModelTopicCatalogResponse)
    assert "no_consul_handler" in response.warnings
    assert len(response.topics) == 0


@pytest.mark.unit
async def test_handler_consul_scan_timeout_passes_through_warnings() -> None:
    """When Consul scan times out, ServiceTopicCatalog warning is passed through."""
    correlation_id = uuid4()
    timeout_response = _empty_catalog_response(
        correlation_id=correlation_id,
        warnings=("consul_scan_timeout",),
    )
    service = _make_catalog_service(response=timeout_response)
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    envelope = _make_query_envelope(correlation_id=correlation_id)
    output = await handler.handle(envelope)

    response = output.events[0]
    assert "consul_scan_timeout" in response.warnings


# ---------------------------------------------------------------------------
# Unexpected exception from catalog service
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_handler_unexpected_exception_returns_internal_error() -> None:
    """Unexpected exception from catalog service returns internal_error warning."""
    service = _make_catalog_service(
        side_effect=RuntimeError("unexpected failure"),
    )
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    envelope = _make_query_envelope()
    output = await handler.handle(envelope)

    assert len(output.events) == 1
    response = output.events[0]
    assert isinstance(response, ModelTopicCatalogResponse)
    assert "internal_error" in response.warnings
    assert len(response.topics) == 0


@pytest.mark.unit
async def test_handler_exception_never_propagates() -> None:
    """Handler never propagates exceptions - always returns a response."""
    service = _make_catalog_service(
        side_effect=Exception("critical failure"),
    )
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    envelope = _make_query_envelope()
    # Should not raise - handler always responds
    output = await handler.handle(envelope)

    assert output is not None
    assert len(output.events) == 1


# ---------------------------------------------------------------------------
# Correlation ID propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_handler_correlation_id_from_envelope() -> None:
    """Handler uses correlation_id from envelope when present."""
    correlation_id = uuid4()
    catalog_response = _empty_catalog_response(correlation_id=correlation_id)
    service = _make_catalog_service(response=catalog_response)
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    envelope = _make_query_envelope(correlation_id=correlation_id)
    output = await handler.handle(envelope)

    assert output.correlation_id == correlation_id
    # Verify correlation_id was passed to build_catalog
    call_kwargs = service.build_catalog.call_args.kwargs
    assert call_kwargs["correlation_id"] == correlation_id


@pytest.mark.unit
async def test_handler_generates_correlation_id_when_absent() -> None:
    """Handler generates a fallback correlation_id when envelope has none."""
    service = _make_catalog_service()
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    payload = ModelTopicCatalogQuery(
        correlation_id=uuid4(),
        client_id="test-client",
    )
    envelope: ModelEventEnvelope[ModelTopicCatalogQuery] = ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=payload,
        envelope_timestamp=TEST_NOW,
        correlation_id=None,
        source="test",
    )

    output = await handler.handle(envelope)

    # Should not raise and should generate a valid correlation_id
    assert output.correlation_id is not None


# ---------------------------------------------------------------------------
# Processing time
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_handler_records_processing_time() -> None:
    """Handler returns non-negative processing_time_ms."""
    service = _make_catalog_service()
    handler = HandlerTopicCatalogQuery(catalog_service=service)

    envelope = _make_query_envelope()
    output = await handler.handle(envelope)

    assert output.processing_time_ms >= 0.0
