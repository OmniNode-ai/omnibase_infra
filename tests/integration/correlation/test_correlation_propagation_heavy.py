# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Heavy integration tests for correlation ID propagation with real infrastructure.

These tests require:
- Real Kafka/Redpanda (via existing kafka fixtures if available)
- Real PostgreSQL (via db_config fixture)
- pytest-httpserver for HTTP testing

Run with: RUN_HEAVY_TESTS=1 pytest tests/integration/correlation/test_correlation_propagation_heavy.py -v

Test Categories
===============

HTTP Boundary Tests:
    Tests that verify correlation IDs propagate correctly through HTTP calls
    using pytest-httpserver as a mock HTTP endpoint.

Error Context Tests:
    Tests that verify correlation IDs are preserved in error context when
    infrastructure operations fail.

Database Tests (placeholder):
    Tests that require real PostgreSQL - skipped until db fixtures available.

Kafka Tests (placeholder):
    Tests that require real Kafka/Redpanda - skipped until kafka fixtures available.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
)

# Check if pytest-httpserver and httpx are available for HTTP boundary tests
try:
    import httpx
    from pytest_httpserver import HTTPServer

    HTTPSERVER_AVAILABLE = True
except ImportError:
    HTTPSERVER_AVAILABLE = False
    # Assign None to module reference for conditional skip logic
    httpx = None  # type: ignore[assignment]

    # Placeholder class to avoid NameError when pytest-httpserver unavailable
    class HTTPServer:  # type: ignore[no-redef]
        """Placeholder class when pytest-httpserver is not available."""


if TYPE_CHECKING:
    import logging
    from uuid import UUID

# =============================================================================
# Module-Level Skip Configuration
# =============================================================================

# Skip entire module if RUN_HEAVY_TESTS is not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.heavy,
    pytest.mark.skipif(
        not os.getenv("RUN_HEAVY_TESTS"),
        reason="Heavy tests require RUN_HEAVY_TESTS=1 environment variable",
    ),
]


# =============================================================================
# HTTP Boundary Tests
# =============================================================================


@pytest.mark.skipif(
    not HTTPSERVER_AVAILABLE,
    reason="pytest-httpserver or httpx not installed - pip install pytest-httpserver httpx",
)
class TestCorrelationHttpBoundary:
    """Tests for correlation ID propagation through HTTP boundaries."""

    @pytest.mark.asyncio
    async def test_correlation_through_http_boundary(
        self,
        httpserver: HTTPServer,
        correlation_id: UUID,
    ) -> None:
        """Verify correlation ID propagates through HTTP calls.

        This test uses pytest-httpserver to create a mock HTTP server that
        expects to receive requests with correlation ID headers. The server
        will fail the test if the expected header is not present.

        Args:
            httpserver: pytest-httpserver fixture providing mock HTTP server
            correlation_id: Test correlation ID from conftest fixture
        """
        # Configure mock server to expect correlation ID header
        httpserver.expect_request(
            "/test-correlation",
            headers={"X-Correlation-ID": str(correlation_id)},
        ).respond_with_json(
            {"status": "ok", "correlation_id": str(correlation_id)},
        )

        # Make HTTP call with correlation ID
        async with httpx.AsyncClient() as client:
            response = await client.get(
                httpserver.url_for("/test-correlation"),
                headers={"X-Correlation-ID": str(correlation_id)},
            )

        assert response.status_code == 200
        response_data = response.json()
        # HTTP responses serialize correlation_id as string for wire transport (JSON body)
        assert response_data["correlation_id"] == str(correlation_id)
        # pytest-httpserver will fail the test if expected header wasn't present

    @pytest.mark.asyncio
    async def test_correlation_echoed_in_response_header(
        self,
        httpserver: HTTPServer,
        correlation_id: UUID,
    ) -> None:
        """Verify correlation ID is echoed back in response headers.

        Tests the common pattern where servers echo the correlation ID
        back in the response headers for end-to-end tracing.

        Args:
            httpserver: pytest-httpserver fixture providing mock HTTP server
            correlation_id: Test correlation ID from conftest fixture
        """
        # Configure mock server to echo correlation ID in response headers
        httpserver.expect_request(
            "/echo-correlation",
        ).respond_with_json(
            {"status": "ok"},
            headers={"X-Correlation-ID": str(correlation_id)},
        )

        async with httpx.AsyncClient() as client:
            response = await client.get(
                httpserver.url_for("/echo-correlation"),
                headers={"X-Correlation-ID": str(correlation_id)},
            )

        assert response.status_code == 200
        # HTTP headers are strings; correlation_id serialized for wire transport
        assert response.headers.get("X-Correlation-ID") == str(correlation_id)


# =============================================================================
# Error Context Preservation Tests
# =============================================================================


class TestCorrelationErrorContext:
    """Tests for correlation ID preservation in error contexts."""

    @pytest.mark.asyncio
    async def test_correlation_preserved_on_connection_error(
        self,
        correlation_id: UUID,
    ) -> None:
        """Verify correlation ID survives connection errors.

        Tests that when infrastructure connection errors occur, the
        correlation ID is properly preserved in the error context.

        Args:
            correlation_id: Test correlation ID from conftest fixture
        """
        # Create error context with correlation ID
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            operation="test_connection",
            transport_type=EnumInfraTransportType.HTTP,
            target_name="test-service",
        )

        # Simulate connection error with context
        error = InfraConnectionError("Connection refused", context=context)

        # Verify correlation ID is preserved in error
        assert error.correlation_id == correlation_id
        assert error.model.correlation_id == correlation_id

        # Verify context fields are preserved
        error_context = error.model.context
        assert error_context is not None
        assert error_context["operation"] == "test_connection"
        assert error_context["transport_type"] == EnumInfraTransportType.HTTP
        assert error_context["target_name"] == "test-service"

    @pytest.mark.asyncio
    async def test_correlation_preserved_on_timeout_error(
        self,
        correlation_id: UUID,
    ) -> None:
        """Verify correlation ID survives timeout errors.

        Tests that when infrastructure timeout errors occur, the
        correlation ID is properly preserved in the error context.

        Args:
            correlation_id: Test correlation ID from conftest fixture
        """
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            operation="database_query",
            transport_type=EnumInfraTransportType.DATABASE,
            target_name="postgresql-primary",
        )

        error = InfraTimeoutError("Query timed out after 30s", context=context)

        # Verify correlation ID is preserved
        assert error.correlation_id == correlation_id
        assert error.model.correlation_id == correlation_id

        # Verify context fields
        error_context = error.model.context
        assert error_context is not None
        assert error_context["operation"] == "database_query"
        assert error_context["transport_type"] == EnumInfraTransportType.DATABASE

    @pytest.mark.asyncio
    async def test_correlation_preserved_on_unavailable_error(
        self,
        correlation_id: UUID,
    ) -> None:
        """Verify correlation ID survives unavailable errors.

        Tests that when services are unavailable, the correlation ID
        is properly preserved in the error context for tracing.

        Args:
            correlation_id: Test correlation ID from conftest fixture
        """
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            operation="kafka_publish",
            transport_type=EnumInfraTransportType.KAFKA,
            target_name="kafka-broker-1",
        )

        error = InfraUnavailableError("Broker not available", context=context)

        # Verify correlation ID is preserved
        assert error.correlation_id == correlation_id
        assert error.model.correlation_id == correlation_id

        # Verify context fields
        error_context = error.model.context
        assert error_context is not None
        assert error_context["operation"] == "kafka_publish"
        assert error_context["transport_type"] == EnumInfraTransportType.KAFKA

    @pytest.mark.asyncio
    async def test_correlation_in_error_string_representation(
        self,
        correlation_id: UUID,
    ) -> None:
        """Verify correlation ID appears in error string representation.

        Tests that the error's string representation includes the
        correlation ID for debugging and logging purposes.

        Args:
            correlation_id: Test correlation ID from conftest fixture
        """
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            operation="test_operation",
            transport_type=EnumInfraTransportType.HTTP,
        )

        error = InfraConnectionError("Test error message", context=context)

        # The error's model dump should contain the correlation ID
        error_dump = error.model_dump()
        assert str(correlation_id) in str(error_dump)
        # model_dump() returns UUID objects; convert both sides to string for comparison
        assert str(error_dump["correlation_id"]) == str(correlation_id)


# =============================================================================
# Database Tests (Placeholder)
# =============================================================================


class TestCorrelationDatabase:
    """Tests for correlation ID propagation through database operations.

    These tests require real PostgreSQL infrastructure and are skipped
    until proper database fixtures are available.
    """

    @pytest.mark.asyncio
    async def test_correlation_preserved_on_db_operation(
        self,
        correlation_id: UUID,
        log_capture: list[logging.LogRecord],
    ) -> None:
        """Verify correlation ID propagates through database operations.

        This test requires real PostgreSQL and proper db fixtures.
        Currently a placeholder documenting expected behavior.

        Args:
            correlation_id: Test correlation ID from conftest fixture
            log_capture: Log capturing fixture from conftest
        """
        # TODO(OMN-1349): Implement when database fixtures available
        # Required fixtures: db_config, initialized_db_handler from handlers/conftest.py
        # Implementation should:
        # 1. Execute a database operation with correlation_id in context
        # 2. Verify correlation_id appears in log records via log_capture
        # 3. Verify correlation_id is preserved in any error contexts
        # See tests/integration/handlers/conftest.py for db_config and initialized_db_handler patterns
        pytest.skip("Requires real PostgreSQL - implement when db fixtures available")

    @pytest.mark.asyncio
    async def test_correlation_in_db_error_context(
        self,
        correlation_id: UUID,
    ) -> None:
        """Verify correlation ID is preserved when database operations fail.

        Tests that database connection and query errors properly preserve
        correlation IDs for distributed tracing.

        Args:
            correlation_id: Test correlation ID from conftest fixture
        """
        # TODO(OMN-1349): Implement when database fixtures available
        # Required fixtures: db_config, initialized_db_handler from handlers/conftest.py
        # Implementation should:
        # 1. Trigger a database error (invalid query, connection failure, etc.)
        # 2. Catch InfraConnectionError or InfraTimeoutError
        # 3. Verify error.correlation_id == correlation_id
        # 4. Verify error.model.context contains expected operation details
        # See tests/integration/handlers/conftest.py for fixture patterns
        pytest.skip("Requires real PostgreSQL - implement when db fixtures available")


# =============================================================================
# Kafka Tests (Placeholder)
# =============================================================================


class TestCorrelationKafka:
    """Tests for correlation ID propagation through Kafka/Redpanda.

    These tests require real Kafka/Redpanda infrastructure and are skipped
    until proper event bus fixtures are available.
    """

    @pytest.mark.asyncio
    async def test_correlation_end_to_end_with_real_kafka(
        self,
        correlation_id: UUID,
    ) -> None:
        """Verify correlation ID propagates end-to-end through Kafka.

        This test requires real Kafka/Redpanda infrastructure.
        Expected behavior:
        1. Publish message with correlation ID to topic
        2. Consume message from topic
        3. Verify correlation ID is preserved

        Args:
            correlation_id: Test correlation ID from conftest fixture
        """
        # TODO(OMN-1349): Implement when Kafka/event bus fixtures available
        # Required fixtures: kafka_producer, kafka_consumer, or event_bus adapter
        # Implementation should:
        # 1. Create ModelEventEnvelope with correlation_id
        # 2. Publish to test topic via Kafka adapter
        # 3. Consume message from topic
        # 4. Verify consumed envelope.correlation_id == original correlation_id
        # 5. Verify X-Correlation-ID header is present in Kafka message headers
        # Note: May need to create kafka fixtures similar to handlers/conftest.py patterns
        pytest.skip(
            "Requires real Kafka/Redpanda - implement when event bus fixtures available"
        )

    @pytest.mark.asyncio
    async def test_correlation_preserved_on_kafka_error(
        self,
        correlation_id: UUID,
    ) -> None:
        """Verify correlation ID is preserved when Kafka operations fail.

        Tests that Kafka publish and consume errors properly preserve
        correlation IDs for distributed tracing.

        Args:
            correlation_id: Test correlation ID from conftest fixture
        """
        # TODO(OMN-1349): Implement when Kafka/event bus fixtures available
        # Required fixtures: kafka_producer or event_bus adapter with error injection
        # Implementation should:
        # 1. Trigger a Kafka error (broker unavailable, topic doesn't exist, etc.)
        # 2. Catch InfraUnavailableError or InfraConnectionError
        # 3. Verify error.correlation_id == correlation_id
        # 4. Verify error.model.context["transport_type"] == EnumInfraTransportType.KAFKA
        # See TestCorrelationErrorContext for error context verification patterns
        pytest.skip(
            "Requires real Kafka/Redpanda - implement when event bus fixtures available"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestCorrelationDatabase",
    "TestCorrelationErrorContext",
    "TestCorrelationHttpBoundary",
    "TestCorrelationKafka",
]
