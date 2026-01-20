# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for ServiceMCPToolDiscovery against real Consul infrastructure.

These tests validate ServiceMCPToolDiscovery behavior against actual Consul
infrastructure running on the remote infrastructure server. They require Consul
to be available and will be skipped gracefully if Consul is not reachable.

CI/CD Graceful Skip Behavior
============================

These tests skip gracefully in CI/CD environments without Consul access:

Skip Conditions:
    - Skips if CONSUL_HOST not set
    - Skips if TCP connection to CONSUL_HOST:CONSUL_PORT fails
    - Reachability check performed at module import time with 5-second timeout

Example CI/CD Output::

    $ pytest tests/integration/services/mcp/test_service_mcp_tool_discovery.py -v
    test_discover_all_with_mcp_service SKIPPED (Consul not available)
    test_discover_by_service_id SKIPPED (Consul not available)

Test Categories
===============

- Discovery Tests: Validate discover_all() and discover_by_service_id()
- Tag Filtering Tests: Validate _is_mcp_orchestrator() behavior
- Tool Name Extraction Tests: Validate _extract_tool_name() behavior
- Error Handling Tests: Validate error scenarios

Environment Variables
=====================

    CONSUL_HOST: Consul server hostname (required - skip if not set)
        Example: localhost or 192.168.86.200
    CONSUL_PORT: Consul server port (default: 8500 or 28500 for remote)
    CONSUL_SCHEME: HTTP scheme (default: http)
    CONSUL_TOKEN: Optional ACL token for authentication

Related Ticket: OMN-1281
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import consul
import pytest

from tests.integration.handlers.conftest import (
    CONSUL_AVAILABLE,
    CONSUL_HOST,
    CONSUL_PORT,
    CONSUL_SCHEME,
    CONSUL_TOKEN,
)

if TYPE_CHECKING:
    from omnibase_infra.services.mcp.service_mcp_tool_discovery import (
        ServiceMCPToolDiscovery,
    )

# Module-level logger for test diagnostics
logger = logging.getLogger(__name__)

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Module-level markers - skip all tests if Consul is not available
pytestmark = [
    pytest.mark.skipif(
        not CONSUL_AVAILABLE,
        reason="Consul not available (cannot connect to remote infrastructure)",
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mcp_discovery_service() -> ServiceMCPToolDiscovery:
    """Create ServiceMCPToolDiscovery instance configured for test infrastructure.

    Returns:
        Configured ServiceMCPToolDiscovery instance.
    """
    from omnibase_infra.services.mcp.service_mcp_tool_discovery import (
        ServiceMCPToolDiscovery,
    )

    return ServiceMCPToolDiscovery(
        consul_host=CONSUL_HOST or "localhost",
        consul_port=CONSUL_PORT,
        consul_scheme=CONSUL_SCHEME,
        consul_token=CONSUL_TOKEN,
    )


@pytest.fixture
def consul_client() -> consul.Consul:
    """Create a Consul client for test service registration.

    Returns:
        Consul client instance.
    """
    return consul.Consul(
        host=CONSUL_HOST or "localhost",
        port=CONSUL_PORT,
        scheme=CONSUL_SCHEME,
        token=CONSUL_TOKEN,
    )


@pytest.fixture
def unique_service_id() -> str:
    """Generate unique service ID for test isolation.

    Returns:
        Unique service ID prefixed with test namespace.
    """
    return f"test-mcp-svc-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def unique_service_name() -> str:
    """Generate unique service name for test isolation.

    Returns:
        Unique service name prefixed with test namespace.
    """
    return f"test-mcp-orchestrator-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def unique_tool_name() -> str:
    """Generate unique tool name for test isolation.

    Returns:
        Unique tool name.
    """
    return f"test_tool_{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def registered_mcp_service(
    consul_client: consul.Consul,
    unique_service_id: str,
    unique_service_name: str,
    unique_tool_name: str,
) -> AsyncGenerator[dict[str, str], None]:
    """Register a test MCP-enabled orchestrator service in Consul.

    Registers the service, yields service details, then cleans up.

    Cleanup Behavior:
        - Deregisters the service after test completion
        - Ignores cleanup errors to prevent test pollution

    Yields:
        Dict with service_id, service_name, and tool_name keys.
    """
    # Register test service with MCP tags
    consul_client.agent.service.register(
        name=unique_service_name,
        service_id=unique_service_id,
        address="127.0.0.1",
        port=8080,
        tags=[
            "mcp-enabled",
            "node-type:orchestrator",
            f"mcp-tool:{unique_tool_name}",
            "integration-test",
        ],
    )

    logger.info(
        "Registered test MCP service: %s (id: %s, tool: %s)",
        unique_service_name,
        unique_service_id,
        unique_tool_name,
    )

    yield {
        "service_id": unique_service_id,
        "service_name": unique_service_name,
        "tool_name": unique_tool_name,
    }

    # Cleanup: deregister test service
    try:
        consul_client.agent.service.deregister(unique_service_id)
        logger.info("Deregistered test MCP service: %s", unique_service_id)
    except Exception as e:
        logger.warning(
            "Cleanup failed for test service %s: %s",
            unique_service_id,
            e,
        )


@pytest.fixture
async def registered_non_mcp_service(
    consul_client: consul.Consul,
    unique_service_id: str,
) -> AsyncGenerator[str, None]:
    """Register a test service without MCP tags.

    This service should be ignored by discovery.

    Yields:
        Service ID.
    """
    service_name = f"test-non-mcp-svc-{uuid.uuid4().hex[:8]}"

    # Register service WITHOUT mcp-enabled tag
    consul_client.agent.service.register(
        name=service_name,
        service_id=unique_service_id,
        address="127.0.0.1",
        port=9090,
        tags=[
            "node-type:orchestrator",
            "some-other-tag",
            "integration-test",
        ],
    )

    logger.info("Registered non-MCP service: %s", unique_service_id)

    yield unique_service_id

    # Cleanup
    try:
        consul_client.agent.service.deregister(unique_service_id)
        logger.info("Deregistered non-MCP service: %s", unique_service_id)
    except Exception as e:
        logger.warning("Cleanup failed for service %s: %s", unique_service_id, e)


@pytest.fixture
async def registered_non_orchestrator_service(
    consul_client: consul.Consul,
) -> AsyncGenerator[str, None]:
    """Register a test service that is MCP-enabled but not an orchestrator.

    This service should be ignored by discovery because it lacks
    node-type:orchestrator tag.

    Yields:
        Service ID.
    """
    service_id = f"test-non-orch-svc-{uuid.uuid4().hex[:12]}"
    service_name = f"test-non-orch-{uuid.uuid4().hex[:8]}"

    # Register service WITH mcp-enabled but WITHOUT node-type:orchestrator
    consul_client.agent.service.register(
        name=service_name,
        service_id=service_id,
        address="127.0.0.1",
        port=7070,
        tags=[
            "mcp-enabled",
            "node-type:effect",  # Not an orchestrator
            "mcp-tool:should_be_ignored",
            "integration-test",
        ],
    )

    logger.info("Registered non-orchestrator service: %s", service_id)

    yield service_id

    # Cleanup
    try:
        consul_client.agent.service.deregister(service_id)
        logger.info("Deregistered non-orchestrator service: %s", service_id)
    except Exception as e:
        logger.warning("Cleanup failed for service %s: %s", service_id, e)


@pytest.fixture
async def registered_mcp_service_without_tool_tag(
    consul_client: consul.Consul,
) -> AsyncGenerator[str, None]:
    """Register a test MCP-enabled orchestrator WITHOUT mcp-tool tag.

    This service should be skipped by discovery because it lacks
    the mcp-tool:{name} tag.

    Yields:
        Service ID.
    """
    service_id = f"test-no-tool-svc-{uuid.uuid4().hex[:12]}"
    service_name = f"test-no-tool-{uuid.uuid4().hex[:8]}"

    # Register service WITH mcp-enabled and node-type:orchestrator
    # but WITHOUT mcp-tool:{name} tag
    consul_client.agent.service.register(
        name=service_name,
        service_id=service_id,
        address="127.0.0.1",
        port=6060,
        tags=[
            "mcp-enabled",
            "node-type:orchestrator",
            # Missing: mcp-tool:{name}
            "integration-test",
        ],
    )

    logger.info("Registered MCP service without tool tag: %s", service_id)

    yield service_id

    # Cleanup
    try:
        consul_client.agent.service.deregister(service_id)
        logger.info("Deregistered service without tool tag: %s", service_id)
    except Exception as e:
        logger.warning("Cleanup failed for service %s: %s", service_id, e)


# =============================================================================
# Discovery Tests - discover_all()
# =============================================================================


class TestDiscoverAll:
    """Tests for ServiceMCPToolDiscovery.discover_all() method."""

    @pytest.mark.asyncio
    async def test_discover_all_finds_mcp_service(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
        registered_mcp_service: dict[str, str],
    ) -> None:
        """Test discover_all() finds service with correct MCP tags.

        Verifies that:
        - MCP-enabled orchestrator services are discovered
        - Tool name is correctly extracted from tags
        - Service metadata is populated
        """
        tools = await mcp_discovery_service.discover_all()

        # Find our test tool
        test_tool = next(
            (t for t in tools if t.name == registered_mcp_service["tool_name"]),
            None,
        )

        assert test_tool is not None, (
            f"Expected to find tool '{registered_mcp_service['tool_name']}' "
            f"in discovered tools: {[t.name for t in tools]}"
        )

        # Verify tool properties
        assert test_tool.name == registered_mcp_service["tool_name"]
        assert registered_mcp_service["service_name"] in test_tool.description
        assert test_tool.orchestrator_service_id == registered_mcp_service["service_id"]
        assert test_tool.metadata.get("source") == "consul_discovery"
        assert "mcp-enabled" in test_tool.metadata.get("tags", [])

    @pytest.mark.asyncio
    async def test_discover_all_ignores_non_mcp_service(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
        registered_non_mcp_service: str,
    ) -> None:
        """Test discover_all() ignores services without mcp-enabled tag.

        Verifies that:
        - Services without mcp-enabled tag are not discovered
        - Only MCP-enabled orchestrators appear in results
        """
        tools = await mcp_discovery_service.discover_all()

        # Verify the non-MCP service is not included
        non_mcp_tools = [
            t for t in tools if t.orchestrator_service_id == registered_non_mcp_service
        ]

        assert len(non_mcp_tools) == 0, (
            f"Non-MCP service should not be discovered: {registered_non_mcp_service}"
        )

    @pytest.mark.asyncio
    async def test_discover_all_ignores_non_orchestrator_service(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
        registered_non_orchestrator_service: str,
    ) -> None:
        """Test discover_all() ignores services without node-type:orchestrator tag.

        Verifies that:
        - MCP-enabled services without orchestrator tag are not discovered
        - Only orchestrator nodes can be exposed as MCP tools
        """
        tools = await mcp_discovery_service.discover_all()

        # Verify the non-orchestrator service is not included
        non_orch_tools = [
            t
            for t in tools
            if t.orchestrator_service_id == registered_non_orchestrator_service
        ]

        assert len(non_orch_tools) == 0, (
            "Non-orchestrator service should not be discovered: "
            f"{registered_non_orchestrator_service}"
        )

    @pytest.mark.asyncio
    async def test_discover_all_skips_service_without_tool_tag(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
        registered_mcp_service_without_tool_tag: str,
    ) -> None:
        """Test discover_all() skips services missing mcp-tool tag.

        Verifies that:
        - MCP-enabled orchestrators without mcp-tool tag are skipped
        - A warning is logged (not tested here, but documented)
        """
        tools = await mcp_discovery_service.discover_all()

        # Verify the service without tool tag is not included
        no_tool_services = [
            t
            for t in tools
            if t.orchestrator_service_id == registered_mcp_service_without_tool_tag
        ]

        assert len(no_tool_services) == 0, (
            "Service without mcp-tool tag should not be discovered: "
            f"{registered_mcp_service_without_tool_tag}"
        )

    @pytest.mark.asyncio
    async def test_discover_all_returns_empty_list_when_no_mcp_services(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test discover_all() returns empty list when no MCP services registered.

        Verifies that:
        - Empty list is returned (not None, not error)
        - No exception is raised
        """
        # Note: There may be pre-existing services in Consul, so we just verify
        # the return type is a list (not None or exception)
        tools = await mcp_discovery_service.discover_all()

        assert isinstance(tools, list), f"Expected list, got {type(tools)}"

    @pytest.mark.asyncio
    async def test_discover_all_populates_endpoint(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
        registered_mcp_service: dict[str, str],
    ) -> None:
        """Test discover_all() populates endpoint from service instance.

        Verifies that:
        - Endpoint is populated from healthy service instances
        - Endpoint format is http://{address}:{port}
        """
        tools = await mcp_discovery_service.discover_all()

        test_tool = next(
            (t for t in tools if t.name == registered_mcp_service["tool_name"]),
            None,
        )

        assert test_tool is not None
        # Our test service was registered at 127.0.0.1:8080
        assert test_tool.endpoint is not None
        assert "127.0.0.1" in test_tool.endpoint
        assert "8080" in test_tool.endpoint


# =============================================================================
# Discovery Tests - discover_by_service_id()
# =============================================================================


class TestDiscoverByServiceId:
    """Tests for ServiceMCPToolDiscovery.discover_by_service_id() method."""

    @pytest.mark.asyncio
    async def test_discover_by_service_id_finds_service(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
        registered_mcp_service: dict[str, str],
    ) -> None:
        """Test discover_by_service_id() finds service by ID.

        Verifies that:
        - Service can be looked up by ID
        - Correct tool definition is returned
        """
        tool = await mcp_discovery_service.discover_by_service_id(
            registered_mcp_service["service_id"]
        )

        assert tool is not None
        assert tool.name == registered_mcp_service["tool_name"]
        assert tool.orchestrator_service_id == registered_mcp_service["service_id"]

    @pytest.mark.asyncio
    async def test_discover_by_service_id_returns_none_for_unknown_id(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test discover_by_service_id() returns None for unknown ID.

        Verifies that:
        - None is returned for non-existent service ID
        - No exception is raised
        """
        tool = await mcp_discovery_service.discover_by_service_id(
            "nonexistent-service-id-12345"
        )

        assert tool is None

    @pytest.mark.asyncio
    async def test_discover_by_service_id_returns_none_for_non_mcp_service(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
        registered_non_mcp_service: str,
    ) -> None:
        """Test discover_by_service_id() returns None for non-MCP service.

        Verifies that:
        - Services without MCP tags are not returned
        - None is returned instead of error
        """
        tool = await mcp_discovery_service.discover_by_service_id(
            registered_non_mcp_service
        )

        assert tool is None

    @pytest.mark.asyncio
    async def test_discover_by_service_id_returns_none_for_non_orchestrator(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
        registered_non_orchestrator_service: str,
    ) -> None:
        """Test discover_by_service_id() returns None for non-orchestrator.

        Verifies that:
        - MCP-enabled non-orchestrators are not returned
        """
        tool = await mcp_discovery_service.discover_by_service_id(
            registered_non_orchestrator_service
        )

        assert tool is None

    @pytest.mark.asyncio
    async def test_discover_by_service_id_returns_none_without_tool_tag(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
        registered_mcp_service_without_tool_tag: str,
    ) -> None:
        """Test discover_by_service_id() returns None when mcp-tool tag missing.

        Verifies that:
        - Services without mcp-tool:{name} tag return None
        """
        tool = await mcp_discovery_service.discover_by_service_id(
            registered_mcp_service_without_tool_tag
        )

        assert tool is None


# =============================================================================
# Tag Filtering Tests - _is_mcp_orchestrator()
# =============================================================================


class TestIsMCPOrchestrator:
    """Tests for ServiceMCPToolDiscovery._is_mcp_orchestrator() method."""

    def test_is_mcp_orchestrator_with_both_tags(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test _is_mcp_orchestrator() returns True with both required tags.

        Verifies that:
        - Both mcp-enabled and node-type:orchestrator tags are required
        """
        tags = ["mcp-enabled", "node-type:orchestrator", "other-tag"]

        result = mcp_discovery_service._is_mcp_orchestrator(tags)

        assert result is True

    def test_is_mcp_orchestrator_missing_mcp_enabled(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test _is_mcp_orchestrator() returns False without mcp-enabled tag.

        Verifies that:
        - Returns False when mcp-enabled tag is missing
        """
        tags = ["node-type:orchestrator", "other-tag"]

        result = mcp_discovery_service._is_mcp_orchestrator(tags)

        assert result is False

    def test_is_mcp_orchestrator_missing_orchestrator_type(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test _is_mcp_orchestrator() returns False without orchestrator tag.

        Verifies that:
        - Returns False when node-type:orchestrator tag is missing
        """
        tags = ["mcp-enabled", "node-type:effect", "other-tag"]

        result = mcp_discovery_service._is_mcp_orchestrator(tags)

        assert result is False

    def test_is_mcp_orchestrator_empty_tags(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test _is_mcp_orchestrator() returns False for empty tags.

        Verifies that:
        - Returns False for empty tag list
        """
        tags: list[str] = []

        result = mcp_discovery_service._is_mcp_orchestrator(tags)

        assert result is False


# =============================================================================
# Tool Name Extraction Tests - _extract_tool_name()
# =============================================================================


class TestExtractToolName:
    """Tests for ServiceMCPToolDiscovery._extract_tool_name() method."""

    def test_extract_tool_name_success(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test _extract_tool_name() extracts name from mcp-tool tag.

        Verifies that:
        - Tool name is correctly extracted from mcp-tool:{name} tag
        """
        tags = ["mcp-enabled", "mcp-tool:my_test_tool", "other-tag"]

        result = mcp_discovery_service._extract_tool_name(tags)

        assert result == "my_test_tool"

    def test_extract_tool_name_with_underscores(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test _extract_tool_name() handles tool names with underscores.

        Verifies that:
        - Tool names with underscores are preserved
        """
        tags = ["mcp-tool:my_complex_tool_name"]

        result = mcp_discovery_service._extract_tool_name(tags)

        assert result == "my_complex_tool_name"

    def test_extract_tool_name_with_dashes(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test _extract_tool_name() handles tool names with dashes.

        Verifies that:
        - Tool names with dashes are preserved
        """
        tags = ["mcp-tool:my-tool-with-dashes"]

        result = mcp_discovery_service._extract_tool_name(tags)

        assert result == "my-tool-with-dashes"

    def test_extract_tool_name_missing_tag(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test _extract_tool_name() returns None when tag is missing.

        Verifies that:
        - Returns None when no mcp-tool tag exists
        """
        tags = ["mcp-enabled", "node-type:orchestrator", "other-tag"]

        result = mcp_discovery_service._extract_tool_name(tags)

        assert result is None

    def test_extract_tool_name_empty_tags(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test _extract_tool_name() returns None for empty tags.

        Verifies that:
        - Returns None for empty tag list
        """
        tags: list[str] = []

        result = mcp_discovery_service._extract_tool_name(tags)

        assert result is None

    def test_extract_tool_name_first_match(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test _extract_tool_name() returns first mcp-tool tag value.

        Verifies that:
        - First matching mcp-tool tag is used if multiple exist
        """
        tags = ["mcp-tool:first_tool", "mcp-tool:second_tool"]

        result = mcp_discovery_service._extract_tool_name(tags)

        assert result == "first_tool"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for ServiceMCPToolDiscovery error handling."""

    @pytest.mark.asyncio
    async def test_discover_all_connection_error(self) -> None:
        """Test discover_all() raises InfraConnectionError on Consul failure.

        Verifies that:
        - InfraConnectionError is raised when Consul is unreachable
        - Error context includes transport type and operation
        """
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.services.mcp.service_mcp_tool_discovery import (
            ServiceMCPToolDiscovery,
        )

        # Create service pointing to invalid Consul address
        bad_service = ServiceMCPToolDiscovery(
            consul_host="invalid-host-that-does-not-exist.local",
            consul_port=9999,
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await bad_service.discover_all()

        # Verify error context
        assert exc_info.value.model.context is not None
        assert exc_info.value.model.context.get("operation") == "discover_all"

    @pytest.mark.asyncio
    async def test_discover_by_service_id_connection_error(self) -> None:
        """Test discover_by_service_id() raises InfraConnectionError.

        Verifies that:
        - InfraConnectionError is raised when Consul is unreachable
        """
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.services.mcp.service_mcp_tool_discovery import (
            ServiceMCPToolDiscovery,
        )

        # Create service pointing to invalid Consul address
        bad_service = ServiceMCPToolDiscovery(
            consul_host="invalid-host-that-does-not-exist.local",
            consul_port=9999,
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await bad_service.discover_by_service_id("any-service-id")

        # Verify error context
        assert exc_info.value.model.context is not None
        assert exc_info.value.model.context.get("operation") == "discover_by_service_id"


# =============================================================================
# Service Metadata Tests
# =============================================================================


class TestServiceMetadata:
    """Tests for ServiceMCPToolDiscovery.describe() method."""

    def test_describe_returns_service_metadata(
        self,
        mcp_discovery_service: ServiceMCPToolDiscovery,
    ) -> None:
        """Test describe() returns correct service metadata.

        Verifies that:
        - Service name is returned
        - Consul connection details are included
        """
        metadata = mcp_discovery_service.describe()

        assert metadata["service_name"] == "ServiceMCPToolDiscovery"
        assert metadata["consul_host"] == (CONSUL_HOST or "localhost")
        assert metadata["consul_port"] == CONSUL_PORT
        assert metadata["consul_scheme"] == CONSUL_SCHEME
