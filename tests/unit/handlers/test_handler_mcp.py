# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S104
# S104 disabled: Binding to 0.0.0.0 is tested for container networking patterns
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for HandlerMCP.

Comprehensive test suite covering initialization, MCP operations,
tool registration, and lifecycle management.

Tests focus on observable behavior via public APIs (describe, health_check,
execute) rather than directly accessing internal state where possible.

Configuration Policy (per CLAUDE.md):
    All tests must provide complete configuration dictionaries. The handler
    does not use hardcoded fallbacks - .env is the single source of truth
    for production. Tests explicitly provide all required config values to
    verify the handler works correctly with explicit configuration.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import (
    InfraUnavailableError,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_mcp import HandlerMCP
from omnibase_infra.handlers.mcp.adapter_onex_to_mcp import MCPToolDefinition
from omnibase_infra.handlers.models.mcp import (
    EnumMcpOperationType,
    ModelMcpHandlerConfig,
)

# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def mcp_test_config() -> dict[str, object]:
    """Standard MCP handler config for unit tests.

    Provides all required config values explicitly to avoid relying on
    hardcoded fallbacks. Uses skip_server=True to avoid port binding
    during unit tests.

    Per CLAUDE.md: .env is the single source of truth. Tests must provide
    explicit configuration values rather than relying on defaults.

    Returns:
        Complete configuration dictionary for HandlerMCP.initialize().
    """
    return {
        "skip_server": True,
        "kafka_enabled": False,
        "dev_mode": True,
    }


@pytest.fixture
def mcp_custom_config() -> dict[str, object]:
    """Custom MCP handler config with non-default server values.

    Provides complete configuration with custom host/port/path values
    for testing configuration parsing.

    Returns:
        Complete configuration dictionary with custom server settings.
    """
    return {
        "host": "127.0.0.1",
        "port": 9000,
        "path": "/api/mcp",
        "timeout_seconds": 60.0,
        "max_tools": 50,
        "skip_server": True,
        "kafka_enabled": False,
        "dev_mode": True,
    }


class TestHandlerMCPInitialization:
    """Test suite for HandlerMCP initialization.

    Note:
        Some tests in this class access internal state (attributes prefixed with _)
        to verify initialization behavior. This is appropriate for unit tests that
        need to verify internal invariants. Integration tests should prefer testing
        via public APIs (health_check, describe, execute).
    """

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerMCP:
        """Create HandlerMCP fixture with mock container."""
        return HandlerMCP(container=mock_container)

    def test_handler_init_default_state(self, handler: HandlerMCP) -> None:
        """Test handler initializes in uninitialized state.

        Note:
            This unit test accesses internal state (_initialized, _config,
            _tool_registry) to verify initialization invariants. Integration
            tests should prefer testing via public APIs (health_check, describe).
        """
        assert handler._initialized is False
        assert handler._config is None
        assert handler._tool_registry == {}

    def test_handler_stores_container(
        self, handler: HandlerMCP, mock_container: MagicMock
    ) -> None:
        """Test handler stores container reference for dependency injection.

        Note:
            This unit test accesses internal _container attribute to verify
            dependency injection worked correctly. This is necessary because
            the container is not exposed via public API.
        """
        assert handler._container is mock_container

    def test_handler_type_returns_infra_handler(self, handler: HandlerMCP) -> None:
        """Test handler_type property returns EnumHandlerType.INFRA_HANDLER."""
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category_returns_effect(self, handler: HandlerMCP) -> None:
        """Test handler_category property returns EnumHandlerTypeCategory.EFFECT."""
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    def test_transport_type_returns_mcp(self, handler: HandlerMCP) -> None:
        """Test transport_type property returns EnumInfraTransportType.MCP."""
        assert handler.transport_type == EnumInfraTransportType.MCP

    @pytest.mark.asyncio
    async def test_initialize_with_test_config(
        self, handler: HandlerMCP, mcp_test_config: dict[str, object]
    ) -> None:
        """Test handler initializes with standard test config.

        Note:
            This unit test accesses internal state (_initialized, _config) to
            verify that configuration values are correctly applied.
            This validates the initialization contract that cannot be fully
            tested via public APIs alone.

            Uses mcp_test_config fixture which provides all required config
            values explicitly (skip_server=True to avoid port binding).
        """
        await handler.initialize(mcp_test_config)

        assert handler._initialized is True
        assert handler._config is not None
        assert handler._config.host == "0.0.0.0"
        assert handler._config.port == 8090
        assert handler._config.path == "/mcp"
        assert handler._config.stateless is True
        assert handler._config.json_response is True

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_config(
        self, handler: HandlerMCP, mcp_custom_config: dict[str, object]
    ) -> None:
        """Test handler initializes with custom configuration.

        Note:
            This unit test accesses internal state (_initialized, _config) to
            verify that custom configuration values are correctly applied.
            This validates the initialization contract that cannot be fully
            tested via public APIs alone.

            Uses mcp_custom_config fixture which provides complete config
            with custom server settings (skip_server=True to avoid port binding).
        """
        await handler.initialize(mcp_custom_config)

        assert handler._initialized is True
        assert handler._config is not None
        assert handler._config.host == "127.0.0.1"
        assert handler._config.port == 9000
        assert handler._config.path == "/api/mcp"
        assert handler._config.timeout_seconds == 60.0
        assert handler._config.max_tools == 50

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_clears_state(
        self, handler: HandlerMCP, mcp_test_config: dict[str, object]
    ) -> None:
        """Test shutdown clears handler state.

        Note:
            This unit test accesses internal state (_initialized, _config,
            _tool_registry) to verify that shutdown properly clears all state.
            This is a critical invariant that must be verified at the unit level
            to ensure no state leaks between handler lifecycles.

            Uses mcp_test_config fixture which provides all required config
            values explicitly (skip_server=True to avoid port binding).
        """
        await handler.initialize(mcp_test_config)
        assert handler._initialized is True

        await handler.shutdown()

        assert handler._initialized is False
        assert handler._config is None
        assert handler._tool_registry == {}


class TestHandlerMCPDescribe:
    """Test suite for describe operation."""

    @pytest.fixture
    async def initialized_handler(
        self, mock_container: MagicMock, mcp_test_config: dict[str, object]
    ) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture with mock container.

        Uses mcp_test_config fixture which provides all required config
        values explicitly (skip_server=True to avoid port binding).
        """
        handler = HandlerMCP(container=mock_container)
        await handler.initialize(mcp_test_config)
        yield handler
        await handler.shutdown()

    def test_describe_returns_metadata(self, initialized_handler: HandlerMCP) -> None:
        """Test describe returns handler metadata."""
        description = initialized_handler.describe()

        assert description["handler_type"] == "infra_handler"
        assert description["handler_category"] == "effect"
        assert description["transport_type"] == "mcp"
        assert "mcp.list_tools" in description["supported_operations"]
        assert "mcp.call_tool" in description["supported_operations"]
        assert "mcp.describe" in description["supported_operations"]
        assert description["initialized"] is True
        assert description["tool_count"] == 0
        assert description["version"] == "0.1.0-mvp"


class TestHandlerMCPListTools:
    """Test suite for list_tools operation."""

    @pytest.fixture
    async def initialized_handler(
        self, mock_container: MagicMock, mcp_test_config: dict[str, object]
    ) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture with mock container.

        Uses mcp_test_config fixture which provides all required config
        values explicitly (skip_server=True to avoid port binding).
        """
        handler = HandlerMCP(container=mock_container)
        await handler.initialize(mcp_test_config)
        yield handler
        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_list_tools_empty_registry(
        self, initialized_handler: HandlerMCP
    ) -> None:
        """Test list_tools returns empty list when no tools registered."""
        envelope = {
            "operation": EnumMcpOperationType.LIST_TOOLS.value,
            "payload": {},
            "correlation_id": str(uuid4()),
        }

        result = await initialized_handler.execute(envelope)

        assert result.result["status"] == "success"
        assert result.result["payload"]["tools"] == []

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises(
        self, mock_container: MagicMock
    ) -> None:
        """Test execute raises RuntimeHostError if not initialized."""
        handler = HandlerMCP(container=mock_container)
        envelope = {
            "operation": EnumMcpOperationType.LIST_TOOLS.value,
            "payload": {},
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "not initialized" in str(exc_info.value).lower()


class TestHandlerMCPCallTool:
    """Test suite for call_tool operation."""

    @pytest.fixture
    async def initialized_handler(
        self, mock_container: MagicMock, mcp_test_config: dict[str, object]
    ) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture with mock container.

        Uses mcp_test_config fixture which provides all required config
        values explicitly (skip_server=True to avoid port binding).
        """
        handler = HandlerMCP(container=mock_container)
        await handler.initialize(mcp_test_config)
        yield handler
        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, initialized_handler: HandlerMCP) -> None:
        """Test call_tool raises error for unregistered tool."""
        envelope = {
            "operation": EnumMcpOperationType.CALL_TOOL.value,
            "payload": {
                "tool_name": "nonexistent_tool",
                "arguments": {},
            },
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(InfraUnavailableError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_call_tool_missing_tool_name(
        self, initialized_handler: HandlerMCP
    ) -> None:
        """Test call_tool raises error when tool_name missing."""
        envelope = {
            "operation": EnumMcpOperationType.CALL_TOOL.value,
            "payload": {
                "arguments": {},
            },
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "tool_name" in str(exc_info.value).lower()


class TestHandlerMCPOperationValidation:
    """Test suite for operation validation."""

    @pytest.fixture
    async def initialized_handler(
        self, mock_container: MagicMock, mcp_test_config: dict[str, object]
    ) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture with mock container.

        Uses mcp_test_config fixture which provides all required config
        values explicitly (skip_server=True to avoid port binding).
        """
        handler = HandlerMCP(container=mock_container)
        await handler.initialize(mcp_test_config)
        yield handler
        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_unsupported_operation_raises(
        self, initialized_handler: HandlerMCP
    ) -> None:
        """Test execute raises error for unsupported operation."""
        envelope = {
            "operation": "mcp.unsupported",
            "payload": {},
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "not supported" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_operation_raises(
        self, initialized_handler: HandlerMCP
    ) -> None:
        """Test execute raises error when operation missing."""
        envelope = {
            "payload": {},
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "operation" in str(exc_info.value).lower()


class TestHandlerMCPHealthCheck:
    """Test suite for health check."""

    @pytest.fixture
    async def initialized_handler(
        self, mock_container: MagicMock, mcp_test_config: dict[str, object]
    ) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture with mock container.

        Uses mcp_test_config fixture which provides all required config
        values explicitly (skip_server=True to avoid port binding).
        """
        handler = HandlerMCP(container=mock_container)
        await handler.initialize(mcp_test_config)
        yield handler
        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_initialized(
        self, initialized_handler: HandlerMCP
    ) -> None:
        """Test health check returns healthy when initialized with skip_server."""
        health = await initialized_handler.health_check()

        assert health["healthy"] is True
        assert health["skip_server"] is True
        assert health["transport_type"] == "mcp"

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(
        self, mock_container: MagicMock
    ) -> None:
        """Test health check returns unhealthy when not initialized."""
        handler = HandlerMCP(container=mock_container)
        health = await handler.health_check()

        assert health["healthy"] is False
        assert health["reason"] == "not_initialized"
        assert health["transport_type"] == "mcp"


class TestMcpHandlerConfig:
    """Test suite for ModelMcpHandlerConfig."""

    def test_config_defaults(self) -> None:
        """Test config has correct defaults."""
        config = ModelMcpHandlerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8090
        assert config.path == "/mcp"
        assert config.stateless is True
        assert config.json_response is True
        assert config.timeout_seconds == 30.0
        assert config.max_tools == 100

    def test_config_custom_values(self) -> None:
        """Test config accepts custom values."""
        config = ModelMcpHandlerConfig(
            host="localhost",
            port=9000,
            path="/api/v1/mcp",
            stateless=False,
            json_response=False,
            timeout_seconds=60.0,
            max_tools=50,
        )

        assert config.host == "localhost"
        assert config.port == 9000
        assert config.path == "/api/v1/mcp"
        assert config.stateless is False
        assert config.json_response is False
        assert config.timeout_seconds == 60.0
        assert config.max_tools == 50

    def test_config_is_frozen(self) -> None:
        """Test config is immutable (frozen)."""
        config = ModelMcpHandlerConfig()

        with pytest.raises(ValidationError):
            config.host = "changed"


class TestHandlerMCPDescribeOperation:
    """Test suite for mcp.describe operation via execute method."""

    @pytest.fixture
    async def initialized_handler(
        self, mock_container: MagicMock, mcp_test_config: dict[str, object]
    ) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture with mock container.

        Uses mcp_test_config fixture which provides all required config
        values explicitly (skip_server=True to avoid port binding).
        """
        handler = HandlerMCP(container=mock_container)
        await handler.initialize(mcp_test_config)
        yield handler
        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_describe_operation_returns_success(
        self, initialized_handler: HandlerMCP
    ) -> None:
        """Test mcp.describe operation via execute returns success with metadata."""
        envelope = {
            "operation": EnumMcpOperationType.DESCRIBE.value,
            "payload": {},
            "correlation_id": str(uuid4()),
        }

        result = await initialized_handler.execute(envelope)

        assert result.result["status"] == "success"
        payload = result.result["payload"]
        assert payload["handler_type"] == "infra_handler"
        assert payload["transport_type"] == "mcp"
        assert payload["initialized"] is True

    @pytest.mark.asyncio
    async def test_describe_operation_includes_correlation_id(
        self, initialized_handler: HandlerMCP
    ) -> None:
        """Test mcp.describe operation includes correlation_id in response."""
        correlation_id = str(uuid4())
        envelope = {
            "operation": EnumMcpOperationType.DESCRIBE.value,
            "payload": {},
            "correlation_id": correlation_id,
        }

        result = await initialized_handler.execute(envelope)

        assert result.result["correlation_id"] == correlation_id


class TestHandlerMCPLifecycle:
    """Test suite for handler lifecycle transitions."""

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerMCP:
        """Create HandlerMCP fixture with mock container."""
        return HandlerMCP(container=mock_container)

    @pytest.mark.asyncio
    async def test_lifecycle_transition_healthy_after_init(
        self, handler: HandlerMCP, mcp_test_config: dict[str, object]
    ) -> None:
        """Test handler becomes healthy after initialization via health_check.

        Uses mcp_test_config fixture which provides all required config
        values explicitly (skip_server=True to avoid port binding).
        """
        # Before init - unhealthy
        health_before = await handler.health_check()
        assert health_before["healthy"] is False

        # After init - healthy
        await handler.initialize(mcp_test_config)
        health_after = await handler.health_check()
        assert health_after["healthy"] is True

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_lifecycle_transition_unhealthy_after_shutdown(
        self, handler: HandlerMCP, mcp_test_config: dict[str, object]
    ) -> None:
        """Test handler becomes unhealthy after shutdown via health_check.

        Uses mcp_test_config fixture which provides all required config
        values explicitly (skip_server=True to avoid port binding).
        """
        await handler.initialize(mcp_test_config)

        # Before shutdown - healthy
        health_before = await handler.health_check()
        assert health_before["healthy"] is True

        # After shutdown - unhealthy
        await handler.shutdown()
        health_after = await handler.health_check()
        assert health_after["healthy"] is False

    @pytest.mark.asyncio
    async def test_describe_reflects_initialization_state(
        self, handler: HandlerMCP, mcp_test_config: dict[str, object]
    ) -> None:
        """Test describe reflects initialization state correctly.

        Uses mcp_test_config fixture which provides all required config
        values explicitly (skip_server=True to avoid port binding).
        """
        # Before init
        desc_before = handler.describe()
        assert desc_before["initialized"] is False

        # After init
        await handler.initialize(mcp_test_config)
        desc_after = handler.describe()
        assert desc_after["initialized"] is True

        # After shutdown
        await handler.shutdown()
        desc_final = handler.describe()
        assert desc_final["initialized"] is False


class TestHandlerMCPConfigValidation:
    """Test suite for configuration validation.

    Per CLAUDE.md: .env is the single source of truth for all configuration.
    No fallbacks are allowed. Tests verify that missing required config
    raises appropriate errors.
    """

    @pytest.fixture
    def handler(self, mock_container: MagicMock) -> HandlerMCP:
        """Create HandlerMCP fixture with mock container."""
        return HandlerMCP(container=mock_container)

    @pytest.mark.asyncio
    async def test_initialize_with_skip_server_only_succeeds(
        self, handler: HandlerMCP
    ) -> None:
        """Test that skip_server=True allows initialization without full config.

        When skip_server=True, the handler skips MCPServerLifecycle initialization
        which requires kafka_enabled, dev_mode, etc. Only the Pydantic model
        config validation runs, which has its own defaults for server-specific
        fields (host, port, path, etc.).

        This test verifies the skip_server path works correctly for unit tests.
        """
        # skip_server=True bypasses the code path that needs kafka_enabled etc.
        await handler.initialize({"skip_server": True})

        assert handler._initialized is True
        assert handler._config is not None

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_validates_pydantic_config(
        self, handler: HandlerMCP
    ) -> None:
        """Test that invalid Pydantic config raises ProtocolConfigurationError.

        The handler uses ModelMcpHandlerConfig for validation. Invalid types
        that cannot be coerced should raise ProtocolConfigurationError.
        """
        invalid_config: dict[str, object] = {
            "skip_server": True,
            "port": "not_a_valid_port",  # Should be int, string "not_a_valid_port" cannot be coerced
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize(invalid_config)

        assert "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_with_complete_config_succeeds(
        self, handler: HandlerMCP, mcp_test_config: dict[str, object]
    ) -> None:
        """Test that complete config with all required values succeeds.

        This test verifies that providing all config values explicitly
        (as required per CLAUDE.md) results in successful initialization.
        """
        await handler.initialize(mcp_test_config)

        assert handler._initialized is True
        assert handler._config is not None

        await handler.shutdown()


# =============================================================================
# Input Schema Validation Tests (OMN-2699)
# =============================================================================

# JSON Schema used across the validation test suite.
_SCHEMA_WITH_REQUIRED_STRING = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string", "description": "Name of the model"},
        "temperature": {"type": "number", "description": "Sampling temperature"},
        "mode": {
            "type": "string",
            "enum": ["fast", "precise", "balanced"],
            "description": "Inference mode",
        },
    },
    "required": ["model_name"],
}


def _make_tool_with_schema(
    name: str = "test_tool",
    input_schema: dict[str, object] | None = None,
) -> MCPToolDefinition:
    """Return a minimal MCPToolDefinition with an optional input_schema."""
    return MCPToolDefinition(
        name=name,
        tool_type="function",
        description="A test tool for validation tests.",
        version="1.0.0",
        input_schema=input_schema,
    )


class TestHandlerMCPInputSchemaValidation:
    """Tests for R1/R2: input schema validation before ONEX dispatch (OMN-2699).

    These tests cover:
    - Missing required field is rejected with isError=True (no dispatch)
    - Wrong type for a field is rejected with isError=True
    - Enum value violation is rejected with isError=True
    - Valid arguments pass validation and reach dispatch
    - Tool without input_schema passes through unchanged (backwards-compatible)
    - Error messages identify the failing field (no raw tracebacks)
    """

    @pytest.fixture
    async def handler(
        self, mock_container: MagicMock, mcp_test_config: dict[str, object]
    ) -> HandlerMCP:
        """Initialized HandlerMCP with skip_server=True."""
        h = HandlerMCP(container=mock_container)
        await h.initialize(mcp_test_config)
        yield h
        await h.shutdown()

    # ------------------------------------------------------------------
    # R1: Validation rejects bad inputs
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_missing_required_field_returns_error(
        self, handler: HandlerMCP
    ) -> None:
        """Missing required field should return isError=True, no dispatch."""
        tool = _make_tool_with_schema(
            name="needs_model_name",
            input_schema=_SCHEMA_WITH_REQUIRED_STRING,
        )
        handler._tool_registry["needs_model_name"] = tool  # type: ignore[assignment]

        envelope = {
            "operation": EnumMcpOperationType.CALL_TOOL.value,
            "payload": {
                "tool_name": "needs_model_name",
                "arguments": {},  # model_name is required but absent
            },
            "correlation_id": str(uuid4()),
        }

        result = await handler.execute(envelope)

        assert result.result["status"] == "error"
        payload = result.result["payload"]
        assert payload["is_error"] is True
        assert payload["success"] is False
        # Error message must mention the field
        assert "model_name" in payload["error_message"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_wrong_type_returns_error(self, handler: HandlerMCP) -> None:
        """Wrong type for a field (int instead of string) should return isError=True."""
        tool = _make_tool_with_schema(
            name="type_check_tool",
            input_schema=_SCHEMA_WITH_REQUIRED_STRING,
        )
        handler._tool_registry["type_check_tool"] = tool  # type: ignore[assignment]

        envelope = {
            "operation": EnumMcpOperationType.CALL_TOOL.value,
            "payload": {
                "tool_name": "type_check_tool",
                "arguments": {"model_name": 42},  # must be string
            },
            "correlation_id": str(uuid4()),
        }

        result = await handler.execute(envelope)

        assert result.result["status"] == "error"
        payload = result.result["payload"]
        assert payload["is_error"] is True
        # Error message must name the failing field
        assert "model_name" in payload["error_message"]

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_enum_violation_returns_error(self, handler: HandlerMCP) -> None:
        """Enum value not in allowed set should return isError=True."""
        tool = _make_tool_with_schema(
            name="enum_tool",
            input_schema=_SCHEMA_WITH_REQUIRED_STRING,
        )
        handler._tool_registry["enum_tool"] = tool  # type: ignore[assignment]

        envelope = {
            "operation": EnumMcpOperationType.CALL_TOOL.value,
            "payload": {
                "tool_name": "enum_tool",
                "arguments": {
                    "model_name": "gpt-4",
                    "mode": "turbo",  # not in ["fast", "precise", "balanced"]
                },
            },
            "correlation_id": str(uuid4()),
        }

        result = await handler.execute(envelope)

        assert result.result["status"] == "error"
        payload = result.result["payload"]
        assert payload["is_error"] is True
        # Should mention the mode field
        assert "mode" in payload["error_message"]

    # ------------------------------------------------------------------
    # R1: Valid inputs dispatch successfully
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_valid_input_dispatches_successfully(
        self, handler: HandlerMCP
    ) -> None:
        """Valid arguments must pass validation and reach dispatch (not short-circuit)."""
        tool = _make_tool_with_schema(
            name="valid_tool",
            input_schema=_SCHEMA_WITH_REQUIRED_STRING,
        )
        handler._tool_registry["valid_tool"] = tool  # type: ignore[assignment]

        envelope = {
            "operation": EnumMcpOperationType.CALL_TOOL.value,
            "payload": {
                "tool_name": "valid_tool",
                "arguments": {
                    "model_name": "gpt-4",
                    "temperature": 0.7,
                    "mode": "fast",
                },
            },
            "correlation_id": str(uuid4()),
        }

        result = await handler.execute(envelope)

        # Should reach dispatch (placeholder) and return success
        assert result.result["status"] == "success"
        payload = result.result["payload"]
        assert payload["is_error"] is False

    # ------------------------------------------------------------------
    # R1: Backwards-compatible pass-through when no schema
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_tool_without_schema_passes_through(
        self, handler: HandlerMCP
    ) -> None:
        """Tool with input_schema=None must not validate (pass-through)."""
        tool = _make_tool_with_schema(
            name="no_schema_tool",
            input_schema=None,  # explicitly no schema
        )
        handler._tool_registry["no_schema_tool"] = tool  # type: ignore[assignment]

        envelope = {
            "operation": EnumMcpOperationType.CALL_TOOL.value,
            "payload": {
                "tool_name": "no_schema_tool",
                "arguments": {"garbage": True},  # would fail if validated
            },
            "correlation_id": str(uuid4()),
        }

        result = await handler.execute(envelope)

        # Must succeed (no validation gate applied)
        assert result.result["status"] == "success"

    # ------------------------------------------------------------------
    # R2: Error message quality — no raw Python tracebacks
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_validation_error_message_has_no_traceback(
        self, handler: HandlerMCP
    ) -> None:
        """Validation error messages must not contain raw Python tracebacks."""
        tool = _make_tool_with_schema(
            name="traceback_check_tool",
            input_schema=_SCHEMA_WITH_REQUIRED_STRING,
        )
        handler._tool_registry["traceback_check_tool"] = tool  # type: ignore[assignment]

        envelope = {
            "operation": EnumMcpOperationType.CALL_TOOL.value,
            "payload": {
                "tool_name": "traceback_check_tool",
                "arguments": {},  # missing required field
            },
            "correlation_id": str(uuid4()),
        }

        result = await handler.execute(envelope)

        error_message: str = result.result["payload"]["error_message"]
        # Must NOT contain any Python traceback markers
        assert "Traceback" not in error_message
        assert "File " not in error_message
        assert "jsonschema" not in error_message.lower()
        # Must be a clean user-facing message
        assert len(error_message) < 300  # bounded length

    # ------------------------------------------------------------------
    # R2: Error message names the failing field
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_validation_error_message_names_field(
        self, handler: HandlerMCP
    ) -> None:
        """Validation error message must name the field that failed validation."""
        tool = _make_tool_with_schema(
            name="field_naming_tool",
            input_schema=_SCHEMA_WITH_REQUIRED_STRING,
        )
        handler._tool_registry["field_naming_tool"] = tool  # type: ignore[assignment]

        envelope = {
            "operation": EnumMcpOperationType.CALL_TOOL.value,
            "payload": {
                "tool_name": "field_naming_tool",
                "arguments": {"model_name": 99},  # wrong type
            },
            "correlation_id": str(uuid4()),
        }

        result = await handler.execute(envelope)

        error_message: str = result.result["payload"]["error_message"]
        assert "model_name" in error_message
