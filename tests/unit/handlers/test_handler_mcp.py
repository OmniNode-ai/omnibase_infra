# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for HandlerMCP.

Comprehensive test suite covering initialization, MCP operations,
tool registration, and lifecycle management.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

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
from omnibase_infra.handlers.models.mcp import (
    EnumMcpOperationType,
    ModelMcpHandlerConfig,
)


class TestHandlerMCPInitialization:
    """Test suite for HandlerMCP initialization."""

    @pytest.fixture
    def handler(self) -> HandlerMCP:
        """Create HandlerMCP fixture."""
        return HandlerMCP()

    def test_handler_init_default_state(self, handler: HandlerMCP) -> None:
        """Test handler initializes in uninitialized state."""
        assert handler._initialized is False
        assert handler._config is None
        assert handler._tool_registry == {}

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
    async def test_initialize_with_empty_config(self, handler: HandlerMCP) -> None:
        """Test handler initializes with empty config (uses defaults)."""
        await handler.initialize({})

        assert handler._initialized is True
        assert handler._config is not None
        assert handler._config.host == "0.0.0.0"  # noqa: S104
        assert handler._config.port == 8090
        assert handler._config.path == "/mcp"
        assert handler._config.stateless is True
        assert handler._config.json_response is True

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_config(self, handler: HandlerMCP) -> None:
        """Test handler initializes with custom configuration."""
        config: dict[str, object] = {
            "host": "127.0.0.1",
            "port": 9000,
            "path": "/api/mcp",
            "timeout_seconds": 60.0,
            "max_tools": 50,
        }
        await handler.initialize(config)

        assert handler._initialized is True
        assert handler._config is not None
        assert handler._config.host == "127.0.0.1"
        assert handler._config.port == 9000
        assert handler._config.path == "/api/mcp"
        assert handler._config.timeout_seconds == 60.0
        assert handler._config.max_tools == 50

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_clears_state(self, handler: HandlerMCP) -> None:
        """Test shutdown clears handler state."""
        await handler.initialize({})
        assert handler._initialized is True

        await handler.shutdown()

        assert handler._initialized is False
        assert handler._config is None
        assert handler._tool_registry == {}


class TestHandlerMCPDescribe:
    """Test suite for describe operation."""

    @pytest.fixture
    async def initialized_handler(self) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture."""
        handler = HandlerMCP()
        await handler.initialize({})
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
    async def initialized_handler(self) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture."""
        handler = HandlerMCP()
        await handler.initialize({})
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
    async def test_execute_without_initialization_raises(self) -> None:
        """Test execute raises RuntimeHostError if not initialized."""
        handler = HandlerMCP()
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
    async def initialized_handler(self) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture."""
        handler = HandlerMCP()
        await handler.initialize({})
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
    async def initialized_handler(self) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture."""
        handler = HandlerMCP()
        await handler.initialize({})
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
    async def initialized_handler(self) -> HandlerMCP:
        """Create and initialize a HandlerMCP fixture."""
        handler = HandlerMCP()
        await handler.initialize({})
        yield handler
        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_initialized(
        self, initialized_handler: HandlerMCP
    ) -> None:
        """Test health check returns healthy when initialized."""
        health = await initialized_handler.health_check()

        assert health["healthy"] is True
        assert health["initialized"] is True
        assert health["tool_count"] == 0
        assert health["transport_type"] == "mcp"

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self) -> None:
        """Test health check returns unhealthy when not initialized."""
        handler = HandlerMCP()
        health = await handler.health_check()

        assert health["healthy"] is False
        assert health["initialized"] is False


class TestMcpHandlerConfig:
    """Test suite for ModelMcpHandlerConfig."""

    def test_config_defaults(self) -> None:
        """Test config has correct defaults."""
        config = ModelMcpHandlerConfig()

        assert config.host == "0.0.0.0"  # noqa: S104
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

        with pytest.raises(Exception):  # Pydantic raises ValidationError on frozen
            config.host = "changed"
