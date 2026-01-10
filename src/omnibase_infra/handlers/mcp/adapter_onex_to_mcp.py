# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX to MCP Adapter - Convert ONEX contracts to MCP tool definitions.

This adapter bridges the ONEX node ecosystem with the MCP (Model Context Protocol)
tool interface, enabling AI agents to discover and invoke ONEX nodes as tools.

The adapter:
1. Scans the ONEX registry for MCP-enabled nodes
2. Converts ONEX contracts to MCP tool definitions
3. Generates JSON schemas from Pydantic input models
4. Routes MCP tool calls to ONEX node execution

Example:
    adapter = ONEXToMCPAdapter(node_registry)
    tools = await adapter.discover_tools()
    result = await adapter.invoke_tool("node_name", {"param": "value"})

Note:
    This adapter is designed for future integration with the ONEX node registry.
    Currently, tool discovery is manual via `register_node_as_tool()`. Once the
    ONEX registry is fully implemented (OMN-1288), this adapter will automatically
    scan the registry for nodes that expose MCP capabilities through their
    contract.yaml `mcp_enabled: true` flag, enabling zero-configuration tool
    discovery for AI agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)


@dataclass
class MCPToolParameter:
    """MCP tool parameter definition.

    Represents a single parameter for an MCP tool, including its type,
    description, and validation constraints.
    """

    name: str
    parameter_type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default_value: object | None = None
    schema: dict[str, object] | None = None
    constraints: dict[str, object] = field(default_factory=dict)
    examples: list[object] = field(default_factory=list)

    def validate_parameter(self) -> bool:
        """Validate the parameter definition."""
        return bool(self.name and self.parameter_type)

    def is_required_parameter(self) -> bool:
        """Check if this parameter is required."""
        return self.required


@dataclass
class MCPToolDefinition:
    """MCP tool definition.

    Represents a complete MCP tool specification including its parameters,
    return schema, and execution metadata.
    """

    name: str
    tool_type: str  # "function", "resource", "prompt", "sampling", "completion"
    description: str
    version: str
    parameters: list[MCPToolParameter] = field(default_factory=list)
    return_schema: dict[str, object] | None = None
    execution_endpoint: str = ""
    timeout_seconds: int = 30
    retry_count: int = 3
    requires_auth: bool = False
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def validate_tool_definition(self) -> bool:
        """Validate the tool definition."""
        return bool(self.name and self.description)


class ONEXToMCPAdapter:
    """Adapter for converting ONEX contracts to MCP tool definitions.

    This adapter provides the bridge between ONEX nodes and MCP tools,
    enabling AI agents to discover and invoke ONEX functionality.

    The adapter supports:
    - Dynamic tool discovery from node registry
    - Contract-to-schema conversion
    - Parameter mapping between ONEX and MCP formats
    - Tool invocation routing to ONEX nodes
    - Container-based dependency injection for ONEX integration

    Attributes:
        _tool_cache: Cache of discovered tool definitions.
        _node_executor: Callback for executing ONEX nodes.
        _container: Optional ONEX container for dependency injection.
    """

    def __init__(
        self,
        node_executor: object | None = None,
        container: ModelONEXContainer | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            node_executor: Optional callback for node execution.
                          If not provided, tools will be discovered but
                          not executable.
            container: Optional ONEX container for dependency injection.
                      Provides access to shared services and configuration
                      when integrating with the ONEX runtime.
        """
        self._tool_cache: dict[str, MCPToolDefinition] = {}
        self._node_executor = node_executor
        self._container = container

    async def discover_tools(
        self,
        tags: list[str] | None = None,
    ) -> Sequence[MCPToolDefinition]:
        """Discover MCP-enabled ONEX nodes.

        Scans the node registry for nodes that expose MCP tool capabilities
        and converts their contracts to MCP tool definitions.

        Args:
            tags: Optional list of tags to filter by.

        Returns:
            Sequence of discovered tool definitions.
        """
        # TODO(OMN-1288): Implement actual registry scanning
        # For now, return cached tools
        tools = list(self._tool_cache.values())

        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]

        logger.info(
            "Discovered MCP tools",
            extra={
                "tool_count": len(tools),
                "filter_tags": tags,
            },
        )

        return tools

    async def register_node_as_tool(
        self,
        node_name: str,
        description: str,
        parameters: list[MCPToolParameter],
        *,
        version: str = "1.0.0",
        tags: list[str] | None = None,
        timeout_seconds: int = 30,
    ) -> MCPToolDefinition:
        """Register an ONEX node as an MCP tool.

        Creates an MCP tool definition from the provided node metadata
        and adds it to the tool cache.

        Args:
            node_name: Name of the ONEX node.
            description: Human-readable description for AI agents.
            parameters: List of parameter definitions.
            version: Tool version (default: "1.0.0").
            tags: Optional categorization tags.
            timeout_seconds: Execution timeout.

        Returns:
            The created tool definition.
        """
        tool = MCPToolDefinition(
            name=node_name,
            tool_type="function",
            description=description,
            version=version,
            parameters=parameters,
            timeout_seconds=timeout_seconds,
            tags=tags or [],
        )

        self._tool_cache[node_name] = tool

        logger.info(
            "Registered node as MCP tool",
            extra={
                "node_name": node_name,
                "parameter_count": len(parameters),
                "tags": tags,
            },
        )

        return tool

    async def invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, object],
        correlation_id: UUID | None = None,
    ) -> dict[str, object]:
        """Invoke an MCP tool by routing to the corresponding ONEX node.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Tool arguments.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Tool execution result.

        Raises:
            InfraUnavailableError: If tool not found.
            ProtocolConfigurationError: If node executor not configured.
        """
        correlation_id = correlation_id or uuid4()

        if tool_name not in self._tool_cache:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.MCP,
                operation="invoke_tool",
                target_name=tool_name,
                correlation_id=correlation_id,
            )
            raise InfraUnavailableError(
                f"Tool '{tool_name}' not found in registry", context=ctx
            )

        if self._node_executor is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.MCP,
                operation="invoke_tool",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Node executor not configured. Cannot invoke tools without executor.",
                context=ctx,
            )

        logger.info(
            "Invoking MCP tool",
            extra={
                "tool_name": tool_name,
                "argument_count": len(arguments),
                "correlation_id": str(correlation_id),
            },
        )

        # TODO(OMN-1288): Implement actual node invocation
        # This will:
        # 1. Build an envelope for the ONEX node
        # 2. Dispatch via the ONEX runtime
        # 3. Transform the response to MCP format

        return {
            "success": True,
            "message": f"Tool '{tool_name}' invoked successfully",
            "arguments": arguments,
            "correlation_id": str(correlation_id),
        }

    def get_tool(self, tool_name: str) -> MCPToolDefinition | None:
        """Get a tool definition by name.

        Args:
            tool_name: Name of the tool.

        Returns:
            Tool definition if found, None otherwise.
        """
        return self._tool_cache.get(tool_name)

    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool.

        Args:
            tool_name: Name of the tool to unregister.

        Returns:
            True if tool was unregistered, False if not found.
        """
        if tool_name in self._tool_cache:
            del self._tool_cache[tool_name]
            logger.info("Unregistered MCP tool", extra={"tool_name": tool_name})
            return True
        return False

    @staticmethod
    def pydantic_to_json_schema(
        model_class: type,
        *,
        raise_on_error: bool = False,
    ) -> dict[str, object]:
        """Convert a Pydantic model to JSON Schema.

        This is useful for generating MCP input schemas from ONEX
        node input models.

        Args:
            model_class: Pydantic model class.
            raise_on_error: If True, raise ProtocolConfigurationError on failure
                instead of returning a fallback schema. Default is False for
                backwards compatibility.

        Returns:
            JSON Schema dict.

        Raises:
            ProtocolConfigurationError: If raise_on_error=True and schema
                generation fails.
        """
        try:
            from pydantic import BaseModel

            if issubclass(model_class, BaseModel):
                return model_class.model_json_schema()

            # model_class is a valid type but not a Pydantic BaseModel subclass
            model_name = getattr(model_class, "__name__", str(model_class))
            logger.warning(
                "Cannot generate Pydantic schema: model_class is not a BaseModel subclass",
                extra={
                    "model_class": model_name,
                    "model_type": type(model_class).__name__,
                    "reason": "not_basemodel_subclass",
                },
            )
            if raise_on_error:
                raise ProtocolConfigurationError(
                    f"Cannot generate schema: {model_name} is not a Pydantic BaseModel subclass",
                )

        except TypeError as e:
            # TypeError occurs when model_class is not a valid class type
            # (e.g., None, primitive, or other non-class object that cannot be
            # checked with issubclass)
            model_repr = getattr(model_class, "__name__", str(model_class))
            logger.warning(
                "Cannot generate Pydantic schema: model_class is not a valid type, "
                "using fallback",
                extra={
                    "model_class": model_repr,
                    "model_type": type(model_class).__name__,
                    "error": str(e),
                    "reason": "not_valid_type",
                },
            )
            if raise_on_error:
                raise ProtocolConfigurationError(
                    f"Cannot generate schema: {model_repr} is not a valid Pydantic model class",
                ) from e

        except ImportError as e:
            # ImportError occurs when pydantic is not installed
            logger.warning(
                "Cannot generate Pydantic schema: pydantic not available, using fallback",
                extra={
                    "model_class": getattr(model_class, "__name__", str(model_class)),
                    "error": str(e),
                    "reason": "pydantic_not_installed",
                },
            )
            if raise_on_error:
                raise ProtocolConfigurationError(
                    "Cannot generate schema: pydantic library is not installed",
                ) from e

        # Fallback for non-Pydantic types or when pydantic unavailable
        return {"type": "object"}

    @staticmethod
    def extract_parameters_from_schema(
        schema: dict[str, object],
    ) -> list[MCPToolParameter]:
        """Extract MCP parameters from a JSON Schema.

        Converts JSON Schema properties to MCPToolParameter instances.

        Args:
            schema: JSON Schema dict.

        Returns:
            List of parameter definitions.
        """
        parameters: list[MCPToolParameter] = []
        properties = schema.get("properties", {})
        required_list = schema.get("required", [])
        required: set[str] = (
            set(required_list) if isinstance(required_list, list) else set()
        )

        if not isinstance(properties, dict):
            return parameters

        for name, prop in properties.items():
            if not isinstance(prop, dict):
                continue

            param_type = prop.get("type", "string")
            if isinstance(param_type, list):
                # Handle union types - use first non-null type
                param_type = next((t for t in param_type if t != "null"), "string")

            param = MCPToolParameter(
                name=name,
                parameter_type=str(param_type),
                description=str(prop.get("description", "")),
                required=name in required,
                default_value=prop.get("default"),
                schema=prop if "enum" in prop or "format" in prop else None,
            )
            parameters.append(param)

        return parameters


__all__ = [
    "MCPToolDefinition",
    "MCPToolParameter",
    "ONEXToMCPAdapter",
]
