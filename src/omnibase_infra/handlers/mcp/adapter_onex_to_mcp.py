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

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.adapters.adapter_onex_tool_execution import AdapterONEXToolExecution
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.models.mcp.model_mcp_tool_definition import ModelMCPToolDefinition

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

# Internal ONEX envelope/protocol fields that must never be forwarded to MCP
# clients.  These appear on the top-level dict when the orchestrator returns
# an envelope-shaped result instead of a bare domain value.
#
# SHALLOW STRIPPING ONLY: only top-level keys are removed.  Nested dicts
# inside "payload" or "metadata" are passed through untouched.  This is
# intentional — orchestrator results are expected to be flat domain values;
# if envelope fields appear at a deeper nesting level that signals a protocol
# violation that should be fixed at the source, not silently scrubbed here.
# Note: "payload" is intentionally in this set even though it is a common
# English word.  Tools that return a top-level "payload" key as domain data
# (rather than as an ONEX envelope wrapper) will have it stripped.  Tools
# with this naming pattern should be updated to use a more specific key.
_PROTOCOL_FIELDS: frozenset[str] = frozenset(
    {
        "envelope_id",
        "correlation_id",
        "source",
        "payload",
        "metadata",
        "success",
    }
)


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
        node_executor: AdapterONEXToolExecution | None = None,
        container: ModelONEXContainer | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            node_executor: Optional AdapterONEXToolExecution for real dispatch.
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
        """Invoke an MCP tool by routing to the ONEX orchestrator via AdapterONEXToolExecution.

        Dispatches the tool call through the full ONEX execution pipeline:
        envelope building, correlation ID threading, per-tool timeout enforcement,
        and circuit breaker protection. The raw response is mapped to the MCP
        CallToolResult format: ``{"content": [...], "isError": bool}``.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Tool arguments.
            correlation_id: Optional correlation ID for tracing; generated if absent.

        Returns:
            MCP CallToolResult dict with ``content`` list and ``isError`` flag.

        Raises:
            InfraUnavailableError: If tool not found in registry.
            ProtocolConfigurationError: If node executor not configured.
        """
        correlation_id = correlation_id or uuid4()

        if tool_name not in self._tool_cache:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.MCP,
                operation="invoke_tool",
                target_name=tool_name,
            )
            raise InfraUnavailableError(
                f"Tool '{tool_name}' not found in registry", context=ctx
            )

        if self._node_executor is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.MCP,
                operation="invoke_tool",
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

        tool_def = self._tool_cache[tool_name]

        # Bridge MCPToolDefinition (dataclass) → ModelMCPToolDefinition (Pydantic)
        # for AdapterONEXToolExecution.execute().
        mcp_tool = ModelMCPToolDefinition(
            name=tool_def.name,
            description=tool_def.description,
            version=tool_def.version,
            endpoint=tool_def.execution_endpoint or None,
            timeout_seconds=tool_def.timeout_seconds,
            metadata=dict(tool_def.metadata),
        )

        # Dispatch via AdapterONEXToolExecution: envelope build, timeout,
        # circuit breaker, and HTTP dispatch are all handled there.
        # AdapterONEXToolExecution.execute() normally catches its own errors and
        # returns {"success": False, "error": ...} dicts, but we also guard
        # against any exceptions that escape (e.g. InfraTimeoutError raised
        # before the circuit breaker catches it).
        try:
            raw = await self._node_executor.execute(
                tool=mcp_tool,
                arguments=arguments,
                correlation_id=correlation_id,
            )
        except InfraTimeoutError as exc:
            return {
                "content": [
                    {"type": "text", "text": f"Tool execution timed out: {exc}"}
                ],
                "isError": True,
            }
        except InfraUnavailableError as exc:
            return {
                "content": [{"type": "text", "text": f"Service unavailable: {exc}"}],
                "isError": True,
            }
        # InfraConnectionError is intentionally absent here: AdapterONEXToolExecution.execute()
        # converts InfraConnectionError to a {success: False, error: ...} dict internally before
        # returning, so it never propagates to invoke_tool. If execute() is refactored to let
        # InfraConnectionError propagate, add it here.

        # Map AdapterONEXToolExecution result → CallToolResult dict.
        # MCP spec: {"content": [{"type": "text", "text": ...}], "isError": bool}
        # Use raw.get("result", "") as fallback (not raw itself) to avoid
        # leaking internal protocol fields into MCP content.
        success: bool = bool(raw.get("success", False))
        if success:
            result_payload: object = raw.get("result", "")
            # Strip internal protocol fields so envelope-shaped orchestrator
            # responses do not leak envelope_id, correlation_id, source,
            # payload, metadata, or success into MCP content.
            if isinstance(result_payload, dict):
                result_payload = {
                    k: v for k, v in result_payload.items() if k not in _PROTOCOL_FIELDS
                }
            content_text = (
                result_payload
                if isinstance(result_payload, str)
                else json.dumps(result_payload, default=str)
            )
            return {
                "content": [{"type": "text", "text": content_text}],
                "isError": False,
            }
        else:
            error_text = str(raw.get("error", "Tool execution failed"))
            return {
                "content": [{"type": "text", "text": error_text}],
                "isError": True,
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
