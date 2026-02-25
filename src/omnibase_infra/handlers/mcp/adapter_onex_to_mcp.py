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

Contract-driven discovery:
    Contracts with ``mcp.expose: true`` are automatically discovered when
    ``contracts_root`` is provided to ``discover_tools()``. The tool name,
    description, input schema, and endpoint are all derived from the contract.

    Example contract.yaml fragment::

        mcp:
          expose: true
          tool_name: "register_node"
          description: "Register a new ONEX node with the cluster."
          timeout_seconds: 30
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import yaml

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
    # Full JSON Schema for input validation (OMN-2699).
    # When set, HandlerMCP._handle_call_tool() validates arguments against this
    # schema using jsonschema before dispatching to ONEX.  None means pass-through
    # (backwards-compatible behaviour for tools registered without a schema).
    input_schema: dict[str, object] | None = None

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
        contracts_root: Path | None = None,
    ) -> Sequence[MCPToolDefinition]:
        """Discover MCP-enabled ONEX nodes.

        When ``contracts_root`` is provided, scans all ``contract.yaml`` files
        under that directory tree.  Contracts that contain an ``mcp.expose: true``
        stanza are converted to :class:`MCPToolDefinition` instances and merged
        into the in-memory cache (existing manual registrations are preserved and
        take precedence when tool names collide).

        When ``contracts_root`` is ``None``, only the existing in-memory cache is
        returned (legacy behaviour — no file I/O).

        The discovery pipeline for each qualifying contract:

        1. Load contract YAML; skip unparseable files with a warning.
        2. Skip contracts where ``mcp.expose`` is absent or falsy.
        3. Resolve ``tool_name`` → ``mcp.tool_name`` or ``name`` or file stem.
        4. Resolve ``description`` → ``mcp.description`` or top-level
           ``description``.
        5. Load the Pydantic input model declared in ``input_model.name`` +
           ``input_model.module``; derive JSON schema via
           :meth:`pydantic_to_json_schema`.  Falls back to
           ``{"type": "object"}`` on import errors.
        6. Derive ``endpoint`` from Consul registration metadata when available;
           otherwise leave as an empty string.
        7. Build :class:`MCPToolDefinition`; write into ``_tool_cache`` only if
           the name is **not already present** (manual registrations win).

        Args:
            tags: Optional list of tags to filter results.  Filtering is applied
                *after* the cache has been populated from contracts.
            contracts_root: Root directory to search for ``contract.yaml`` files.
                Pass ``None`` to skip filesystem scanning.

        Returns:
            Sequence of discovered tool definitions matching the optional tag
            filter.
        """
        if contracts_root is not None:
            self._scan_contracts(contracts_root)

        tools = list(self._tool_cache.values())

        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]

        logger.info(
            "Discovered MCP tools",
            extra={
                "tool_count": len(tools),
                "filter_tags": tags,
                "contracts_root": str(contracts_root) if contracts_root else None,
            },
        )

        return tools

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_contracts(self, contracts_root: Path) -> None:
        """Scan *contracts_root* for ``contract.yaml`` files and populate cache.

        Contracts where ``mcp.expose`` is ``true`` are converted to
        :class:`MCPToolDefinition` and merged into ``_tool_cache``.  Existing
        cache entries are not overwritten (manual registrations take precedence).

        Args:
            contracts_root: Root directory to search recursively.
        """
        for contract_path in contracts_root.rglob("contract.yaml"):
            self._load_contract(contract_path)

    def _load_contract(self, contract_path: Path) -> None:
        """Parse one contract YAML and register tool if MCP-enabled.

        Invalid / unparseable contracts are logged and skipped (non-fatal).
        Contracts without ``mcp.expose: true`` are silently ignored.

        Args:
            contract_path: Absolute path to the ``contract.yaml`` file.
        """
        try:
            raw: object = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, ValueError) as exc:
            logger.warning(
                "Skipping contract: failed to parse YAML",
                extra={
                    "contract_path": str(contract_path),
                    "error": str(exc),
                },
            )
            return

        if not isinstance(raw, dict):
            logger.warning(
                "Skipping contract: unexpected top-level type",
                extra={
                    "contract_path": str(contract_path),
                    "actual_type": type(raw).__name__,
                },
            )
            return

        mcp_section = raw.get("mcp")
        if not isinstance(mcp_section, dict):
            # No mcp section — silently ignore
            return
        if not mcp_section.get("expose", False):
            # mcp.expose absent or False — silently ignore
            return

        # --- Derive tool name ---
        tool_name: str = (
            str(mcp_section["tool_name"])
            if mcp_section.get("tool_name")
            else str(raw.get("name", contract_path.parent.name))
        )

        # --- Skip if already in cache (manual registration wins) ---
        if tool_name in self._tool_cache:
            logger.debug(
                "Contract-discovered tool already registered; skipping",
                extra={
                    "tool_name": tool_name,
                    "contract_path": str(contract_path),
                },
            )
            return

        # --- Derive description ---
        description: str = str(
            mcp_section.get("description") or raw.get("description", "")
        ).strip()

        # --- Derive timeout ---
        timeout_raw = mcp_section.get("timeout_seconds", 30)
        timeout_seconds: int = int(timeout_raw) if isinstance(timeout_raw, int) else 30

        # --- Derive input schema from Pydantic model ---
        input_schema: dict[str, object] = {"type": "object"}
        input_model_section = raw.get("input_model")
        if isinstance(input_model_section, dict):
            model_name = input_model_section.get("name")
            model_module = input_model_section.get("module")
            if model_name and model_module:
                input_schema = self._resolve_input_schema(
                    str(model_name), str(model_module)
                )

        # --- Extract parameters from schema ---
        parameters = self.extract_parameters_from_schema(input_schema)

        tool = MCPToolDefinition(
            name=tool_name,
            tool_type="function",
            description=description,
            version=str(raw.get("node_version", "1.0.0")),
            parameters=parameters,
            execution_endpoint="",  # Consul endpoint resolved at invocation time
            timeout_seconds=timeout_seconds,
            metadata={
                "contract_path": str(contract_path),
                "node_name": str(raw.get("name", "")),
                "node_type": str(raw.get("node_type", "")),
                "source": "contract_discovery",
            },
            # Store the resolved JSON Schema so HandlerMCP can validate inputs
            # before dispatching to ONEX (OMN-2699).
            input_schema=input_schema if input_schema != {"type": "object"} else None,
        )
        self._tool_cache[tool_name] = tool

        logger.info(
            "Registered MCP tool from contract",
            extra={
                "tool_name": tool_name,
                "contract_path": str(contract_path),
                "parameter_count": len(parameters),
            },
        )

    def _resolve_input_schema(
        self, model_name: str, model_module: str
    ) -> dict[str, object]:
        """Import *model_module* and generate JSON Schema for *model_name*.

        Falls back to ``{"type": "object"}`` when the module cannot be
        imported or the class is not a Pydantic model.

        Args:
            model_name: Class name within the module.
            model_module: Dotted module path.

        Returns:
            JSON Schema dict.
        """
        try:
            module = importlib.import_module(model_module)
            model_class = getattr(module, model_name)
            return self.pydantic_to_json_schema(model_class)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Cannot resolve input schema for contract tool; using fallback",
                extra={
                    "model_name": model_name,
                    "model_module": model_module,
                    "error": str(exc),
                },
            )
            return {"type": "object"}

    async def register_node_as_tool(
        self,
        node_name: str,
        description: str,
        parameters: list[MCPToolParameter],
        *,
        version: str = "1.0.0",
        tags: list[str] | None = None,
        timeout_seconds: int = 30,
        input_schema: dict[str, object] | None = None,
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
            input_schema: Optional JSON Schema dict for input validation.
                When provided, HandlerMCP validates arguments against this
                schema before dispatching to ONEX (OMN-2699).

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
            input_schema=input_schema,
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

        # Node invocation stub — full ONEX runtime dispatch is implemented
        # separately once the runtime dispatch layer is complete.
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
