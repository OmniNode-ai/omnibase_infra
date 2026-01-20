# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP tool definition model for representing ONEX nodes as MCP tools.

This model is used by the MCP adapter layer to:
1. Cache discovered tools in the registry
2. Generate MCP tool schemas for AI agents
3. Route tool invocations to ONEX orchestrators
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from omnibase_infra.models.mcp.model_mcp_tool_parameter import ModelMCPToolParameter


class ModelMCPToolDefinition(BaseModel):
    """Complete MCP tool definition derived from an ONEX orchestrator node.

    This model captures all information needed to:
    - Expose the tool to AI agents via MCP protocol
    - Route invocations to the correct ONEX orchestrator
    - Enforce execution constraints (timeout, etc.)

    Attributes:
        name: Stable tool name for AI agent invocation. This is derived from
            contract.mcp.tool_name or falls back to the node name.
        description: AI-friendly description of tool functionality.
        version: Tool version from the node contract.
        parameters: List of input parameters with type information.
        input_schema: JSON Schema for input validation.
        orchestrator_node_id: UUID of the ONEX orchestrator node.
        orchestrator_service_id: Consul service ID for routing.
        endpoint: HTTP endpoint for direct invocation (if available).
        timeout_seconds: Execution timeout for tool invocations.
        metadata: Additional metadata for routing and observability.
    """

    name: str = Field(description="Stable tool name for MCP invocation")
    description: str = Field(description="AI-friendly tool description")
    version: str = Field(default="1.0.0", description="Tool version")
    parameters: list[ModelMCPToolParameter] = Field(
        default_factory=list, description="Input parameters"
    )
    input_schema: dict[str, object] = Field(
        default_factory=lambda: dict[str, object]({"type": "object", "properties": {}}),
        description="JSON Schema for input validation",
    )
    orchestrator_node_id: str | None = Field(
        default=None, description="UUID of the source orchestrator node"
    )
    orchestrator_service_id: str | None = Field(
        default=None, description="Consul service ID for routing"
    )
    endpoint: str | None = Field(
        default=None, description="HTTP endpoint for direct invocation"
    )
    timeout_seconds: int = Field(
        default=30, ge=1, le=300, description="Execution timeout in seconds"
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Additional metadata (tags, input_model module, etc.)",
    )

    @property
    def tool_type(self) -> str:
        """Return the MCP tool type. Always 'function' for ONEX nodes."""
        return "function"


__all__ = ["ModelMCPToolDefinition"]
