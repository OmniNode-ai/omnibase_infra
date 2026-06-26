# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registration record for a generated tool routed onto the canonical MCP registry.

This is the canonical replacement for the bespoke SEA ``ToolRegistration`` record
(``onex-self-extending-agent/src/agent/tool_registry.py``). It captures the
evidence-binding fields that prove *which* generated artifact was registered,
without conflating registration with in-process handler execution (the exec/invoke
sandbox is the Phase 0 hot-load executor concern, OMN-13605, not MCP registration).

The MCP server in ``omnibase_infra`` is the protocol-infra owner of tool
registration (feedback_mcp_server_in_omnibase_infra.md); generated tools register
through ``ServiceMCPToolRegistry.register_generated_tool`` and surface via the
canonical ``ServiceMCPToolRegistry`` / ``ServiceMCPToolSync`` exposure path.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelMCPGeneratedToolRegistration(BaseModel):
    """Typed record returned when a generated tool is registered on the MCP registry.

    The artifact-hash fields bind the registration to the exact contract + handler
    source that produced it, so a later audit can prove provenance. The
    ``registry_event_version`` is the version key the registry used for the upsert
    (idempotency / out-of-order resolution).

    Attributes:
        registration_id: Unique identifier for this registration record.
        name: Canonical MCP tool name (the generated node name).
        description: AI-friendly description surfaced to MCP clients.
        generated_contract_hash: ``sha256:<hex>`` of the generating contract YAML.
        generated_handler_hash: ``sha256:<hex>`` of the generating handler source.
        generation_correlation_id: Correlation id of the generation request.
        registry_event_version: Version key used for the registry upsert.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    registration_id: UUID = Field(
        description="Unique identifier for this registration record",
    )
    name: str = Field(
        description="Canonical MCP tool name (the generated node name)",
    )
    description: str = Field(
        description="AI-friendly description surfaced to MCP clients",
    )
    generated_contract_hash: str = Field(
        description="sha256:<hex> of the generating contract YAML",
    )
    generated_handler_hash: str = Field(
        description="sha256:<hex> of the generating handler source",
    )
    generation_correlation_id: UUID = Field(
        description="Correlation id of the generation request",
    )
    registry_event_version: str = Field(
        description="Version key used for the registry upsert (idempotency)",
    )


__all__ = ["ModelMCPGeneratedToolRegistration"]
