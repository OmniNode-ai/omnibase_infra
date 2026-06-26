# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed spec for a generated COMPUTE node that should surface as an MCP tool.

This is the input to ``ServiceMCPWrapperContractEmitter`` (OMN-12841). The
node-generation pipeline produces a COMPUTE node (CONTRACT + handler); when MCP
exposure is requested, it hands this spec to the emitter, which returns the
declarative ORCHESTRATOR wrapper contract that is the MCP-exposed surface.

Why an orchestrator wrapper (Option B, contract-native): the MCP tool boundary
stays orchestrator-only (``service_mcp_tool_discovery.py:197-212`` /
``service_mcp_tool_sync.py``); a COMPUTE node is stateless/deterministic and is
not a workflow surface. Rather than relaxing the gate (Option A, rejected), the
generated COMPUTE node is fronted by a thin declarative ORCHESTRATOR that
satisfies the existing gate and routes the inbound invocation to the COMPUTE
handler.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelGeneratedComputeNodeSpec(BaseModel):
    """Spec describing a generated COMPUTE node to expose as an MCP tool.

    Attributes:
        node_name: Generated COMPUTE node name (snake_case, ``node_`` prefixed).
        description: AI-friendly description surfaced to MCP clients.
        tool_name: Stable MCP tool name (typically the COMPUTE node name).
        compute_contract_yaml: The generated COMPUTE contract YAML (passed
            through verbatim so both artifacts are emitted together).
        compute_handler_module: Import module of the generated COMPUTE handler
            (the wrapper routes the inbound invocation to this handler).
        compute_handler_class: Class name of the generated COMPUTE handler.
        invocation_input_model_module: Module of the invocation input model.
        invocation_input_model_name: Class name of the invocation input model.
        invocation_output_model_module: Module of the invocation output model.
        invocation_output_model_name: Class name of the invocation output model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_name: str = Field(
        ...,
        min_length=1,
        description="Generated COMPUTE node name (node_ prefixed snake_case)",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="AI-friendly description surfaced to MCP clients",
    )
    tool_name: str = Field(
        ...,
        min_length=1,
        description="Stable MCP tool name (typically the COMPUTE node name)",
    )
    compute_contract_yaml: str = Field(
        ...,
        min_length=1,
        description="The generated COMPUTE contract YAML (emitted alongside)",
    )
    compute_handler_module: str = Field(
        ...,
        min_length=1,
        description="Import module of the generated COMPUTE handler",
    )
    compute_handler_class: str = Field(
        ...,
        min_length=1,
        description="Class name of the generated COMPUTE handler",
    )
    invocation_input_model_module: str = Field(
        ...,
        min_length=1,
        description="Module of the invocation input model",
    )
    invocation_input_model_name: str = Field(
        ...,
        min_length=1,
        description="Class name of the invocation input model",
    )
    invocation_output_model_module: str = Field(
        ...,
        min_length=1,
        description="Module of the invocation output model",
    )
    invocation_output_model_name: str = Field(
        ...,
        min_length=1,
        description="Class name of the invocation output model",
    )

    @field_validator(
        "node_name",
        "description",
        "tool_name",
        "compute_contract_yaml",
        "compute_handler_module",
        "compute_handler_class",
        "invocation_input_model_module",
        "invocation_input_model_name",
        "invocation_output_model_module",
        "invocation_output_model_name",
    )
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Fail fast on whitespace-only values (no silent empty defaults)."""
        if not value.strip():
            raise ValueError("must be a non-empty, non-whitespace string")
        return value


__all__ = ["ModelGeneratedComputeNodeSpec"]
