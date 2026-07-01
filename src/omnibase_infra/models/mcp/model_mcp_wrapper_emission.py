# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed result of emitting a generated COMPUTE node + its MCP wrapper orchestrator.

``ServiceMCPWrapperContractEmitter.emit`` returns this record (OMN-12841). It
carries BOTH artifacts the node-generation pipeline must persist for a generated
COMPUTE node that is exposed as an MCP tool via Option B (contract-native):

- ``compute_contract_yaml`` -- the generated COMPUTE contract (passed through
  verbatim from the spec).
- ``wrapper_contract_yaml`` -- the thin declarative ORCHESTRATOR wrapper contract
  whose ``mcp.expose=true`` + ``tool_name`` and ``handler_routing`` route the
  inbound MCP invocation to the generated COMPUTE handler. The wrapper is the
  MCP-exposed entity that satisfies the unchanged orchestrator-only MCP gate.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelMCPWrapperEmission(BaseModel):
    """Both contracts emitted for a generated COMPUTE node exposed as an MCP tool.

    Attributes:
        compute_node_name: Name of the generated COMPUTE node.
        wrapper_node_name: Name of the emitted ORCHESTRATOR wrapper node.
        tool_name: MCP tool name carried by the wrapper.
        compute_contract_yaml: The generated COMPUTE contract YAML (verbatim).
        wrapper_contract_yaml: The declarative ORCHESTRATOR wrapper contract YAML.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    compute_node_name: str = Field(
        ...,
        description="Name of the generated COMPUTE node",
    )
    wrapper_node_name: str = Field(
        ...,
        description="Name of the emitted ORCHESTRATOR wrapper node",
    )
    tool_name: str = Field(
        ...,
        description="MCP tool name carried by the wrapper",
    )
    compute_contract_yaml: str = Field(
        ...,
        description="The generated COMPUTE contract YAML (verbatim)",
    )
    wrapper_contract_yaml: str = Field(
        ...,
        description="The declarative ORCHESTRATOR wrapper contract YAML",
    )


__all__ = ["ModelMCPWrapperEmission"]
