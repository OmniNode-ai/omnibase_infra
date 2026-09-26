# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Opt-in configuration for the signed execution-graph command ingress."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedExecutionGraphGatewayPolicy,
)


class ModelExecutionGraphReadIngressConfig(BaseModel):
    """The contract-derived command topic and its admitted gateway signers."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    command_topic: str = Field(min_length=1)
    gateway_policy: TrustedExecutionGraphGatewayPolicy


__all__ = ["ModelExecutionGraphReadIngressConfig"]
