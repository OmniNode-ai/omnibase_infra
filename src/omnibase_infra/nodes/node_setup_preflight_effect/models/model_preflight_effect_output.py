# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Output model for the setup preflight effect node.

Ticket: OMN-3491
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_setup_preflight_effect.models.model_preflight_check_result import (
    ModelPreflightCheckResult,
)


class ModelPreflightEffectOutput(BaseModel):
    """Output from the setup preflight effect node.

    Aggregates results of all individual preflight checks. The ``passed``
    field reflects the AND of all individual check results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    passed: bool = Field(
        ...,
        description="True if all preflight checks passed.",
    )
    checks: tuple[ModelPreflightCheckResult, ...] = Field(
        default=(),
        description="Tuple of individual check results.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID echoed from the input.",
    )
    duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Total time spent running all preflight checks, in milliseconds.",
    )


__all__: list[str] = ["ModelPreflightEffectOutput"]
