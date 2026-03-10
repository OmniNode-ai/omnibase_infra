# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Output model for the setup validate effect node.

Ticket: OMN-3491
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_setup_validate_effect.models.model_service_health_result import (
    ModelSetupNodeHealthResult,
)


class ModelSetupValidateEffectOutput(BaseModel):
    """Output from the setup validate effect node.

    Aggregates health check results for all validated deployment nodes. The
    ``all_healthy`` field reflects the AND of all individual health results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    all_healthy: bool = Field(
        ...,
        description="True if all nodes passed their health checks.",
    )
    results: tuple[ModelSetupNodeHealthResult, ...] = Field(
        default=(),
        description="Tuple of per-node health check results.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID echoed from the input.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if validation could not complete.",
    )


__all__: list[str] = ["ModelSetupValidateEffectOutput"]
