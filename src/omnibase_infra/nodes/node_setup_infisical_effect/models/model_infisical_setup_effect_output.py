# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Output model for the setup Infisical effect node.

Ticket: OMN-3491
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelInfisicalSetupEffectOutput(BaseModel):
    """Output from the setup Infisical effect node."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether Infisical provisioning completed successfully.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID echoed from the input.",
    )
    status: Literal["completed", "skipped", "failed"] = Field(
        ...,
        description=(
            "Outcome: completed (provisioned), skipped (service disabled), "
            "failed (error encountered)."
        ),
    )
    infisical_addr: str | None = Field(
        default=None,
        description="Resolved Infisical address used during provisioning.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if provisioning failed.",
    )


__all__: list[str] = ["ModelInfisicalSetupEffectOutput"]
