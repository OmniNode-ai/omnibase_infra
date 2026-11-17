# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Output model for the setup local provision effect node.

Ticket: OMN-3491
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLocalProvisionEffectOutput(BaseModel):
    """Output from the setup local provision effect node."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the provisioning operation completed successfully.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID echoed from the input.",
    )
    services_started: tuple[str, ...] = Field(
        default=(),
        description="Names of services successfully started (provision_local only).",
    )
    services_stopped: tuple[str, ...] = Field(
        default=(),
        description="Names of services successfully stopped (teardown_local only).",
    )
    services_running: tuple[str, ...] = Field(
        default=(),
        description="Names of services currently running (status_check only).",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the operation failed.",
    )


__all__: list[str] = ["ModelLocalProvisionEffectOutput"]
