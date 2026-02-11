# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Auth gate request model — input to the authorization decision cascade.

Carries the tool invocation context (which tool, which path, which repo)
together with the current authorization state (run context, authorization
contract, emergency override flags).

Ticket: OMN-2125
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_auth_gate_compute.models.model_contract_work_authorization import (
    ModelContractWorkAuthorization,
)


class ModelAuthGateRequest(BaseModel):
    """Input model for the auth gate compute handler.

    Attributes:
        tool_name: Name of the tool being invoked (e.g., "Edit", "Write", "Bash").
        target_path: File path the tool targets. Empty string for non-file tools.
        target_repo: Repository identifier the tool targets. Empty for current repo.
        run_id: Current run ID, or None if not determinable.
        authorization: Active work authorization contract, or None if not granted.
        emergency_override_active: Whether ONEX_UNSAFE_ALLOW_EDITS=1 is set.
        emergency_override_reason: Value of ONEX_UNSAFE_REASON env var, or empty.
        now: Current UTC timestamp for expiry checks. Defaults to now.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    tool_name: str = Field(
        ..., min_length=1, description="Tool being invoked."
    )  # ONEX_EXCLUDE: pattern - simple tool identifier
    target_path: str = Field(default="", description="File path the tool targets.")
    target_repo: str = Field(default="", description="Repository the tool targets.")
    run_id: UUID | None = Field(
        default=None, description="Current run ID, or None if not determinable."
    )
    authorization: ModelContractWorkAuthorization | None = Field(
        default=None, description="Active work authorization, or None."
    )
    emergency_override_active: bool = Field(
        default=False,
        description="Whether ONEX_UNSAFE_ALLOW_EDITS=1 is set.",
    )
    emergency_override_reason: str = Field(
        default="",
        description="Value of ONEX_UNSAFE_REASON env var.",
    )
    now: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Current UTC timestamp for expiry checks.",
    )


__all__: list[str] = ["ModelAuthGateRequest"]
