# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Auth gate request model — input to the authorization decision cascade.

Carries the tool invocation context (which tool, which path, which repo)
together with the current authorization state (run context, authorization
contract, emergency override flags).

Ticket: OMN-2125
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_auth_gate_compute.models.model_contract_work_authorization import (
    ModelContractWorkAuthorization,
)

# Same control-character pattern used in HandlerAuthGate for
# emergency_override_reason sanitization.  Applied at the model
# boundary so downstream reason strings never embed raw control chars.
_CONTROL_CHAR_RE: re.Pattern[str] = re.compile(
    r"[\x00-\x1f\x7f\u200b-\u200f\u2028-\u202f\u2060-\u2069\ufeff]"
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
        ...,
        min_length=1,
        max_length=200,
        description="Canonical tool name (case-sensitive, e.g., 'Edit', 'Write', 'Bash').",
    )  # ONEX_EXCLUDE: pattern - simple tool identifier
    target_path: str = Field(
        default="",
        max_length=8192,
        description="File path the tool targets.",
    )
    target_repo: str = Field(
        default="",
        max_length=500,
        description="Repository the tool targets.",
    )
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
        max_length=1000,
        description="Value of ONEX_UNSAFE_REASON env var.",
    )
    now: AwareDatetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Current UTC timestamp for expiry checks.",
    )

    @field_validator(
        "tool_name",
        "target_path",
        "target_repo",
        "emergency_override_reason",
        mode="before",
    )
    @classmethod
    def _strip_control_characters(cls, v: str) -> str:
        """Strip control characters from string fields for defense-in-depth."""
        if not isinstance(v, str):
            return v
        return _CONTROL_CHAR_RE.sub("", v)


__all__: list[str] = ["ModelAuthGateRequest"]
