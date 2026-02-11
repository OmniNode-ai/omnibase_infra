# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Work authorization contract model.

Defines the scope under which a run is authorized to invoke tools. Each
authorization is bound to a specific run_id and carries an expiry timestamp,
allowed tools, allowed path globs, and repository scopes.

Ticket: OMN-2125
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_auth_source import EnumAuthSource


class ModelContractWorkAuthorization(BaseModel):
    """Immutable work authorization contract bound to a specific run.

    Attributes:
        run_id: The run this authorization is bound to.
        allowed_tools: Tool names permitted (e.g., ["Edit", "Write"]).
        allowed_paths: Glob patterns for permitted file paths.
        repo_scopes: Repository identifiers in scope for cross-repo support.
        source: How the authorization was granted.
        expires_at: UTC timestamp when authorization expires.
        reason: Optional human-readable reason for the grant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    run_id: UUID = Field(..., description="Run ID this authorization is bound to.")
    allowed_tools: tuple[str, ...] = Field(
        ..., description="Tool names permitted under this authorization."
    )
    allowed_paths: tuple[str, ...] = Field(
        ..., description="Glob patterns for permitted file paths."
    )
    repo_scopes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Repository identifiers in scope.",
    )
    source: EnumAuthSource = Field(
        ..., description="How the authorization was granted."
    )
    expires_at: datetime = Field(
        ..., description="UTC timestamp when this authorization expires."
    )
    reason: str = Field(
        default="",
        description="Human-readable reason for the authorization grant.",
    )

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check whether this authorization has expired.

        Args:
            now: Current UTC time. Defaults to ``datetime.now(timezone.utc)``.

        Returns:
            True if the authorization has expired.
        """
        if now is None:
            now = datetime.now(UTC)
        return now >= self.expires_at


__all__: list[str] = ["ModelContractWorkAuthorization"]
