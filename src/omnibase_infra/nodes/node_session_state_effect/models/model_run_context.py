# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Run context model — maps to ``~/.claude/state/runs/{run_id}.json``.

Each pipeline instance gets its own run context document. These documents
are single-writer (the pipeline that owns the run_id), so no file locking
is required for run context files.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums import EnumSessionLifecycleState

_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")


class ModelRunContext(BaseModel):
    """Persistent run context stored at ``~/.claude/state/runs/{run_id}.json``.

    Attributes:
        run_id: Unique identifier for this pipeline run.
        status: Current lifecycle state of the run.
        created_at: When the run was created (UTC).
        updated_at: Last modification timestamp (UTC).
        metadata: Arbitrary key-value data attached to the run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    run_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this pipeline run.",
    )
    status: EnumSessionLifecycleState = Field(
        default=EnumSessionLifecycleState.RUN_CREATED,
        description="Current lifecycle state of the run.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the run was created (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last modification timestamp (UTC).",
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value data attached to the run. "
            "Warning: ``frozen=True`` prevents reassigning this field but does "
            "not prevent in-place mutation of dict contents. Always use "
            "``with_metadata()`` to add entries immutably."
        ),
    )

    @field_validator("run_id")
    @classmethod
    def _validate_run_id_safe(cls, v: str) -> str:
        """Reject run_id values with unsafe filesystem characters.

        Uses an allowlist (alphanumeric, dot, hyphen, underscore) rather than
        a denylist, to guard against unexpected special characters.
        """
        if not _RUN_ID_PATTERN.match(v):
            msg = (
                "run_id must contain only alphanumeric characters, "
                "dots, hyphens, and underscores"
            )
            raise ValueError(msg)
        # Double-dot still disallowed (path traversal)
        if ".." in v:
            msg = "run_id must not contain '..'"
            raise ValueError(msg)
        return v

    @field_validator("created_at", "updated_at")
    @classmethod
    def _validate_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            msg = "Timestamps must be timezone-aware"
            raise ValueError(msg)
        return v

    # ------------------------------------------------------------------
    # Transition helpers (pure — return new instances)
    # ------------------------------------------------------------------

    def with_status(self, status: EnumSessionLifecycleState) -> ModelRunContext:
        """Return a new run context with an updated status.

        Args:
            status: The new lifecycle state.

        Returns:
            New ModelRunContext with updated status and timestamp.
        """
        return ModelRunContext(
            run_id=self.run_id,
            status=status,
            created_at=self.created_at,
            updated_at=datetime.now(UTC),
            metadata={**self.metadata},
        )

    def with_metadata(self, key: str, value: object) -> ModelRunContext:
        """Return a new run context with an additional metadata entry.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            New ModelRunContext with the metadata entry added.
        """
        new_meta = {**self.metadata, key: value}
        return ModelRunContext(
            run_id=self.run_id,
            status=self.status,
            created_at=self.created_at,
            updated_at=datetime.now(UTC),
            metadata=new_meta,
        )

    def is_stale(self, ttl_seconds: float = 14400.0) -> bool:
        """Check if this run context is stale (default 4hr TTL).

        Args:
            ttl_seconds: Time-to-live in seconds (default: 14400 = 4 hours).

        Returns:
            True if the run is older than the TTL.
        """
        age = (datetime.now(UTC) - self.updated_at).total_seconds()
        return age > ttl_seconds


__all__: list[str] = ["ModelRunContext"]
