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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_core.types.type_json import StrictJsonPrimitive
from omnibase_infra.enums import EnumSessionLifecycleState

RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
"""Compiled regex for validating run IDs (alphanumeric, dot, hyphen, underscore)."""


def validate_run_id(value: str) -> str:
    """Validate a run_id against the filesystem-safe allowlist.

    Args:
        value: The run_id to validate.

    Returns:
        The validated run_id (unchanged).

    Raises:
        ValueError: If the run_id contains unsafe characters or '..'.
    """
    if not RUN_ID_PATTERN.match(value):
        msg = (
            "run_id must contain only alphanumeric characters, "
            "dots, hyphens, and underscores"
        )
        raise ValueError(msg)
    if ".." in value:
        msg = "run_id must not contain '..'"
        raise ValueError(msg)
    return value


class ModelRunContext(BaseModel):
    """Persistent run context stored at ``~/.claude/state/runs/{run_id}.json``.

    Attributes:
        run_id: Unique identifier for this pipeline run.
        status: Current lifecycle state of the run.
        created_at: When the run was created (UTC).
        updated_at: Last modification timestamp (UTC).
        metadata: Arbitrary key-value metadata. **Warning**: Although this model
            is frozen (field reassignment is prevented), the dict container itself
            is mutable. Callers MUST NOT mutate the dict in-place; use
            ``with_metadata(key, value)`` to produce a new instance instead.
            Values are restricted to JSON-serializable primitives to ensure
            lossless round-trip through ``json.dump``/``json.loads``.
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
    metadata: dict[str, StrictJsonPrimitive] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value data attached to the run. "
            "Values are restricted to JSON-serializable primitives."
        ),
    )

    @field_validator("run_id")
    @classmethod
    def _validate_run_id_safe(cls, v: str) -> str:
        """Reject run_id values with unsafe filesystem characters.

        Uses an allowlist (alphanumeric, dot, hyphen, underscore) rather than
        a denylist, to guard against unexpected special characters.
        """
        return validate_run_id(v)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _validate_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            msg = "Timestamps must be timezone-aware"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _freeze_metadata(self) -> ModelRunContext:
        """Create a defensive copy of the metadata dict container.

        This copies the ``dict`` object so that callers who hold a reference
        to the original dict cannot mutate this model's metadata through that
        reference.  A shallow copy is sufficient here because
        ``StrictJsonPrimitive`` restricts values to immutable JSON primitives
        (``str``, ``int``, ``float``, ``bool``, ``None``), so only the dict
        container itself needs to be duplicated.
        """
        object.__setattr__(self, "metadata", dict(self.metadata))
        return self

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

    def with_metadata(self, key: str, value: StrictJsonPrimitive) -> ModelRunContext:
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

        Compares ``updated_at`` against the current UTC wall-clock, so the
        result is sensitive to system clock skew between the process that
        wrote the document and the process running GC.

        Args:
            ttl_seconds: Time-to-live in seconds (default: 14400 = 4 hours).

        Returns:
            True if the run is older than the TTL.
        """
        age = (datetime.now(UTC) - self.updated_at).total_seconds()
        return age >= ttl_seconds


__all__: list[str] = ["ModelRunContext", "RUN_ID_PATTERN", "validate_run_id"]
