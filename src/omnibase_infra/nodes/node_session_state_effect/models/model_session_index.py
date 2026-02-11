# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Session index model — maps to ``~/.claude/state/session.json``.

The session index is the central pointer that tracks which runs exist
and which (if any) is the active interactive run. It is protected by
``flock`` during writes to handle concurrent pipeline access.

Concurrency model:
    - ``recent_run_ids`` is append-only (new runs are prepended).
    - ``active_run_id`` is advisory for interactive sessions.
    - Multiple concurrent active runs are allowed — destructive ops
      are denied until ``/onex:set-active-run {run_id}`` disambiguates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelSessionIndex(BaseModel):
    """Persistent session index stored at ``~/.claude/state/session.json``.

    Attributes:
        active_run_id: Currently selected interactive run (advisory).
        recent_run_ids: Ordered list of known run IDs (most recent first).
        updated_at: Last modification timestamp (timezone-aware UTC).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    active_run_id: str | None = Field(
        default=None,
        description="Currently selected interactive run (advisory).",
    )
    recent_run_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Ordered run IDs, most recent first.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last modification timestamp (UTC).",
    )

    @field_validator("updated_at")
    @classmethod
    def _validate_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            msg = "updated_at must be timezone-aware"
            raise ValueError(msg)
        return v

    # ------------------------------------------------------------------
    # Transition helpers (pure — return new instances)
    # ------------------------------------------------------------------

    #: Maximum number of run IDs retained in the session index.
    #: Older entries are trimmed on each ``with_run_added()`` call.
    MAX_RECENT_RUNS: ClassVar[int] = 1000

    def with_run_added(
        self, run_id: str, *, set_active: bool = False
    ) -> ModelSessionIndex:
        """Return a new index with *run_id* prepended to ``recent_run_ids``.

        The list is capped at :attr:`MAX_RECENT_RUNS` entries to prevent
        unbounded growth of ``session.json`` over long-running sessions.

        Args:
            run_id: The new run identifier to register.
            set_active: If True, also set ``active_run_id`` to this run.

        Returns:
            New ModelSessionIndex with the run added.
        """
        ids = (run_id, *[rid for rid in self.recent_run_ids if rid != run_id])
        ids = ids[: self.MAX_RECENT_RUNS]
        return ModelSessionIndex(
            active_run_id=run_id if set_active else self.active_run_id,
            recent_run_ids=ids,
            updated_at=datetime.now(UTC),
        )

    def with_run_removed(self, run_id: str) -> ModelSessionIndex:
        """Return a new index with *run_id* removed from ``recent_run_ids``.

        If the removed run was ``active_run_id``, the active pointer is cleared.

        Args:
            run_id: The run identifier to remove.

        Returns:
            New ModelSessionIndex with the run removed.
        """
        ids = tuple(rid for rid in self.recent_run_ids if rid != run_id)
        active = None if self.active_run_id == run_id else self.active_run_id
        return ModelSessionIndex(
            active_run_id=active,
            recent_run_ids=ids,
            updated_at=datetime.now(UTC),
        )

    def with_active_run(self, run_id: str) -> ModelSessionIndex:
        """Return a new index with ``active_run_id`` set.

        Args:
            run_id: The run to set as active (must be in recent_run_ids).

        Returns:
            New ModelSessionIndex with active_run_id updated.

        Raises:
            ValueError: If run_id is not in recent_run_ids.
        """
        if run_id not in self.recent_run_ids:
            msg = f"run_id {run_id!r} not in recent_run_ids"
            raise ValueError(msg)
        return ModelSessionIndex(
            active_run_id=run_id,
            recent_run_ids=self.recent_run_ids,
            updated_at=datetime.now(UTC),
        )


__all__: list[str] = ["ModelSessionIndex"]
