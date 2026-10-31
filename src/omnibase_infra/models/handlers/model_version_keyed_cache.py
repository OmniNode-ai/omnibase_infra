# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Snapshots a handler caches by source version, held outside the handler (OMN-21536)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelVersionKeyedCache[T](BaseModel):
    """Snapshots keyed by the integer version of their source.

    Keeping a snapshot drops every snapshot of an older version, so the cache never
    holds more than the versions at or above the newest one stored.
    """

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    entries: dict[int, T] = Field(
        default_factory=dict, description="Snapshot by source version."
    )

    def lookup(self, version: int) -> T | None:
        """Return the snapshot of ``version``, or ``None`` when none is held."""
        return self.entries.get(version)

    def keep(self, version: int, snapshot: T) -> None:
        """Store ``snapshot`` for ``version`` and drop every older version."""
        self.entries[version] = snapshot
        for stale in [v for v in self.entries if v < version]:
            del self.entries[stale]

    def discard_all(self) -> None:
        """Forget every snapshot."""
        self.entries.clear()


__all__: list[str] = ["ModelVersionKeyedCache"]
