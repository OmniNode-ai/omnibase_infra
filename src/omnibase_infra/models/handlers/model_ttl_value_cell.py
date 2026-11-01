# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One value a handler caches for a bounded time, held outside the handler (OMN-21536)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelTtlValueCell[T](BaseModel):
    """A cached value and the instant it was stored.

    The handler owns the cell as an injected collaborator and asks it for the value
    while it is fresh, so the handler's own public methods assign nothing to ``self``.
    """

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    ttl_seconds: float = Field(
        ..., gt=0, description="Seconds a stored value stays fresh."
    )
    value: T | None = Field(default=None, description="The cached value, if any.")
    stored_at: float = Field(
        default=0.0, description="Clock reading at the last store."
    )

    def fresh(self, now: float) -> T | None:
        """Return the value when it was stored less than ``ttl_seconds`` before ``now``."""
        if self.value is not None and now - self.stored_at < self.ttl_seconds:
            return self.value
        return None

    def store(self, value: T, now: float) -> None:
        """Keep ``value`` as fresh at ``now``."""
        self.value = value
        self.stored_at = now

    def discard(self) -> None:
        """Forget the value and reset the stored instant."""
        self.value = None
        self.stored_at = 0.0


__all__: list[str] = ["ModelTtlValueCell"]
