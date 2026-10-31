# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One long-lived value a handler borrows from a collaborator (OMN-21536)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelValueHolder[T](BaseModel):
    """Holds the one live value (a connection, a restore point) a handler works with.

    The handler asks the holder for ``current`` and hands it a new value to keep, so
    the handler's own public methods assign nothing to ``self``.
    """

    model_config = ConfigDict(
        extra="forbid", from_attributes=True, arbitrary_types_allowed=True
    )

    current: T | None = Field(default=None, description="The held value, if any.")

    def hold(self, value: T) -> T:
        """Keep ``value`` and return it."""
        self.current = value
        return value

    def release(self) -> T | None:
        """Give up the held value and return what was held."""
        released, self.current = self.current, None
        return released


__all__: list[str] = ["ModelValueHolder"]
