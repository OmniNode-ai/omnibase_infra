# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Entries a handler registers by name, held outside the handler (OMN-21536)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelNameKeyedTable[T](BaseModel):
    """A registry of entries by unique name.

    The handler reads ``entries`` and changes the table only through ``register``,
    ``unregister`` and ``discard_all``, so its public methods assign nothing to
    ``self``.
    """

    model_config = ConfigDict(
        extra="forbid", from_attributes=True, arbitrary_types_allowed=True
    )

    entries: dict[str, T] = Field(
        default_factory=dict, description="Registered entry by name."
    )

    def register(self, name: str, entry: T) -> None:
        """Store ``entry`` under ``name``, replacing any entry of that name."""
        self.entries[name] = entry

    def unregister(self, name: str) -> bool:
        """Drop the entry named ``name``; True when there was one."""
        return self.entries.pop(name, None) is not None

    def discard_all(self) -> None:
        """Drop every entry."""
        self.entries.clear()


__all__: list[str] = ["ModelNameKeyedTable"]
