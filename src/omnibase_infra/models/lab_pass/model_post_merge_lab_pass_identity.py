# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The deployment identity a post-merge lab pass attests to.

Ticket: OMN-20772
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPostMergeLabPassIdentity(BaseModel):
    """What the lane was running when the pass was taken.

    ``pins`` maps a dependency name to its pinned spec, ``image_ids`` maps a
    container name to its image id, ``overlay_digest`` is the sha256 of the
    deployed compose overlay and ``migration_fingerprint`` is the sha256 of
    the lane's applied-migration ids. All are empty when the provenance was
    unreadable; the event then carries ``binding_valid=False``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pins: dict[str, str] = Field(default_factory=dict)
    image_ids: dict[str, str] = Field(default_factory=dict)
    overlay_digest: str = ""
    migration_fingerprint: str = ""

    @property
    def complete(self) -> bool:
        return bool(
            self.pins
            and self.image_ids
            and self.overlay_digest
            and self.migration_fingerprint
        )


__all__ = ["ModelPostMergeLabPassIdentity"]
