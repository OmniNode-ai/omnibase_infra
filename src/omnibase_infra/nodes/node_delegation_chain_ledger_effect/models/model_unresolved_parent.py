# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A replay envelope whose parent cannot be resolved (OMN-19729)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelUnresolvedParent(BaseModel):
    """Records a missing direct parent without pretending the graph is valid."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    child_envelope_id: UUID
    parent_envelope_id: UUID


__all__ = ["ModelUnresolvedParent"]
