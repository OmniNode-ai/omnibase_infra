# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The container generation a post-merge lab pass was bound to.

Ticket: OMN-20772
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelPostMergeLabPassGeneration(BaseModel):
    """One running container, identified so two reads can be compared.

    ``container_ref`` is the full container id docker reports.

    All four fields are empty together when the provenance was unreadable; the
    event then carries ``binding_valid=False``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    container: str
    container_ref: str
    image: str
    revision: str


__all__ = ["ModelPostMergeLabPassGeneration"]
