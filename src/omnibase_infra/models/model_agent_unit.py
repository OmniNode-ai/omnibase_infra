# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One unit-drift manifest entry as the restart decision reads it (OMN-24677)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelAgentUnit(BaseModel):
    """A manifest entry's unit name and the repository path of its tracked unit."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str
    tracked: str
