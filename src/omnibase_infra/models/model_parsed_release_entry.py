# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A rendered release block read back into the facts the companions pin."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelParsedReleaseEntry(BaseModel):
    """Version, date, previous tag and change subjects of one release block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    date: str
    since: str
    subjects: tuple[str, ...]
