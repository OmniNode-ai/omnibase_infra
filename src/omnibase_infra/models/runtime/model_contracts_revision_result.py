# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of the contracts-revision guard (OMN-25856)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelContractsRevisionResult(BaseModel):
    """The guard passed after judging ``files_checked`` mounted files."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    files_checked: int
