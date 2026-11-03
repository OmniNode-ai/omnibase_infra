# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Public terminal summary; response bodies and exception details stay off the bus."""

from pydantic import BaseModel, ConfigDict, Field


class ModelApiFixtureRecordResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    pair_count: int = 0
    error_code: str = ""
    before: dict[str, int] = Field(default_factory=dict)
    after: dict[str, int] = Field(default_factory=dict)
    cleanup_completed: bool = False
