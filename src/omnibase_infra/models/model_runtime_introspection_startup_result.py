# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result: whether the startup introspection was published (OMN-1930, OMN-21590)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeIntrospectionStartupResult(BaseModel):
    """Outcome of one startup introspection attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    published: bool = Field(
        ..., description="True when the startup introspection was published"
    )
    last_introspection_time: float | None = Field(
        default=None,
        description="Monotonic time of the last published introspection, for throttling",
    )


__all__ = ["ModelRuntimeIntrospectionStartupResult"]
