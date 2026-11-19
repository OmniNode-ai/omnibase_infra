# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of the local provision node's ``engine_info`` read (OMN-23413)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLocalEngineInfoResult(BaseModel):
    """The Docker engine's CPU count and memory, or why they could not be read."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(..., description="True only when engine_status is read.")
    correlation_id: UUID = Field(..., description="Correlation ID for tracing.")
    engine_status: Literal["read", "unavailable", "malformed"] = Field(
        ...,
        description=(
            "read, unavailable (Docker missing or the engine not answering) or "
            "malformed (no positive whole NCPU and MemTotal)."
        ),
    )
    engine_cpus: int | None = Field(
        default=None, ge=1, description="CPUs the Docker engine exposes."
    )
    engine_memory_bytes: int | None = Field(
        default=None, ge=1, description="Memory the Docker engine exposes, in bytes."
    )
    error: str | None = Field(
        default=None, description="Why the read failed, when it did."
    )


__all__: list[str] = ["ModelLocalEngineInfoResult"]
