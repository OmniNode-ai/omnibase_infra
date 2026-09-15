# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""One lab host's load and free memory.

Ticket: OMN-18412
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelCIRunnerLabHost(BaseModel):
    """One lab host's load and free memory."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    label: str = Field(description="Host label the reading came from.")
    ratio: float = Field(ge=0.0, description="load1 divided by core count.")
    free_mem_mib: int = Field(ge=0, description="Free memory in MiB.")


__all__ = ["ModelCIRunnerLabHost"]
