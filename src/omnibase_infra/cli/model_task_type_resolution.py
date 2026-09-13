# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The resolved task class, how it was resolved, and why (OMN-18305)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_task_type_resolution import EnumTaskTypeResolution

__all__ = ["ModelTaskTypeResolution"]


class ModelTaskTypeResolution(BaseModel):
    """The resolved class, how it was resolved, and why — all three on the record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_type: str = Field(min_length=1)
    resolution: EnumTaskTypeResolution
    reason: str = Field(min_length=1)
