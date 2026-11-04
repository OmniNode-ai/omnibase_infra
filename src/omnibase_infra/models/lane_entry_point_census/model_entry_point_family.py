# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A named group of entry points that the census table reports as one row."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEntryPointFamily(BaseModel):
    """An entry point whose label matches ``pattern`` is reported under ``label``."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    label: str = Field(..., min_length=1, description="Row label in the census table.")
    pattern: str = Field(
        ...,
        min_length=1,
        description="Regular expression matched against the entry point.",
    )


__all__: list[str] = ["ModelEntryPointFamily"]
