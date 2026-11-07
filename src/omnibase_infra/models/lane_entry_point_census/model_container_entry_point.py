# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One runtime-image container with the command it was started with."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelContainerEntryPoint(BaseModel):
    """A runtime-image container, its entrypoint and cmd as run, and the verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    container: str = Field(..., min_length=1)
    image: str = Field(..., min_length=1)
    state: str = Field(default="")
    entrypoint: tuple[str, ...] = Field(default=())
    cmd: tuple[str, ...] = Field(default=())
    command: str = Field(
        default="",
        description="Entrypoint and cmd joined as run, launcher prefix included.",
    )
    entry_point: str = Field(
        default="",
        description="The command after the launcher prefix, e.g. 'onex-runtime' or 'python -m pkg.mod'.",
    )
    family: str = Field(
        default="", description="The table row this entry point is reported under."
    )
    verdict: Literal["allowed", "remove"] = Field(
        ..., description="'remove' when the entry point is outside the allowed list."
    )


__all__: list[str] = ["ModelContainerEntryPoint"]
