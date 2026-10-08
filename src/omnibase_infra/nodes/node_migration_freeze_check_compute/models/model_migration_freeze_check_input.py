# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Input model for the migration_freeze_check COMPUTE node."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelMigrationFreezeCheckInput"]


class ModelMigrationFreezeCheckInput(BaseModel):
    """Everything the freeze decision needs, collected by the runtime.

    The handler reads no file, runs no git command and reads no clock: the
    runtime supplies the freeze file text, today's date and the diff's paths.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    freeze_active: bool = Field(
        description="True when a .migration_freeze file exists at the repo root."
    )
    freeze_text: str = Field(
        default="",
        description="Text of .migration_freeze; empty when the freeze is inactive.",
    )
    today: date = Field(description="Today's date in UTC, supplied by the runtime.")
    added_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Paths added (A) or renamed (R, destination path) by the diff under "
            "inspection: staged files in pre-commit mode, base...HEAD in CI."
        ),
    )
    migration_dirs: tuple[str, ...] = Field(
        default=("docker/migrations/",),
        description="Path prefixes whose new files the freeze blocks.",
    )
