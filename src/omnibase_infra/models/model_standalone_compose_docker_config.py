# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host process configuration supplied to the standalone Docker effect."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelStandaloneComposeDockerConfig(BaseModel):
    """Host working directory, process environment and contract-declared timeout."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    repo_root: Path
    environment: dict[str, str]
    command: tuple[str, ...] = ("docker",)
    timeout_seconds: float = Field(gt=0)
