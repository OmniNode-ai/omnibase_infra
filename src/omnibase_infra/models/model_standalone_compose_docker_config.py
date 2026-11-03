# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host process configuration supplied to the standalone Docker effect."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelStandaloneComposeDockerConfig(BaseModel):
    """Host working directory, source clone, process environment and contract-declared timeout.

    ``repo_root`` is the host tree the compose file runs from. ``source_root`` is the git clone the
    target's carried files are read from at the requested ref.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    repo_root: Path
    source_root: Path
    environment: dict[str, str]
    command: tuple[str, ...] = ("docker",)
    timeout_seconds: float = Field(gt=0)
