# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Injected host boundary and contract policy for replacement effects."""

from pathlib import Path
from types import ModuleType

from pydantic import BaseModel, ConfigDict, Field


class ModelDeployAgentReplacementConfig(BaseModel):
    """Host I/O belongs to the caller; admission policy belongs to the contract."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    execution: ModuleType
    ledger_path: Path
    clones_root: str
    git_environment: dict[str, str]
    python_executable: str
    gate_timeout: int
    ref_timeout: int
    removed: list[str] = Field(default_factory=list)
