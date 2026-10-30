# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolved workspace config owner and the rule that selected it."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelWorkspaceConfigRootResolution(BaseModel):
    """The resolved path, or None when no config root is declared.

    ``source`` is ``unset`` exactly when ``config_root`` is None: the deployment
    declared no config root, so callers use their neutral default or refuse.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    config_root: Path | None
    source: Literal["environment", "developer", "unset"]
