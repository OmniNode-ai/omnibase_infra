# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request policy for resolving a workspace config owner."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ModelWorkspaceConfigRootRequest(BaseModel):
    """Resolve against the process environment or a captured environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    workspace_root: Path
    environment: dict[str, str] | None = None
    allow_developer_config: bool = True
