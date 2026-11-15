# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whether a merge may sync and restart one deploy-agent unit (OMN-24626)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelRestartOnMerge(BaseModel):
    """The unit asked about, the settings its process carries and, in the result, the decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit_name: str
    repo_root: str
    carried_settings: tuple[str, ...] = ()
    restartable: bool = False
    reason: str = ""
