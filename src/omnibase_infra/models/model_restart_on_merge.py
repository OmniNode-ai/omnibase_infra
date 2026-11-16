# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whether a merge may sync and restart one deploy-agent unit (OMN-24626, OMN-24677)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_unit_settings_gap import ModelUnitSettingsGap


class ModelRestartOnMerge(BaseModel):
    """The unit asked about (``verify``) or a whole host (``check_host``), and the decision.

    ``verify`` decides one unit, and names the unit a restart acts on; ``check_host``
    reads every non-prod agent unit installed under ``home`` and lists each one whose
    env files lack a required setting. ``restartable`` is true only with no gap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["verify", "check_host"] = "verify"
    unit: str = ""
    repo_root: str
    home: str = ""
    restartable: bool = False
    reason: str = ""
    gaps: tuple[ModelUnitSettingsGap, ...] = ()
