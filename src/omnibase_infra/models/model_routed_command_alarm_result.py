# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The alarms one poll wrote (OMN-26485)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_routed_command_alarm_row import (
    ModelRoutedCommandAlarmRow,
)


class ModelRoutedCommandAlarmResult(BaseModel):
    """``alarms`` holds one row per command past the window with no job record on its
    instance; ``waiting`` the commands still inside the window with none."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alarms: tuple[ModelRoutedCommandAlarmRow, ...] = ()
    waiting: tuple[UUID, ...] = ()
