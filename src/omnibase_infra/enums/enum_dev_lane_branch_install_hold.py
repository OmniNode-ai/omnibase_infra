# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why a registered dev-lane branch install is held (OMN-26300)."""

from __future__ import annotations

from enum import StrEnum


class EnumDevLaneBranchInstallHold(StrEnum):
    """The reason code of one held install, shown by ``/branch-installs`` and ``status.json``."""

    WAITING_RECREATE_WINDOW = "waiting_recreate_window"
    """The reapply restart waits for the hourly recreate window."""

    LANE_LOCK_HELD_BY_DEPLOY = "lane_lock_held_by_deploy"
    """A deploy job holds the dev lane lock; the apply runs inside that job."""

    NOT_APPLIED = "not_applied"
    """The last readback found the install's files missing from the running containers."""

    NOT_CHECKED = "not_checked"
    """Registered; no readback of the running containers is recorded yet."""

    APPLY_FAILED = "apply_failed"
    """An apply was tried and a file does not read back; a person reads the detail."""
