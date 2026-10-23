# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The verdict on one dev-lane branch install (OMN-17427)."""

from __future__ import annotations

from enum import StrEnum


class EnumDevLaneBranchInstallState(StrEnum):
    """Whether one declared install is whole in the running containers."""

    PRESENT = "present"
    """Every file of every service reads back at its recorded sha256."""

    REAPPLIED = "reapplied"
    """This apply put the missing files back and every file reads back."""

    MISSING = "missing"
    """Read only: at least one file is not in its container."""

    FAILED = "failed"
    """An apply was tried and at least one file does not read back."""
