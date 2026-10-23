# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What one file of a dev-lane branch install reads as in a container (OMN-17427)."""

from __future__ import annotations

from enum import StrEnum


class EnumDevLaneBranchInstallFileState(StrEnum):
    """The readback state of one file of one install in one container."""

    PRESENT = "present"
    """The file hashes to the recorded sha256."""

    REAPPLIED = "reapplied"
    """The file was put back by this apply and read back at the recorded sha256."""

    MISSING = "missing"
    """The container has no file at the recorded path."""

    DRIFTED = "drifted"
    """The file exists and hashes to something else."""

    UNREADABLE = "unreadable"
    """The container could not say what the file hashes to."""

    FAILED = "failed"
    """An apply was tried and the file does not read back at the recorded sha256."""

    NO_CONTAINER = "no_container"
    """The service has no single running container on the lane."""
