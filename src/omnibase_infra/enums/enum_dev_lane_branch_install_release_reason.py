# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why a dev-lane branch install was dropped (OMN-17427)."""

from __future__ import annotations

from enum import StrEnum


class EnumDevLaneBranchInstallReleaseReason(StrEnum):
    """The reasons a declared install leaves the record."""

    MERGED = "merged"
    """Its PR is on the dev branch and the lane's running image already carries it."""

    EXPIRED = "expired"
    """It reached its declared expiry before the PR merged."""

    OPERATOR = "operator"
    """A person or lane released it by name."""

    REPLACED = "replaced"
    """A registration of the same id with replace superseded it."""
