# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why a dev lane built from a pull request's branch goes back to dev (OMN-22996)."""

from __future__ import annotations

from enum import StrEnum


class EnumDevLaneReturnReason(StrEnum):
    """The reasons a recorded branch build is due for its return to dev."""

    MERGED = "merged"
    """The pull request is on the dev branch."""

    CLOSED = "closed"
    """The pull request's branch is gone from origin and no merge is on dev."""

    EXPIRED = "expired"
    """The record reached its expiry before the pull request merged or closed."""

    LANE_ON_DEV = "lane_on_dev"
    """The lane already runs a commit on the dev branch: nothing to return."""

    SUPERSEDED = "superseded"
    """A later successful dev job built the lane, so the branch build is gone."""
