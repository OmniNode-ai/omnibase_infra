# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner-fleet alert kind enum (OMN-17427)."""

from __future__ import annotations

from enum import StrEnum


class EnumRunnerFleetAlertKind(StrEnum):
    """Condition surfaced by a runner-fleet health alert."""

    RUNNER_STUCK = "runner_stuck"
    RUNNER_OFFLINE = "runner_offline"
    QUEUE_AGE_EXCEEDED = "queue_age_exceeded"


__all__ = ["EnumRunnerFleetAlertKind"]
