# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Recommended-action type enum (OMN-13942).

Increment 1 only RECORDS/SURFACES these recommendations -- nothing in this
repo executes them. Execution (Increment 2, design-only) would route through
``node_runner_fleet_remediate_effect`` (RESTART_RUNNER / CANCEL_RUN) and the
existing ``/onex:unstick_queue`` GraphQL primitive (DEQUEUE_REENQUEUE).
"""

from __future__ import annotations

from enum import StrEnum


class EnumRecommendedActionType(StrEnum):
    """Type of remediation action a health assessment recommends (never executed here)."""

    RESTART_RUNNER = "restart_runner"
    CANCEL_RUN = "cancel_run"
    DEQUEUE_REENQUEUE = "dequeue_reenqueue"
    NONE = "none"


__all__ = ["EnumRecommendedActionType"]
