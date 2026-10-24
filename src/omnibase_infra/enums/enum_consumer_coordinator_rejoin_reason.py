# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coordinator availability decision reasons (OMN-21337)."""

from enum import StrEnum


class EnumConsumerCoordinatorRejoinReason(StrEnum):
    """Classification of one coordinator rejoin evaluation."""

    COORDINATOR_KNOWN = "coordinator_known"
    BROKER_UNREACHABLE = "broker_unreachable"
    WITHIN_BOUND = "within_bound"
    WITHIN_COOLDOWN = "within_cooldown"
    COORDINATOR_UNAVAILABLE_PAST_BOUND = "coordinator_unavailable_past_bound"


__all__ = ["EnumConsumerCoordinatorRejoinReason"]
