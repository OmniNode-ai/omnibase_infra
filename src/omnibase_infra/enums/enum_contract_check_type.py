# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Contract check type classification for runtime verification probes."""

from __future__ import annotations

from enum import Enum


class EnumContractCheckType(str, Enum):
    """Classifies the type of runtime verification check performed against a contract.

    Each member corresponds to a specific probe class that verifies one aspect
    of contract compliance at runtime.

    Values:
        REGISTRATION: Verifies the node is registered in the registration_projections table.
        SUBSCRIPTION: Verifies the node's consumer group subscribes to declared topics.
        PUBLICATION: Verifies declared publish_topics have non-zero offsets.
        HANDLER_EXECUTION: Verifies declared handlers execute when triggered.
        PROJECTION_STATE: Verifies projection tables reflect completed FSM cycles.
        FSM_STATE: Verifies FSM state machine transitions are consistent with contract.
    """

    REGISTRATION = "registration"
    """Node is registered in the registration_projections table."""

    SUBSCRIPTION = "subscription"
    """Consumer group subscribes to all declared subscribe_topics."""

    PUBLICATION = "publication"
    """Declared publish_topics have non-zero high-water-mark offsets."""

    HANDLER_EXECUTION = "handler_execution"
    """Declared handlers produce expected output events when triggered."""

    PROJECTION_STATE = "projection_state"
    """Projection tables contain rows in terminal FSM states."""

    FSM_STATE = "fsm_state"
    """FSM state machine transitions are consistent with contract declarations."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumContractCheckType"]
