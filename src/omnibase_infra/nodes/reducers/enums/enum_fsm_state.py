# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM State Enum for Dual Registration Workflow.

This module provides EnumFSMState for representing the states of the dual
registration finite state machine.
"""

from __future__ import annotations

from enum import Enum


class EnumFSMState(str, Enum):
    """FSM states for dual registration workflow.

    Matches states defined in contracts/fsm/dual_registration_reducer_fsm.yaml.

    States:
        IDLE: Waiting for introspection events.
        RECEIVING_INTROSPECTION: Parsing NODE_INTROSPECTION event.
        VALIDATING_PAYLOAD: Validating event structure.
        REGISTERING_PARALLEL: Parallel registration to both backends.
        AGGREGATING_RESULTS: Combining registration outcomes.
        REGISTRATION_COMPLETE: Both backends succeeded.
        PARTIAL_FAILURE: One backend failed (graceful degradation).
        REGISTRATION_FAILED: Both backends failed.
    """

    IDLE = "idle"
    RECEIVING_INTROSPECTION = "receiving_introspection"
    VALIDATING_PAYLOAD = "validating_payload"
    REGISTERING_PARALLEL = "registering_parallel"
    AGGREGATING_RESULTS = "aggregating_results"
    REGISTRATION_COMPLETE = "registration_complete"
    PARTIAL_FAILURE = "partial_failure"
    REGISTRATION_FAILED = "registration_failed"


__all__ = ["EnumFSMState"]
