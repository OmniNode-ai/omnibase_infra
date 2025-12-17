# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM Trigger Enum for Dual Registration Workflow.

This module provides EnumFSMTrigger for representing the triggers/events that
cause state transitions in the dual registration finite state machine.
"""

from __future__ import annotations

from enum import Enum


class EnumFSMTrigger(str, Enum):
    """FSM triggers for dual registration workflow.

    Matches triggers defined in contracts/fsm/dual_registration_reducer_fsm.yaml.

    Triggers:
        INTROSPECTION_EVENT_RECEIVED: Initial event received from event bus.
        EVENT_PARSED: Event successfully parsed into structured model.
        VALIDATION_PASSED: Event payload passed validation checks.
        VALIDATION_FAILED: Event payload failed validation checks.
        REGISTRATION_ATTEMPTS_COMPLETE: Both registration attempts finished.
        ALL_BACKENDS_SUCCEEDED: Both Consul and PostgreSQL succeeded.
        PARTIAL_SUCCESS: One backend succeeded, one failed.
        ALL_BACKENDS_FAILED: Both backends failed.
        RESULT_EMITTED: Success result emitted to event bus.
        PARTIAL_RESULT_EMITTED: Partial result emitted to event bus.
    """

    INTROSPECTION_EVENT_RECEIVED = "introspection_event_received"
    EVENT_PARSED = "event_parsed"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    REGISTRATION_ATTEMPTS_COMPLETE = "registration_attempts_complete"
    ALL_BACKENDS_SUCCEEDED = "all_backends_succeeded"
    PARTIAL_SUCCESS = "partial_success"
    ALL_BACKENDS_FAILED = "all_backends_failed"
    RESULT_EMITTED = "result_emitted"
    PARTIAL_RESULT_EMITTED = "partial_result_emitted"


__all__ = ["EnumFSMTrigger"]
