# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Why a declared task class cannot be routed yet (OMN-13966).

The task-class contract marks a class it declares but cannot serve with a
``routing_availability`` block (OMN-16811), and ``onex delegate`` refuses an
explicit ``--task-type`` naming that class in the block's own words. The
status vocabulary is closed: a status this enum does not name is a contract
change the CLI has not been taught, and it fails closed rather than guessing
whether the class routes. omnimarket's
``tests/unit/inference/test_task_class_admission_omn13966.py`` pins the live
contract to these values.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["EnumRoutingAvailabilityStatus"]


class EnumRoutingAvailabilityStatus(StrEnum):
    """The declared reason a task class resolves no backend on any tier."""

    #: No tier can supply a capability the class requires (agent_delegation).
    PENDING_CAPABILITY = "pending_capability"
