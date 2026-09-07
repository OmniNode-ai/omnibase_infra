# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Phase of the contract-attach readiness gate (OMN-17372).

Related Tickets:
    - OMN-17372: ``/ready`` returned 200 with zero command topics subscribed,
      because ``required_for_readiness=True`` was passed at exactly three sites
      (the contract-registry control topics) and at none of the 328 auto-wired
      command/event topics.
"""

from __future__ import annotations

from enum import Enum


class EnumContractAttachGatePhase(str, Enum):
    """Where the runtime is in the contract-attach lifecycle.

    Values:
        WIRING_IN_PROGRESS: The boot interleave has not reported a result for
            every required contract yet. Fail-closed: NOT ready.
        BLOCKED: Every required contract has reported, and at least one is
            NOT_READY or FAILED. NOT ready.
        ATTACHED: Every required contract reported ATTACHED. Ready.
    """

    WIRING_IN_PROGRESS = "wiring_in_progress"
    BLOCKED = "blocked"
    ATTACHED = "attached"


__all__: list[str] = ["EnumContractAttachGatePhase"]
