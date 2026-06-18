# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-contract attach status for the boot interleave (OMN-13237, §3.10).

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
"""

from __future__ import annotations

from enum import Enum


class EnumContractAttachStatus(str, Enum):
    """Per-contract attach outcome surfaced on the readiness endpoint (§3.10).

    Values:
        ATTACHED: The contract's consumer was attached after readiness confirm.
        NOT_READY: Provision/readiness failed; consumer attach was skipped.
        FAILED: Consumer attach itself raised after readiness passed.
    """

    ATTACHED = "attached"
    NOT_READY = "not_ready"
    FAILED = "failed"


__all__: list[str] = ["EnumContractAttachStatus"]
