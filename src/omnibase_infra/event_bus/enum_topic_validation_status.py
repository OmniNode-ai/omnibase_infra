# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Topic Validation Status Enum.

Enum for the overall outcome of startup topic existence validation.

Related Tickets:
    - OMN-3769: Registry-First Startup Assertions
"""

from __future__ import annotations

from enum import Enum


class EnumTopicValidationStatus(str, Enum):
    """Overall outcome of startup topic validation.

    Values:
        SUCCESS: All required topics present on the broker.
        DEGRADED: Some required topics missing (non-fatal).
        UNAVAILABLE: Broker unreachable; validation could not run.
        SKIPPED: aiokafka not installed; validation skipped.
    """

    SUCCESS = "success"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


__all__: list[str] = [
    "EnumTopicValidationStatus",
]
