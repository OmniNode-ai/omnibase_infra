# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration status enumeration.

This module defines the status enumeration for registration operations,
tracking the overall progress of multi-backend registration workflows.
"""

from enum import Enum


class EnumRegistrationStatus(str, Enum):
    """Registration workflow status.

    Tracks the overall status of a registration operation.

    Attributes:
        IDLE: Registration not started
        PENDING: Registration in progress, awaiting backend confirmation
        PARTIAL: Deprecated — previously used for multi-backend partial success
        COMPLETE: All backends confirmed successfully
        FAILED: Registration failed
    """

    IDLE = "idle"
    PENDING = "pending"
    PARTIAL = "partial"  # Deprecated: no longer reachable after OMN-3540 consul removal
    COMPLETE = "complete"
    FAILED = "failed"


__all__: list[str] = ["EnumRegistrationStatus"]
