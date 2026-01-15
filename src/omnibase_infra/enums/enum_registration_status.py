# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration status enumeration.

This module defines the status enumeration for registration operations,
tracking the overall progress of multi-backend registration workflows.
"""

from enum import Enum


class EnumRegistrationStatus(str, Enum):
    """Registration workflow status.

    Tracks the overall status of a registration operation that may involve
    multiple backends (e.g., Consul and PostgreSQL).

    Attributes:
        IDLE: Registration not started
        PENDING: Registration in progress, awaiting backend confirmations
        PARTIAL: Some backends confirmed, others pending or failed
        COMPLETE: All backends confirmed successfully
        FAILED: Registration failed across all backends
    """

    IDLE = "idle"
    PENDING = "pending"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"


__all__: list[str] = ["EnumRegistrationStatus"]
