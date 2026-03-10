# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Delivery status enum for retry worker.

Related Tickets:
    - OMN-1454: Implement RetryWorker for subscription notification delivery
"""

from __future__ import annotations

from enum import Enum


class EnumDeliveryStatus(str, Enum):
    """Status of a notification delivery attempt.

    Attributes:
        PENDING: Initial state, delivery not yet attempted.
        FAILED: Delivery failed, eligible for retry.
        SUCCEEDED: Delivery completed successfully.
        DLQ: Maximum retries exceeded, moved to dead letter queue.
    """

    PENDING = "pending"
    FAILED = "failed"
    SUCCEEDED = "succeeded"
    DLQ = "dlq"


__all__ = ["EnumDeliveryStatus"]
