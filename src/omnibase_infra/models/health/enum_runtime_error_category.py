# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime error category classification enum.

Ticket: OMN-5513
"""

from __future__ import annotations

from enum import StrEnum


class EnumRuntimeErrorCategory(StrEnum):
    """Classification of runtime errors by subsystem."""

    KAFKA_CONSUMER = "kafka_consumer"
    KAFKA_PRODUCER = "kafka_producer"
    DATABASE = "database"
    HTTP_CLIENT = "http_client"
    HTTP_SERVER = "http_server"
    RUNTIME = "runtime"
    UNKNOWN = "unknown"


__all__ = ["EnumRuntimeErrorCategory"]
