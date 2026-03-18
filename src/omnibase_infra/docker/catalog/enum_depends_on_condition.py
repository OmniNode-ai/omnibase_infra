# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker Compose depends_on condition types."""

from __future__ import annotations

from enum import Enum


class EnumDependsOnCondition(str, Enum):
    """Docker Compose depends_on condition types."""

    SERVICE_HEALTHY = "service_healthy"
    SERVICE_STARTED = "service_started"
    SERVICE_COMPLETED_SUCCESSFULLY = "service_completed_successfully"
