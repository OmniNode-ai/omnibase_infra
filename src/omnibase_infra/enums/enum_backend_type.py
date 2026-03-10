# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Backend type enumeration.

This module defines the backend type enumeration for infrastructure
registration and service discovery operations.
"""

from enum import Enum


class EnumBackendType(str, Enum):
    """Infrastructure backend types.

    Identifies the backend infrastructure service for registration
    and service discovery operations.

    Attributes:
        POSTGRES: PostgreSQL database for persistent registration storage
    """

    POSTGRES = "postgres"


__all__: list[str] = ["EnumBackendType"]
