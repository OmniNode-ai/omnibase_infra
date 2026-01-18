# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration reference scheme enumeration.

.. versionadded:: 0.8.0
    Initial implementation for OMN-765.

This module provides the EnumConfigRefScheme enum for identifying
configuration source types in config references.
"""

from __future__ import annotations

from enum import Enum


class EnumConfigRefScheme(str, Enum):
    """Supported configuration reference schemes.

    Identifies the type of external configuration source.

    Attributes:
        FILE: File-based configuration (local filesystem).
        ENV: Environment variable containing configuration.
        VAULT: HashiCorp Vault secret containing configuration.
    """

    FILE = "file"
    ENV = "env"
    VAULT = "vault"


__all__: list[str] = ["EnumConfigRefScheme"]
