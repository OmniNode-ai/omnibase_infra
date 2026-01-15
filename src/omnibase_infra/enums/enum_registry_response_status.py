# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry response status enumeration.

This module defines the status enumeration for registry operation responses,
indicating the outcome of registry queries and mutations.
"""

from enum import Enum


class EnumRegistryResponseStatus(str, Enum):
    """Registry operation response status.

    Indicates the outcome of a registry operation.

    Attributes:
        SUCCESS: Registry operation completed successfully
        PARTIAL: Operation partially succeeded (some items processed)
        FAILED: Registry operation failed
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


__all__: list[str] = ["EnumRegistryResponseStatus"]
