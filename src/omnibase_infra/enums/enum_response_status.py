# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler response status enumeration.

This module defines the status enumeration for handler responses,
indicating whether the handler operation completed successfully or with an error.
"""

from enum import Enum


class EnumResponseStatus(str, Enum):
    """Handler response status for operation results.

    Indicates the outcome of a handler operation.

    Attributes:
        SUCCESS: Handler operation completed successfully
        ERROR: Handler operation failed with an error
    """

    SUCCESS = "success"
    ERROR = "error"


__all__: list[str] = ["EnumResponseStatus"]
