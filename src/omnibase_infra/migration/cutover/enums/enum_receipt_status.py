# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Transformation receipt outcome."""

from enum import StrEnum


class EnumReceiptStatus(StrEnum):
    """A receipt is either wholly proven or fail-closed."""

    PASS = "pass"
    FAIL = "fail"


__all__ = ["EnumReceiptStatus"]
