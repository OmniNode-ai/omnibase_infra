# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Enum for contract violation severity levels."""

from enum import Enum


class EnumContractViolationSeverity(str, Enum):
    """Severity levels for contract violations."""

    ERROR = "error"  # Must be fixed before merge
    WARNING = "warning"  # Should be fixed, but not blocking
    INFO = "info"  # Informational, best practice suggestion
