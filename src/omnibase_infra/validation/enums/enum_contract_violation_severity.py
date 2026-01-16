# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Enum for contract violation severity levels."""

from enum import Enum


class EnumContractViolationSeverity(str, Enum):
    """Severity levels for contract violations."""

    ERROR = "error"  # Must be fixed before merge
    WARNING = "warning"  # Should be fixed, but not blocking
    INFO = "info"  # Informational, best practice suggestion
