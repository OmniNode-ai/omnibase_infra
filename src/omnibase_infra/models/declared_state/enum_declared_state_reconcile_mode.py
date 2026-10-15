# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whether a round may change the surface."""

from enum import StrEnum, unique


@unique
class EnumDeclaredStateReconcileMode(StrEnum):
    """Observe plans only; apply changes the kinds the policy allows."""

    OBSERVE = "observe"
    APPLY = "apply"


__all__ = ["EnumDeclaredStateReconcileMode"]
