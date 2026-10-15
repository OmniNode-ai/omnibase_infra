# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How one apply of a diff ended."""

from enum import StrEnum, unique


@unique
class EnumDeclaredStateApplyStatus(StrEnum):
    """Terminal status of an apply; every refusal names its cause."""

    APPLIED = "applied"
    REFUSED_POLICY = "refused_policy"
    REFUSED_UNOBSERVABLE = "refused_unobservable"
    REFUSED_STALE = "refused_stale"


__all__ = ["EnumDeclaredStateApplyStatus"]
