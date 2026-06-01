# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Disposition of a Docker network under the janitor ownership contract."""

from __future__ import annotations

from enum import StrEnum


class EnumNetworkDisposition(StrEnum):
    """Per-network decision produced by the network janitor.

    The janitor is *bounded*: only ``RECLAIM`` networks are ever deleted, and
    a network reaches ``RECLAIM`` only when it matches a declared ownership
    rule AND is older than the rule's age threshold AND carries zero attached
    containers. Every other outcome is a flavour of *preserve*.
    """

    RECLAIM = "reclaim"
    """Matches a declared ownership rule, age-eligible, idle — safe to delete."""

    PRESERVE_UNKNOWN_OWNERSHIP = "preserve_unknown_ownership"
    """No declared ownership rule matched — preserve (never delete on a guess)."""

    PRESERVE_ACTIVE = "preserve_active"
    """Owned + age-eligible but has attached containers — an active lane."""

    PRESERVE_TOO_YOUNG = "preserve_too_young"
    """Owned but younger than the ownership rule's age threshold."""

    PRESERVE_AGE_UNKNOWN = "preserve_age_unknown"
    """Owned but creation time could not be determined — preserve, never guess."""

    PRESERVE_BUILTIN = "preserve_builtin"
    """Docker builtin network (bridge/host/none) — never a janitor target."""


__all__ = ["EnumNetworkDisposition"]
