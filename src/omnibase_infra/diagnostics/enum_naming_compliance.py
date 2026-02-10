# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX topic naming compliance enumeration for bus health diagnostics."""

from __future__ import annotations

from enum import Enum


class EnumNamingCompliance(str, Enum):
    """ONEX topic naming convention compliance.

    Values:
        COMPLIANT: Passes ``validate_topic_suffix()`` from omnibase_core.
        LEGACY: Non-ONEX name but declared in ``legacy_topics`` config set.
        NON_COMPLIANT: Fails validation and not in legacy set.
    """

    COMPLIANT = "compliant"
    """Passes ONEX 5-segment naming validation."""

    LEGACY = "legacy"
    """Non-ONEX name, declared as known legacy topic."""

    NON_COMPLIANT = "non_compliant"
    """Fails validation and not in legacy set."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumNamingCompliance"]
