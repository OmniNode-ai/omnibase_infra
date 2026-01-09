# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared classification security level mapping.

This module provides a single source of truth for data classification
security level mappings. Both registration-time validation and
invocation-time enforcement use this shared mapping to ensure
consistent security decisions.

Usage:
    >>> from omnibase_infra.models.security.classification_levels import (
    ...     CLASSIFICATION_SECURITY_LEVELS,
    ...     get_security_level,
    ... )
    >>> from omnibase_core.enums import EnumDataClassification
    >>>
    >>> level = get_security_level(EnumDataClassification.INTERNAL)
    >>> level
    2

Security Level Hierarchy (lowest to highest):
    0: PUBLIC, OPEN (publicly accessible)
    1: UNCLASSIFIED (not classified but not public)
    2: INTERNAL, PRIVATE (organization internal)
    3: SENSITIVE (requires extra handling)
    4: CONFIDENTIAL (business confidential)
    5: RESTRICTED, CLASSIFIED (restricted access)
    6: SECRET (secret clearance required)
    7: TOP_SECRET (highest classification)

Note:
    This module is intentionally simple with no external dependencies
    beyond omnibase_core.enums to prevent circular imports.

See Also:
    - RegistrationSecurityValidator: Uses this for registration-time checks
    - InvocationSecurityEnforcer: Uses this for runtime enforcement
    - EnumDataClassification: The classification enum values
"""

from __future__ import annotations

from omnibase_core.enums import EnumDataClassification

# Security level mapping for data classification comparison.
# Higher values indicate more sensitive/restricted data.
# This ordering reflects standard data classification hierarchies.
#
# IMPORTANT: This is the SINGLE SOURCE OF TRUTH for classification levels.
# Both registration_security_validator.py and invocation_security_enforcer.py
# MUST use this mapping to ensure consistent security decisions.
CLASSIFICATION_SECURITY_LEVELS: dict[EnumDataClassification, int] = {
    EnumDataClassification.PUBLIC: 0,
    EnumDataClassification.OPEN: 0,
    EnumDataClassification.UNCLASSIFIED: 1,
    EnumDataClassification.INTERNAL: 2,
    EnumDataClassification.PRIVATE: 2,
    EnumDataClassification.SENSITIVE: 3,
    EnumDataClassification.CONFIDENTIAL: 4,
    EnumDataClassification.RESTRICTED: 5,
    EnumDataClassification.CLASSIFIED: 5,
    EnumDataClassification.SECRET: 6,
    EnumDataClassification.TOP_SECRET: 7,
}


def get_security_level(classification: EnumDataClassification) -> int:
    """Get numeric security level for a data classification.

    This function provides consistent security level lookups for both
    registration-time validation and invocation-time enforcement.

    Args:
        classification: The data classification enum value.

    Returns:
        Integer security level (higher = more sensitive).
        Range is 0 (PUBLIC) to 7 (TOP_SECRET).

    Raises:
        KeyError: If the classification is not in the mapping.
            This should not happen with valid EnumDataClassification values.

    Example:
        >>> from omnibase_core.enums import EnumDataClassification
        >>> get_security_level(EnumDataClassification.PUBLIC)
        0
        >>> get_security_level(EnumDataClassification.CONFIDENTIAL)
        4
        >>> get_security_level(EnumDataClassification.TOP_SECRET)
        7
    """
    return CLASSIFICATION_SECURITY_LEVELS[classification]


__all__ = [
    "CLASSIFICATION_SECURITY_LEVELS",
    "get_security_level",
]
