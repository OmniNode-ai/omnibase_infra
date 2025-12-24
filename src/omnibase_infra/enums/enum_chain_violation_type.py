# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chain Violation Type Enumeration.

Defines violation types for correlation and causation chain validation.
Used by the chain validation system to classify violations in message
propagation chains.

Thread Safety:
    All enums in this module are immutable and thread-safe.
    Enum values can be safely shared across threads without synchronization.
"""

from __future__ import annotations

from enum import Enum, unique

# Module-level cached descriptions for O(1) lookup without repeated dict creation.
# Must be defined outside the enum class since enums don't support ClassVar attributes
# in the same way regular classes do.
_CHAIN_VIOLATION_DESCRIPTIONS: dict[str, str] = {
    "correlation_mismatch": (
        "correlation_id doesn't match parent message's correlation_id"
    ),
    "causation_chain_broken": (
        "causation_id doesn't equal parent message's message_id"
    ),
    "causation_ancestor_skipped": (
        "causation chain skips one or more ancestors in the message lineage"
    ),
}


@unique
class EnumChainViolationType(str, Enum):
    """Violation types for correlation and causation chain validation.

    Represents specific types of violations that can occur when validating
    that messages properly maintain their correlation and causation chains
    during propagation through the system.

    Values:
        CORRELATION_MISMATCH: correlation_id doesn't match parent message's
            correlation_id. All messages in a chain must share the same
            correlation_id to enable distributed tracing.
        CAUSATION_CHAIN_BROKEN: causation_id doesn't equal parent's message_id.
            Each message's causation_id must reference its direct parent.
        CAUSATION_ANCESTOR_SKIPPED: causation chain skips one or more ancestors.
            The causation chain must form an unbroken sequence back to the
            original message.

    Example:
        >>> violation_type = EnumChainViolationType.CORRELATION_MISMATCH
        >>> str(violation_type)
        'correlation_mismatch'
        >>> violation_type.get_description()
        "correlation_id doesn't match parent message's correlation_id"
    """

    CORRELATION_MISMATCH = "correlation_mismatch"
    """correlation_id doesn't match parent message's correlation_id."""

    CAUSATION_CHAIN_BROKEN = "causation_chain_broken"
    """causation_id doesn't equal parent message's message_id."""

    CAUSATION_ANCESTOR_SKIPPED = "causation_ancestor_skipped"
    """causation chain skips one or more ancestors in the message lineage."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    def get_description(self) -> str:
        """Get a human-readable description of the violation type.

        Returns:
            A human-readable description of what this violation type means.

        Example:
            >>> EnumChainViolationType.CORRELATION_MISMATCH.get_description()
            "correlation_id doesn't match parent message's correlation_id"
        """
        return _CHAIN_VIOLATION_DESCRIPTIONS.get(
            self.value, "Unknown chain violation type"
        )

    def is_correlation_violation(self) -> bool:
        """Check if this is a correlation-related violation.

        Returns:
            True if the violation relates to correlation_id propagation.

        Example:
            >>> EnumChainViolationType.CORRELATION_MISMATCH.is_correlation_violation()
            True
            >>> EnumChainViolationType.CAUSATION_CHAIN_BROKEN.is_correlation_violation()
            False
        """
        return self == EnumChainViolationType.CORRELATION_MISMATCH

    def is_causation_violation(self) -> bool:
        """Check if this is a causation-related violation.

        Returns:
            True if the violation relates to causation_id propagation.

        Example:
            >>> EnumChainViolationType.CAUSATION_CHAIN_BROKEN.is_causation_violation()
            True
            >>> EnumChainViolationType.CORRELATION_MISMATCH.is_causation_violation()
            False
        """
        return self in {
            EnumChainViolationType.CAUSATION_CHAIN_BROKEN,
            EnumChainViolationType.CAUSATION_ANCESTOR_SKIPPED,
        }


__all__ = ["EnumChainViolationType"]
