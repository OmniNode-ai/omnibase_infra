# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Selection Strategy Enumeration.

Defines strategies for selecting a node from multiple candidates that match
capability-based discovery criteria.

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery for capability-based discovery

Example:
    >>> from omnibase_infra.enums import EnumSelectionStrategy
    >>> strategy = EnumSelectionStrategy.ROUND_ROBIN
    >>> print(strategy.value)
    'round_robin'
"""

from enum import Enum, unique


@unique
class EnumSelectionStrategy(str, Enum):
    """Selection strategies for choosing among multiple capability matches.

    When capability-based discovery finds multiple nodes that match the
    requested criteria, this strategy determines which node is selected.

    Values:
        FIRST: Return the first candidate (deterministic, fast).
            Best for: Development, testing, or when order matters.
        RANDOM: Randomly select from candidates.
            Best for: Simple load distribution without state.
        ROUND_ROBIN: Cycle through candidates sequentially.
            Best for: Even distribution with state tracking.
        LEAST_LOADED: Select based on current load metrics.
            Best for: Production load balancing (requires metrics).

    Example:
        >>> strategy = EnumSelectionStrategy.FIRST
        >>> strategy.value
        'first'
        >>> str(EnumSelectionStrategy.ROUND_ROBIN)
        'round_robin'
    """

    FIRST = "first"
    """Return the first candidate (deterministic, fast)."""

    RANDOM = "random"
    """Randomly select from candidates."""

    ROUND_ROBIN = "round_robin"
    """Cycle through candidates sequentially."""

    LEAST_LOADED = "least_loaded"
    """Select based on current load metrics (future implementation)."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    def requires_state(self) -> bool:
        """Check if this strategy requires state tracking.

        Returns:
            True if the strategy needs persistent state (e.g., round-robin index).

        Example:
            >>> EnumSelectionStrategy.ROUND_ROBIN.requires_state()
            True
            >>> EnumSelectionStrategy.FIRST.requires_state()
            False
        """
        return self in {EnumSelectionStrategy.ROUND_ROBIN}

    def requires_metrics(self) -> bool:
        """Check if this strategy requires load metrics.

        Returns:
            True if the strategy needs external metrics (e.g., least-loaded).

        Example:
            >>> EnumSelectionStrategy.LEAST_LOADED.requires_metrics()
            True
            >>> EnumSelectionStrategy.RANDOM.requires_metrics()
            False
        """
        return self == EnumSelectionStrategy.LEAST_LOADED


__all__: list[str] = ["EnumSelectionStrategy"]
