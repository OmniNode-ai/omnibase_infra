# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Selector Service.

Provides selection logic for choosing a node from multiple candidates
that match capability-based discovery criteria.

Thread Safety:
    This service maintains round-robin state in a dictionary. For concurrent
    access, external synchronization is required. In async contexts, use
    asyncio.Lock if multiple coroutines may modify state simultaneously.

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery for capability-based discovery

Example:
    >>> from omnibase_infra.services import ServiceNodeSelector, EnumSelectionStrategy
    >>> selector = ServiceNodeSelector()
    >>> selected = selector.select(candidates, EnumSelectionStrategy.ROUND_ROBIN, "db")
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from omnibase_infra.services.enum_selection_strategy import EnumSelectionStrategy

if TYPE_CHECKING:
    from omnibase_infra.models.projection import ModelRegistrationProjection

logger = logging.getLogger(__name__)


class ServiceNodeSelector:
    """Selects a node from candidates using various strategies.

    This service implements node selection logic for capability-based discovery.
    When multiple nodes match a capability query, this selector chooses one
    based on the configured strategy.

    Strategies:
        - FIRST: Return first candidate (deterministic)
        - RANDOM: Random selection (stateless load distribution)
        - ROUND_ROBIN: Sequential cycling (stateful, even distribution)
        - LEAST_LOADED: Load-based selection (requires metrics, stub for now)

    State Management:
        Round-robin state is tracked per selection_key. This allows independent
        cycling for different dependency types (e.g., "db" vs "consul").

    Example:
        >>> selector = ServiceNodeSelector()
        >>>
        >>> # First strategy - always returns first
        >>> node = selector.select(candidates, EnumSelectionStrategy.FIRST)
        >>>
        >>> # Round-robin with key tracking
        >>> node1 = selector.select(candidates, EnumSelectionStrategy.ROUND_ROBIN, "db")
        >>> node2 = selector.select(candidates, EnumSelectionStrategy.ROUND_ROBIN, "db")
        >>> # node1 and node2 will be different if len(candidates) > 1

    Attributes:
        _round_robin_state: Internal state tracking last index per selection key.
    """

    def __init__(self) -> None:
        """Initialize the node selector with empty round-robin state."""
        self._round_robin_state: dict[str, int] = {}

    def select(
        self,
        candidates: list[ModelRegistrationProjection],
        strategy: EnumSelectionStrategy,
        selection_key: str | None = None,
    ) -> ModelRegistrationProjection | None:
        """Select a node from candidates using the specified strategy.

        Args:
            candidates: List of nodes matching capability criteria.
            strategy: Selection strategy to use.
            selection_key: Optional key for state tracking (required for round-robin).
                Different keys maintain independent round-robin sequences.

        Returns:
            Selected node, or None if candidates is empty.

        Example:
            >>> selector = ServiceNodeSelector()
            >>>
            >>> # Empty candidates
            >>> result = selector.select([], EnumSelectionStrategy.FIRST)
            >>> result is None
            True
            >>>
            >>> # First strategy
            >>> result = selector.select(candidates, EnumSelectionStrategy.FIRST)
            >>> result == candidates[0]
            True
        """
        if not candidates:
            logger.debug("No candidates provided for selection")
            return None

        if len(candidates) == 1:
            logger.debug(
                "Single candidate, returning directly",
                extra={"entity_id": str(candidates[0].entity_id)},
            )
            return candidates[0]

        if strategy == EnumSelectionStrategy.FIRST:
            return self._select_first(candidates)
        elif strategy == EnumSelectionStrategy.RANDOM:
            return self._select_random(candidates)
        elif strategy == EnumSelectionStrategy.ROUND_ROBIN:
            return self._select_round_robin(candidates, selection_key)
        elif strategy == EnumSelectionStrategy.LEAST_LOADED:
            return self._select_least_loaded(candidates)
        else:
            # Fallback to first for unknown strategies
            logger.warning(
                "Unknown selection strategy, falling back to FIRST",
                extra={"strategy": str(strategy)},
            )
            return self._select_first(candidates)

    def _select_first(
        self,
        candidates: list[ModelRegistrationProjection],
    ) -> ModelRegistrationProjection:
        """Select the first candidate (deterministic).

        Args:
            candidates: Non-empty list of candidates.

        Returns:
            First candidate in the list.
        """
        selected = candidates[0]
        logger.debug(
            "Selected first candidate",
            extra={
                "entity_id": str(selected.entity_id),
                "total_candidates": len(candidates),
            },
        )
        return selected

    def _select_random(
        self,
        candidates: list[ModelRegistrationProjection],
    ) -> ModelRegistrationProjection:
        """Select a random candidate.

        Args:
            candidates: Non-empty list of candidates.

        Returns:
            Randomly selected candidate.
        """
        selected = random.choice(candidates)
        logger.debug(
            "Selected random candidate",
            extra={
                "entity_id": str(selected.entity_id),
                "total_candidates": len(candidates),
            },
        )
        return selected

    def _select_round_robin(
        self,
        candidates: list[ModelRegistrationProjection],
        selection_key: str | None = None,
    ) -> ModelRegistrationProjection:
        """Select the next candidate in round-robin sequence.

        State is tracked per selection_key, allowing independent cycling
        for different dependency types.

        Args:
            candidates: Non-empty list of candidates.
            selection_key: Key for state tracking. If None, uses "_default".

        Returns:
            Next candidate in the round-robin sequence.
        """
        key = selection_key or "_default"

        # Get current index, default to -1 (so first selection is index 0)
        last_index = self._round_robin_state.get(key, -1)
        next_index = (last_index + 1) % len(candidates)

        # Update state
        self._round_robin_state[key] = next_index

        selected = candidates[next_index]
        logger.debug(
            "Selected round-robin candidate",
            extra={
                "entity_id": str(selected.entity_id),
                "selection_key": key,
                "index": next_index,
                "total_candidates": len(candidates),
            },
        )
        return selected

    def _select_least_loaded(
        self,
        candidates: list[ModelRegistrationProjection],
    ) -> ModelRegistrationProjection:
        """Select the least loaded candidate (stub implementation).

        This is a placeholder for future implementation that will use
        load metrics from the registry or external monitoring.

        Currently falls back to first candidate with a warning.

        Args:
            candidates: Non-empty list of candidates.

        Returns:
            First candidate (stub behavior).
        """
        logger.warning(
            "LEAST_LOADED strategy not yet implemented, falling back to FIRST. "
            "Load metrics integration required for proper implementation.",
            extra={"total_candidates": len(candidates)},
        )
        return self._select_first(candidates)

    def reset_round_robin_state(self, selection_key: str | None = None) -> None:
        """Reset round-robin state for a specific key or all keys.

        Args:
            selection_key: Key to reset. If None, resets all keys.

        Example:
            >>> selector = ServiceNodeSelector()
            >>> selector.reset_round_robin_state("db")  # Reset specific key
            >>> selector.reset_round_robin_state()  # Reset all keys
        """
        if selection_key is not None:
            if selection_key in self._round_robin_state:
                del self._round_robin_state[selection_key]
                logger.debug(
                    "Reset round-robin state for key",
                    extra={"selection_key": selection_key},
                )
        else:
            self._round_robin_state.clear()
            logger.debug("Reset all round-robin state")

    def get_round_robin_state(self) -> dict[str, int]:
        """Get a copy of the current round-robin state.

        Returns:
            Dictionary mapping selection keys to their last used index.

        Example:
            >>> selector = ServiceNodeSelector()
            >>> state = selector.get_round_robin_state()
            >>> print(state)
            {'db': 2, 'consul': 0}
        """
        return dict(self._round_robin_state)


__all__: list[str] = ["ServiceNodeSelector"]
