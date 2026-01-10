# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Selector Service.

Provides selection logic for choosing a node from multiple candidates
that match capability-based discovery criteria.

Thread Safety:
    Coroutine Safety (Single Event Loop):
        This service uses an asyncio.Lock to protect round-robin state access.
        All methods that access the state are async and properly synchronized.
        Safe for concurrent use from multiple coroutines within the SAME event loop.

    Multi-Threading (Multiple Event Loops):
        NOT thread-safe across multiple event loops or threads.
        Each event loop should have its own ServiceNodeSelector instance.
        Do not share instances between threads.

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery for capability-based discovery

Example:
    >>> from omnibase_infra.services import ServiceNodeSelector, EnumSelectionStrategy
    >>> selector = ServiceNodeSelector()
    >>> selected = await selector.select(candidates, EnumSelectionStrategy.ROUND_ROBIN, "db")
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, assert_never

from omnibase_core.container import ModelONEXContainer

from omnibase_infra.services.enum_selection_strategy import EnumSelectionStrategy

if TYPE_CHECKING:
    from omnibase_infra.models.projection import ModelRegistrationProjection

logger = logging.getLogger(__name__)

DEFAULT_SELECTION_KEY: str = "_default"
"""Default key for round-robin state tracking when no selection_key is provided."""


class ServiceNodeSelector:
    """Selects a node from candidates using various strategies.

    This service implements node selection logic for capability-based discovery.
    When multiple nodes match a capability query, this selector chooses one
    based on the configured strategy.

    Note:
        Coroutine-safe within a single event loop (uses asyncio.Lock).
        NOT thread-safe across multiple event loops - create separate instances
        per event loop or thread.

    Strategies:
        - FIRST: Return first candidate (deterministic)
        - RANDOM: Random selection (stateless load distribution)
        - ROUND_ROBIN: Sequential cycling (stateful, even distribution)
        - LEAST_LOADED: Not implemented (raises NotImplementedError)

    State Management:
        Round-robin state is tracked per selection_key. This allows independent
        cycling for different dependency types (e.g., "db" vs "consul").
        All state access is protected by an asyncio.Lock for coroutine safety.

    Example:
        >>> selector = ServiceNodeSelector()
        >>>
        >>> # First strategy - always returns first
        >>> node = await selector.select(candidates, EnumSelectionStrategy.FIRST)
        >>>
        >>> # Round-robin with key tracking
        >>> node1 = await selector.select(candidates, EnumSelectionStrategy.ROUND_ROBIN, "db")
        >>> node2 = await selector.select(candidates, EnumSelectionStrategy.ROUND_ROBIN, "db")
        >>> # node1 and node2 will be different if len(candidates) > 1

    Attributes:
        _round_robin_state: Internal state tracking last index per selection key.
        _round_robin_lock: asyncio.Lock protecting state access.
    """

    def __init__(self, container: ModelONEXContainer | None = None) -> None:
        """Initialize the node selector with empty round-robin state and lock.

        Args:
            container: Optional ONEX container for dependency injection.
                When provided, enables access to shared services and configuration.
        """
        self._container = container
        self._round_robin_state: dict[str, int] = {}
        self._round_robin_lock: asyncio.Lock = asyncio.Lock()

    async def select(
        self,
        candidates: list[ModelRegistrationProjection],
        strategy: EnumSelectionStrategy,
        selection_key: str | None = None,
        correlation_id: str | None = None,
    ) -> ModelRegistrationProjection | None:
        """Select a node from candidates using the specified strategy.

        Args:
            candidates: List of nodes matching capability criteria.
            strategy: Selection strategy to use. Must be one of:
                - FIRST: Return first candidate (deterministic)
                - RANDOM: Random selection (stateless load distribution)
                - ROUND_ROBIN: Sequential cycling (stateful, even distribution)
                - LEAST_LOADED: Not yet implemented (raises NotImplementedError)
            selection_key: Optional key for state tracking (recommended for round-robin).
                If None, round-robin uses a shared "_default" key.
                Different keys maintain independent round-robin sequences.
            correlation_id: Optional correlation ID for distributed tracing.
                When provided, included in all log messages for request tracking.

        Returns:
            Selected node, or None if candidates is empty.

        Raises:
            NotImplementedError: If LEAST_LOADED strategy is requested.
            AssertionError: If strategy is not a valid EnumSelectionStrategy value.
                This should never happen with properly typed code, as all enum
                values are explicitly handled. The check exists for runtime
                safety and to ensure exhaustive handling if new enum values
                are added.

        Example:
            >>> selector = ServiceNodeSelector()
            >>>
            >>> # Empty candidates
            >>> result = await selector.select([], EnumSelectionStrategy.FIRST)
            >>> result is None
            True
            >>>
            >>> # First strategy
            >>> result = await selector.select(candidates, EnumSelectionStrategy.FIRST)
            >>> result == candidates[0]
            True
        """
        if not candidates:
            logger.debug(
                "No candidates provided for selection",
                extra={"correlation_id": correlation_id},
            )
            return None

        if len(candidates) == 1:
            logger.debug(
                "Single candidate, returning directly",
                extra={
                    "entity_id": str(candidates[0].entity_id),
                    "correlation_id": correlation_id,
                },
            )
            return candidates[0]

        if strategy == EnumSelectionStrategy.FIRST:
            return self._select_first(candidates, correlation_id)
        elif strategy == EnumSelectionStrategy.RANDOM:
            return self._select_random(candidates, correlation_id)
        elif strategy == EnumSelectionStrategy.ROUND_ROBIN:
            return await self._select_round_robin(
                candidates, selection_key, correlation_id
            )
        elif strategy == EnumSelectionStrategy.LEAST_LOADED:
            raise NotImplementedError(
                f"LEAST_LOADED selection strategy is not yet implemented "
                f"(selection_key={selection_key!r}). "
                "Use FIRST, RANDOM, or ROUND_ROBIN instead."
            )
        else:
            # Type-safe exhaustiveness check: ensures all enum values are handled.
            # If a new EnumSelectionStrategy value is added, the type checker will
            # flag this as an error because strategy won't narrow to Never.
            assert_never(strategy)

    def _select_first(
        self,
        candidates: list[ModelRegistrationProjection],
        correlation_id: str | None = None,
    ) -> ModelRegistrationProjection:
        """Select the first candidate (deterministic).

        Args:
            candidates: Non-empty list of candidates.
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            First candidate in the list.
        """
        selected = candidates[0]
        logger.debug(
            "Selected first candidate",
            extra={
                "entity_id": str(selected.entity_id),
                "total_candidates": len(candidates),
                "correlation_id": correlation_id,
            },
        )
        return selected

    def _select_random(
        self,
        candidates: list[ModelRegistrationProjection],
        correlation_id: str | None = None,
    ) -> ModelRegistrationProjection:
        """Select a random candidate.

        Args:
            candidates: Non-empty list of candidates.
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            Randomly selected candidate.
        """
        selected = random.choice(candidates)
        logger.debug(
            "Selected random candidate",
            extra={
                "entity_id": str(selected.entity_id),
                "total_candidates": len(candidates),
                "correlation_id": correlation_id,
            },
        )
        return selected

    async def _select_round_robin(
        self,
        candidates: list[ModelRegistrationProjection],
        selection_key: str | None = None,
        correlation_id: str | None = None,
    ) -> ModelRegistrationProjection:
        """Select the next candidate in round-robin sequence.

        State is tracked per selection_key, allowing independent cycling
        for different dependency types. Access is protected by asyncio.Lock.

        Args:
            candidates: Non-empty list of candidates.
            selection_key: Key for state tracking. If None, uses "_default".
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            Next candidate in the round-robin sequence.
        """
        key = selection_key or DEFAULT_SELECTION_KEY

        async with self._round_robin_lock:
            # Get current index, default to -1 (so first selection is index 0)
            last_index = self._round_robin_state.get(key, -1)
            next_index = (last_index + 1) % len(candidates)

            # Update state
            self._round_robin_state[key] = next_index

            # Access candidate inside lock for transaction safety
            selected = candidates[next_index]
        logger.debug(
            "Selected round-robin candidate",
            extra={
                "entity_id": str(selected.entity_id),
                "selection_key": key,
                "index": next_index,
                "total_candidates": len(candidates),
                "correlation_id": correlation_id,
            },
        )
        return selected

    async def reset_round_robin_state(self, selection_key: str | None = None) -> None:
        """Reset round-robin state for a specific key or all keys.

        Access is protected by asyncio.Lock.

        Args:
            selection_key: Key to reset. If None, resets all keys.

        Example:
            >>> selector = ServiceNodeSelector()
            >>> await selector.reset_round_robin_state("db")  # Reset specific key
            >>> await selector.reset_round_robin_state()  # Reset all keys
        """
        async with self._round_robin_lock:
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

    async def get_round_robin_state(self) -> dict[str, int]:
        """Get a copy of the current round-robin state.

        Access is protected by asyncio.Lock.

        Returns:
            Dictionary mapping selection keys to their last used index.

        Example:
            >>> selector = ServiceNodeSelector()
            >>> state = await selector.get_round_robin_state()
            >>> print(state)
            {'db': 2, 'consul': 0}
        """
        async with self._round_robin_lock:
            return dict(self._round_robin_state)


__all__: list[str] = ["DEFAULT_SELECTION_KEY", "ServiceNodeSelector"]
