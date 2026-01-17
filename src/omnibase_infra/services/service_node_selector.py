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

State Management:
    Round-Robin State Growth:
        The round-robin state dictionary grows as new selection_key values are used.
        To prevent unbounded growth, an LRU eviction mechanism is implemented:
        - Default max entries: 1000 (configurable via max_round_robin_entries)
        - When limit is reached, oldest 10% of entries are evicted
        - Use reset_round_robin_state() for manual cleanup
        - Call prune_round_robin_state() for explicit eviction

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
import time
from typing import TYPE_CHECKING, NamedTuple, Never
from uuid import UUID, uuid4

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumSelectionStrategy
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError

if TYPE_CHECKING:
    from omnibase_infra.models.projection import ModelRegistrationProjection

logger = logging.getLogger(__name__)

DEFAULT_SELECTION_KEY: str = "_default"
"""Default key for round-robin state tracking when no selection_key is provided."""

DEFAULT_MAX_ROUND_ROBIN_ENTRIES: int = 1000
"""Default maximum number of round-robin state entries before LRU eviction."""

EVICTION_PERCENTAGE: float = 0.1
"""Percentage of entries to evict when max limit is reached (10%)."""


class RoundRobinEntry(NamedTuple):
    """Round-robin state entry with LRU tracking.

    Attributes:
        current_index: The last used index for this selection key.
        last_access: Timestamp of last access for LRU eviction.
    """

    current_index: int
    last_access: float


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
        - LEAST_LOADED: Not implemented (raises RuntimeHostError)

    State Management:
        Round-robin state is tracked per selection_key. This allows independent
        cycling for different dependency types (e.g., "db" vs "consul").
        All state access is protected by an asyncio.Lock for coroutine safety.

        To prevent unbounded memory growth, the round-robin state implements
        LRU eviction. When the number of entries exceeds max_round_robin_entries,
        the oldest 10% of entries (by last access time) are evicted.

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
        _round_robin_state: Internal state tracking index and last access per key.
        _round_robin_lock: asyncio.Lock protecting state access.
        _max_round_robin_entries: Maximum entries before LRU eviction triggers.
    """

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        max_round_robin_entries: int = DEFAULT_MAX_ROUND_ROBIN_ENTRIES,
    ) -> None:
        """Initialize the node selector with empty round-robin state and lock.

        Args:
            container: Optional ONEX container for dependency injection.
                Stored for interface compliance and future DI integration.
            max_round_robin_entries: Maximum number of round-robin state entries
                before LRU eviction. Defaults to 1000. Set to 0 for unlimited
                (not recommended in production).

        Note:
            The container is stored for interface compliance with the standard ONEX
            service pattern and to enable future DI-based enhancements (e.g., metrics
            reporting, configuration injection). Currently, the selector operates
            with stateless selection logic that doesn't require container services.
        """
        self._container = container
        self._round_robin_state: dict[str, RoundRobinEntry] = {}
        self._round_robin_lock: asyncio.Lock = asyncio.Lock()
        self._max_round_robin_entries = max_round_robin_entries

    def _ensure_correlation_id(self, correlation_id: UUID | None) -> UUID:
        """Ensure correlation ID is present, generating one if missing.

        Args:
            correlation_id: Optional correlation ID from caller.

        Returns:
            The provided correlation ID, or a newly generated UUID4.
        """
        return correlation_id or uuid4()

    async def _prune_round_robin_state(self) -> int:
        """Prune oldest round-robin entries when limit is exceeded.

        This method is called automatically when the state dictionary exceeds
        max_round_robin_entries. It evicts the oldest 10% of entries based on
        last access time.

        Must be called while holding _round_robin_lock.

        Returns:
            Number of entries evicted.
        """
        if (
            self._max_round_robin_entries <= 0
            or len(self._round_robin_state) <= self._max_round_robin_entries
        ):
            return 0

        # Calculate how many to evict (10% of max, minimum 1)
        evict_count = max(1, int(self._max_round_robin_entries * EVICTION_PERCENTAGE))

        # Sort by last_access and get oldest entries
        sorted_keys = sorted(
            self._round_robin_state.keys(),
            key=lambda k: self._round_robin_state[k].last_access,
        )

        # Evict oldest entries
        keys_to_evict = sorted_keys[:evict_count]
        for key in keys_to_evict:
            del self._round_robin_state[key]

        logger.debug(
            "Pruned round-robin state entries (LRU eviction)",
            extra={
                "evicted_count": len(keys_to_evict),
                "remaining_count": len(self._round_robin_state),
                "max_entries": self._max_round_robin_entries,
            },
        )
        return len(keys_to_evict)

    async def prune_round_robin_state(self) -> int:
        """Explicitly prune oldest round-robin entries.

        This is the public interface for triggering LRU eviction. Useful for
        maintenance operations or when you want to proactively free memory.

        Returns:
            Number of entries evicted.

        Example:
            >>> selector = ServiceNodeSelector(max_round_robin_entries=100)
            >>> # After many operations...
            >>> evicted = await selector.prune_round_robin_state()
            >>> print(f"Evicted {evicted} stale entries")
        """
        async with self._round_robin_lock:
            return await self._prune_round_robin_state()

    async def select(
        self,
        candidates: list[ModelRegistrationProjection],
        strategy: EnumSelectionStrategy,
        selection_key: str | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelRegistrationProjection | None:
        """Select a node from candidates using the specified strategy.

        Coroutine Safety:
            This method is coroutine-safe when called concurrently from multiple
            coroutines within the same event loop. The asyncio.Lock protects
            round-robin state access.

        Immutability:
            A defensive copy of the candidates list is made at the start of
            selection. This protects against caller modifications during async
            operations and ensures consistent behavior.

        Args:
            candidates: List of nodes matching capability criteria. A defensive
                copy is made internally - the original list is never modified
                or accessed after the copy.
            strategy: Selection strategy to use. Must be one of:
                - FIRST: Return first candidate (deterministic)
                - RANDOM: Random selection (stateless load distribution)
                - ROUND_ROBIN: Sequential cycling (stateful, even distribution)
                - LEAST_LOADED: Not yet implemented (raises RuntimeHostError)
            selection_key: Key for round-robin state tracking. Defaults to "_default".
                Different keys maintain independent round-robin sequences.
                For FIRST and RANDOM strategies, this parameter is ignored.
            correlation_id: Optional correlation ID for distributed tracing.
                When provided, included in all log messages for request tracking.
                Auto-generated as UUID4 if not provided.

        Returns:
            Selected node, or None if candidates is empty.

        Raises:
            RuntimeHostError: If LEAST_LOADED strategy is requested, or if an
                unhandled strategy value is encountered. The latter should never
                happen with properly typed code, as all enum values are explicitly
                handled. The check exists for runtime safety and to ensure
                exhaustive handling if new enum values are added. When raised for
                unhandled strategies, includes full context (correlation_id,
                strategy, selection_key, candidates_count) for debugging.

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
        # Ensure correlation_id is present for distributed tracing
        cid = self._ensure_correlation_id(correlation_id)

        # Defensive copy to protect against caller modifications during async operations
        # This ensures consistent behavior even if the caller modifies the original list
        candidates = list(candidates)

        if not candidates:
            logger.debug(
                "No candidates provided for selection",
                extra={"correlation_id": str(cid)},
            )
            return None

        if len(candidates) == 1:
            logger.debug(
                "Single candidate, returning directly",
                extra={
                    "entity_id": str(candidates[0].entity_id),
                    "correlation_id": str(cid),
                },
            )
            return candidates[0]

        if strategy == EnumSelectionStrategy.FIRST:
            return self._select_first(candidates, cid)
        elif strategy == EnumSelectionStrategy.RANDOM:
            return self._select_random(candidates, cid)
        elif strategy == EnumSelectionStrategy.ROUND_ROBIN:
            return await self._select_round_robin(candidates, selection_key, cid)
        elif strategy == EnumSelectionStrategy.LEAST_LOADED:
            raise RuntimeHostError(
                "LEAST_LOADED selection strategy is not yet implemented",
                context=ModelInfraErrorContext(
                    operation="select",
                    correlation_id=cid,
                ),
                strategy=strategy.value,
                selection_key=selection_key,
                candidates_count=len(candidates),
            )
        else:
            # Type-safe exhaustiveness check: ensures all enum values are handled.
            # If a new EnumSelectionStrategy value is added, the type checker will
            # flag this as an error because strategy won't narrow to Never.
            _: Never = strategy
            raise RuntimeHostError(
                f"Unhandled selection strategy: {strategy.value}",
                context=ModelInfraErrorContext(
                    operation="select",
                    correlation_id=cid,
                ),
                strategy=strategy.value,
                selection_key=selection_key,
                candidates_count=len(candidates),
            )

    def _select_first(
        self,
        candidates: list[ModelRegistrationProjection],
        correlation_id: UUID,
    ) -> ModelRegistrationProjection:
        """Select the first candidate (deterministic).

        Args:
            candidates: Non-empty list of candidates.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            First candidate in the list.
        """
        selected = candidates[0]
        logger.debug(
            "Selected first candidate",
            extra={
                "entity_id": str(selected.entity_id),
                "total_candidates": len(candidates),
                "correlation_id": str(correlation_id),
            },
        )
        return selected

    def _select_random(
        self,
        candidates: list[ModelRegistrationProjection],
        correlation_id: UUID,
    ) -> ModelRegistrationProjection:
        """Select a random candidate.

        Args:
            candidates: Non-empty list of candidates.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Randomly selected candidate.
        """
        selected = random.choice(candidates)
        logger.debug(
            "Selected random candidate",
            extra={
                "entity_id": str(selected.entity_id),
                "total_candidates": len(candidates),
                "correlation_id": str(correlation_id),
            },
        )
        return selected

    async def _select_round_robin(
        self,
        candidates: list[ModelRegistrationProjection],
        selection_key: str | None,
        correlation_id: UUID,
    ) -> ModelRegistrationProjection:
        """Select the next candidate in round-robin sequence.

        State is tracked per selection_key, allowing independent cycling
        for different dependency types. Access is protected by asyncio.Lock.

        Coroutine Safety:
            This method is coroutine-safe when called concurrently from multiple
            coroutines within the same event loop.

        Implements LRU eviction: when state entries exceed max_round_robin_entries,
        the oldest 10% (by last access time) are automatically evicted.

        Note:
            This is an internal method. Callers should use select() which
            makes a defensive copy of the candidates list before calling this.

        Args:
            candidates: Non-empty list of candidates (already copied by select()).
            selection_key: Key for state tracking. Defaults to "_default".
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Next candidate in the round-robin sequence.
        """
        key = selection_key or DEFAULT_SELECTION_KEY

        async with self._round_robin_lock:
            # Get current entry, default to index -1 (so first selection is index 0)
            current_entry = self._round_robin_state.get(key)
            last_index = current_entry.current_index if current_entry else -1
            next_index = (last_index + 1) % len(candidates)

            # Defensive bounds check - ensure index is valid for current candidates list
            # This protects against edge cases where candidates list size changed
            if next_index >= len(candidates):
                next_index = 0
                logger.warning(
                    "Round-robin index out of bounds, resetting to 0",
                    extra={
                        "selection_key": key,
                        "computed_index": (last_index + 1) % len(candidates),
                        "candidates_count": len(candidates),
                        "correlation_id": str(correlation_id),
                    },
                )

            # Update state with new entry (index + current timestamp for LRU)
            self._round_robin_state[key] = RoundRobinEntry(
                current_index=next_index,
                last_access=time.monotonic(),
            )

            # Trigger LRU eviction if needed (while holding lock)
            await self._prune_round_robin_state()

            # Access candidate inside lock for transaction safety
            selected = candidates[next_index]
        logger.debug(
            "Selected round-robin candidate",
            extra={
                "entity_id": str(selected.entity_id),
                "selection_key": key,
                "index": next_index,
                "total_candidates": len(candidates),
                "correlation_id": str(correlation_id),
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
        """Get a copy of the current round-robin state (indices only).

        Access is protected by asyncio.Lock.

        Returns:
            Dictionary mapping selection keys to their last used index.
            Note: Last access timestamps are not included for API simplicity.

        Example:
            >>> selector = ServiceNodeSelector()
            >>> state = await selector.get_round_robin_state()
            >>> print(state)
            {'db': 2, 'consul': 0}
        """
        async with self._round_robin_lock:
            return {
                key: entry.current_index
                for key, entry in self._round_robin_state.items()
            }

    async def get_round_robin_state_full(self) -> dict[str, RoundRobinEntry]:
        """Get a copy of the current round-robin state with full details.

        Access is protected by asyncio.Lock.

        Returns:
            Dictionary mapping selection keys to RoundRobinEntry objects
            containing both index and last access timestamp.

        Example:
            >>> selector = ServiceNodeSelector()
            >>> state = await selector.get_round_robin_state_full()
            >>> for key, entry in state.items():
            ...     print(f"{key}: idx={entry.current_index}, access={entry.last_access}")
        """
        async with self._round_robin_lock:
            return dict(self._round_robin_state)


__all__: list[str] = [
    "DEFAULT_MAX_ROUND_ROBIN_ENTRIES",
    "DEFAULT_SELECTION_KEY",
    "EVICTION_PERCENTAGE",
    "RoundRobinEntry",
    "ServiceNodeSelector",
]
