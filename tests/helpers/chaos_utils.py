# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chaos testing utilities for OMN-955.

This module provides shared utilities for chaos testing scenarios including:
- Message chain builders for correlation/causation testing
- Envelope creation helpers
- Chaos configuration builders

These utilities are extracted from chaos tests to reduce duplication and
provide a consistent interface for chaos testing across the test suite.

Example usage:
    >>> from tests.helpers.chaos_utils import ChainBuilder, ChaosChainConfig
    >>>
    >>> # Build a valid message chain
    >>> builder = ChainBuilder(seed=42)
    >>> chain = builder.build_valid_chain(length=5)
    >>>
    >>> # Build a chain with correlation break
    >>> broken_chain = builder.build_chain_with_correlation_break(length=5, break_at=3)

Related Tickets:
    - OMN-955: Chaos scenario tests
    - OMN-951: Correlation Chain Validation
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

from omnibase_infra.validation.chain_propagation_validator import (
    ChainPropagationValidator,
)

# =============================================================================
# Configuration Models
# =============================================================================


@dataclass
class ChaosChainConfig:
    """Configuration for chaos injection in chain tests.

    Controls the probability of breaking correlation and causation chains
    during chain construction for chaos testing scenarios.

    Attributes:
        failure_rate: Probability of failure (0.0 to 1.0).
        max_chain_length: Maximum length of message chains to test.
        break_correlation_rate: Rate at which to break correlation_id.
        break_causation_rate: Rate at which to break causation_id.
    """

    failure_rate: float = 0.2
    max_chain_length: int = 10
    break_correlation_rate: float = 0.1
    break_causation_rate: float = 0.1


# =============================================================================
# Message Models
# =============================================================================


@dataclass
class ChainedMessage:
    """A message in a chain with correlation and causation tracking.

    Represents a single message in a message chain, capturing the
    correlation and causation relationships that enable distributed
    tracing and workflow validation.

    Attributes:
        message_id: Unique identifier for this message.
        correlation_id: Shared workflow correlation ID (same for all messages in workflow).
        causation_id: Parent message ID (None for root message).
        payload: Message payload data.
        position: Position in the chain (0 = root).
    """

    message_id: UUID
    correlation_id: UUID
    causation_id: UUID | None
    payload: dict[str, str | int]
    position: int


# =============================================================================
# Envelope Creation Helpers
# =============================================================================


def create_envelope_from_chained_message(
    message: ChainedMessage,
) -> ModelEventEnvelope[dict[str, str | int]]:
    """Create ModelEventEnvelope from ChainedMessage.

    Converts a ChainedMessage dataclass to a proper ModelEventEnvelope
    with correct correlation and causation metadata.

    Args:
        message: ChainedMessage to convert.

    Returns:
        ModelEventEnvelope with proper correlation and causation fields.
    """
    tags: dict[str, str] = {}
    if message.causation_id is not None:
        tags["causation_id"] = str(message.causation_id)

    metadata = ModelEnvelopeMetadata(
        tags=tags,
    )

    return ModelEventEnvelope(
        envelope_id=message.message_id,
        correlation_id=message.correlation_id,
        payload=message.payload,
        metadata=metadata,
    )


# =============================================================================
# Chain Builder
# =============================================================================


class ChainBuilder:
    """Builder for creating message chains with optional chaos injection.

    Creates chains of messages with proper correlation and causation,
    optionally injecting chaos (broken chains, missing IDs, etc.).
    This enables testing of chain validation under various failure scenarios.

    Attributes:
        chaos_config: Configuration for chaos injection.
        rng: Random number generator for reproducibility.
        validator: Chain validation instance.

    Example:
        >>> builder = ChainBuilder(seed=42)
        >>> # Build a valid chain
        >>> valid_chain = builder.build_valid_chain(length=5)
        >>> # Build a chain with broken correlation at position 3
        >>> broken_chain = builder.build_chain_with_correlation_break(length=5, break_at=3)
    """

    def __init__(
        self,
        chaos_config: ChaosChainConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize chain builder.

        Args:
            chaos_config: Optional chaos configuration for injection rates.
            seed: Random seed for reproducibility (recommended for tests).
        """
        self.chaos_config = chaos_config or ChaosChainConfig()
        self.rng = random.Random(seed)
        self.validator = ChainPropagationValidator()

    def build_valid_chain(
        self,
        length: int,
        correlation_id: UUID | None = None,
    ) -> list[ChainedMessage]:
        """Build a valid message chain with proper propagation.

        Creates a chain where:
        - All messages share the same correlation_id
        - Each message's causation_id equals the previous message's message_id
        - The root message (position 0) has no causation_id

        Args:
            length: Number of messages in chain.
            correlation_id: Optional correlation ID (generates if None).

        Returns:
            List of ChainedMessage forming a valid chain.
        """
        correlation_id = correlation_id or uuid4()
        chain: list[ChainedMessage] = []

        for i in range(length):
            message = ChainedMessage(
                message_id=uuid4(),
                correlation_id=correlation_id,
                causation_id=chain[i - 1].message_id if i > 0 else None,
                payload={"position": i, "type": "chain_message"},
                position=i,
            )
            chain.append(message)

        return chain

    def build_chain_with_correlation_break(
        self,
        length: int,
        break_at: int,
    ) -> list[ChainedMessage]:
        """Build a chain with correlation_id break at specified position.

        Creates a chain where the correlation_id changes at the specified
        position, simulating a correlation break that should be detected
        by chain validation.

        Args:
            length: Number of messages in chain.
            break_at: Position where correlation breaks (messages at this
                position and later will have a different correlation_id).

        Returns:
            List of ChainedMessage with broken correlation at break_at.
        """
        correlation_id = uuid4()
        broken_correlation_id = uuid4()
        chain: list[ChainedMessage] = []

        for i in range(length):
            # Break correlation at specified position
            current_correlation = (
                broken_correlation_id if i >= break_at else correlation_id
            )

            message = ChainedMessage(
                message_id=uuid4(),
                correlation_id=current_correlation,
                causation_id=chain[i - 1].message_id if i > 0 else None,
                payload={"position": i, "type": "chain_message"},
                position=i,
            )
            chain.append(message)

        return chain

    def build_chain_with_causation_break(
        self,
        length: int,
        break_at: int,
    ) -> list[ChainedMessage]:
        """Build a chain with causation_id break at specified position.

        Creates a chain where the causation_id at the specified position
        points to a random UUID instead of the parent's message_id,
        simulating a causation break that should be detected by validation.

        Args:
            length: Number of messages in chain.
            break_at: Position where causation breaks (this message will
                have a random causation_id instead of parent's message_id).

        Returns:
            List of ChainedMessage with broken causation at break_at.
        """
        correlation_id = uuid4()
        chain: list[ChainedMessage] = []

        for i in range(length):
            if i == 0:
                causation_id = None
            elif i == break_at:
                # Break causation by using wrong parent ID
                causation_id = uuid4()  # Random ID instead of parent
            else:
                causation_id = chain[i - 1].message_id

            message = ChainedMessage(
                message_id=uuid4(),
                correlation_id=correlation_id,
                causation_id=causation_id,
                payload={"position": i, "type": "chain_message"},
                position=i,
            )
            chain.append(message)

        return chain

    def build_chain_with_random_chaos(
        self,
        length: int,
    ) -> tuple[list[ChainedMessage], list[int]]:
        """Build a chain with random chaos injection.

        Randomly breaks correlation and/or causation based on config rates.
        Useful for fuzz testing chain validation.

        Args:
            length: Number of messages in chain.

        Returns:
            Tuple of (chain, list of break positions) where break positions
            indicates which messages have broken correlation or causation.
        """
        correlation_id = uuid4()
        chain: list[ChainedMessage] = []
        break_positions: list[int] = []

        for i in range(length):
            current_correlation = correlation_id
            causation_id = chain[i - 1].message_id if i > 0 else None

            # Randomly break correlation
            if i > 0 and self.rng.random() < self.chaos_config.break_correlation_rate:
                current_correlation = uuid4()
                break_positions.append(i)

            # Randomly break causation
            if i > 0 and self.rng.random() < self.chaos_config.break_causation_rate:
                causation_id = uuid4()
                if i not in break_positions:
                    break_positions.append(i)

            message = ChainedMessage(
                message_id=uuid4(),
                correlation_id=current_correlation,
                causation_id=causation_id,
                payload={"position": i, "type": "chain_message"},
                position=i,
            )
            chain.append(message)

        return chain, sorted(break_positions)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ChaosChainConfig",
    "ChainedMessage",
    "create_envelope_from_chained_message",
    "ChainBuilder",
]
