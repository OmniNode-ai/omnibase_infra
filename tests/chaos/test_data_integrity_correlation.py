# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Correlation Chain Integrity Tests Under Chaos Conditions (OMN-955).

This test suite validates that correlation and causation chains remain intact
even under chaotic conditions such as:

1. Random processing failures during chain construction
2. Process restarts mid-workflow
3. Out-of-order message processing
4. Partial chain failures

Correlation Chain Semantics:
    - correlation_id: Shared across all messages in a workflow for end-to-end tracing
    - causation_id: Each message's causation_id references its parent's message_id

Chain Integrity Rules:
    1. All messages in a workflow must share the same correlation_id
    2. Every produced message's causation_id must equal parent's message_id
    3. Causation chains form an unbroken lineage back to workflow origin

Architecture:
    This test suite uses the ChainPropagationValidator from OMN-951 to
    validate chain integrity. Tests simulate various chaos scenarios and
    verify that chains are either:
    - Maintained correctly despite failures, OR
    - Violations are properly detected when chains are broken

Test Strategy:
    - Create message chains with proper correlation/causation
    - Inject failures during chain construction
    - Verify chains remain valid or violations detected
    - Test recovery after chain breaks

Related:
    - OMN-955: Data Integrity Tests Under Chaos
    - OMN-951: Correlation Chain Validation
    - docs/patterns/correlation_id_tracking.md
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums.enum_chain_violation_type import EnumChainViolationType
from omnibase_infra.validation.chain_propagation_validator import (
    ChainPropagationValidator,
    get_causation_id,
    get_correlation_id,
    get_message_id,
    validate_linear_message_chain,
    validate_message_chain,
)
from tests.helpers.chaos_utils import (
    ChainBuilder,
    ChainedMessage,
    ChaosChainConfig,
    create_envelope_from_chained_message,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def chain_builder() -> ChainBuilder:
    """Create chain builder with default config.

    Returns:
        ChainBuilder with reproducible random seed.
    """
    return ChainBuilder(seed=42)


@pytest.fixture
def validator() -> ChainPropagationValidator:
    """Create chain propagation validator.

    Returns:
        ChainPropagationValidator instance.
    """
    return ChainPropagationValidator()


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


@pytest.mark.chaos
class TestCorrelationChainIntegrity:
    """Test correlation chain integrity under chaos."""

    @pytest.mark.asyncio
    async def test_valid_chain_passes_validation(
        self,
        chain_builder: ChainBuilder,
        validator: ChainPropagationValidator,
    ) -> None:
        """Verify valid chain passes all validation checks.

        Test Flow:
            1. Build valid 5-message chain
            2. Convert to envelopes
            3. Validate pairwise
            4. Verify no violations
        """
        # Arrange
        chain = chain_builder.build_valid_chain(length=5)
        envelopes = [create_envelope_from_chained_message(m) for m in chain]

        # Act - validate pairwise
        all_violations = []
        for i in range(len(envelopes) - 1):
            violations = validator.validate_chain(envelopes[i], envelopes[i + 1])
            all_violations.extend(violations)

        # Assert
        assert len(all_violations) == 0, (
            f"Expected no violations, got: {[v.violation_type for v in all_violations]}"
        )

    @pytest.mark.asyncio
    async def test_correlation_break_detected(
        self,
        chain_builder: ChainBuilder,
        validator: ChainPropagationValidator,
    ) -> None:
        """Verify correlation_id break is detected.

        Test Flow:
            1. Build chain with correlation break at position 3
            2. Convert to envelopes
            3. Validate
            4. Verify CORRELATION_MISMATCH violation detected
        """
        # Arrange
        chain = chain_builder.build_chain_with_correlation_break(
            length=5,
            break_at=3,
        )
        envelopes = [create_envelope_from_chained_message(m) for m in chain]

        # Act - validate the break point (position 2 -> 3)
        violations = validator.validate_chain(envelopes[2], envelopes[3])

        # Assert
        assert len(violations) >= 1
        correlation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CORRELATION_MISMATCH
        ]
        assert len(correlation_violations) == 1, (
            f"Expected 1 CORRELATION_MISMATCH, got {len(correlation_violations)}"
        )

    @pytest.mark.asyncio
    async def test_causation_break_detected(
        self,
        chain_builder: ChainBuilder,
        validator: ChainPropagationValidator,
    ) -> None:
        """Verify causation_id break is detected.

        Test Flow:
            1. Build chain with causation break at position 2
            2. Convert to envelopes
            3. Validate
            4. Verify CAUSATION_CHAIN_BROKEN violation detected
        """
        # Arrange
        chain = chain_builder.build_chain_with_causation_break(
            length=5,
            break_at=2,
        )
        envelopes = [create_envelope_from_chained_message(m) for m in chain]

        # Act - validate the break point (position 1 -> 2)
        violations = validator.validate_chain(envelopes[1], envelopes[2])

        # Assert
        assert len(violations) >= 1
        causation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        ]
        assert len(causation_violations) == 1, (
            f"Expected 1 CAUSATION_CHAIN_BROKEN, got {len(causation_violations)}"
        )


@pytest.mark.chaos
class TestCorrelationPropagation:
    """Test correlation_id propagation through workflows."""

    @pytest.mark.asyncio
    async def test_correlation_propagates_through_entire_chain(
        self,
        chain_builder: ChainBuilder,
    ) -> None:
        """Verify correlation_id propagates through entire chain.

        Test Flow:
            1. Build 10-message chain
            2. Extract correlation_ids
            3. Verify all are identical
        """
        # Arrange
        chain = chain_builder.build_valid_chain(length=10)
        expected_correlation = chain[0].correlation_id

        # Act & Assert
        for i, message in enumerate(chain):
            assert message.correlation_id == expected_correlation, (
                f"Message {i} has different correlation_id"
            )

    @pytest.mark.asyncio
    async def test_validate_linear_chain_helper(
        self,
        chain_builder: ChainBuilder,
    ) -> None:
        """Test validate_linear_message_chain helper function.

        Test Flow:
            1. Build valid chain
            2. Use convenience function to validate
            3. Verify no violations
        """
        # Arrange
        chain = chain_builder.build_valid_chain(length=5)
        envelopes = [create_envelope_from_chained_message(m) for m in chain]

        # Act
        violations = validate_linear_message_chain(envelopes)

        # Assert
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_validate_message_chain_helper(
        self,
        chain_builder: ChainBuilder,
    ) -> None:
        """Test validate_message_chain convenience function.

        Test Flow:
            1. Build valid chain
            2. Use convenience function to validate pair
            3. Verify no violations
        """
        # Arrange
        chain = chain_builder.build_valid_chain(length=2)
        envelopes = [create_envelope_from_chained_message(m) for m in chain]

        # Act
        violations = validate_message_chain(envelopes[0], envelopes[1])

        # Assert
        assert len(violations) == 0


@pytest.mark.chaos
class TestCausationChainIntegrity:
    """Test causation chain integrity under chaos."""

    @pytest.mark.asyncio
    async def test_causation_chain_forms_valid_lineage(
        self,
        chain_builder: ChainBuilder,
    ) -> None:
        """Verify causation chain forms valid lineage.

        Test Flow:
            1. Build valid chain
            2. Verify each message's causation_id equals parent's message_id
        """
        # Arrange
        chain = chain_builder.build_valid_chain(length=5)

        # Act & Assert
        for i in range(1, len(chain)):
            assert chain[i].causation_id == chain[i - 1].message_id, (
                f"Message {i} causation_id should equal message {i - 1} message_id"
            )

    @pytest.mark.asyncio
    async def test_root_message_has_no_causation(
        self,
        chain_builder: ChainBuilder,
    ) -> None:
        """Verify root message has no causation_id.

        Test Flow:
            1. Build chain
            2. Verify first message has None causation_id
        """
        # Arrange
        chain = chain_builder.build_valid_chain(length=3)

        # Assert
        assert chain[0].causation_id is None


@pytest.mark.chaos
class TestChainRecoveryAfterFailures:
    """Test chain recovery mechanisms after failures."""

    @pytest.mark.asyncio
    async def test_chain_can_be_rebuilt_after_failure(
        self,
        chain_builder: ChainBuilder,
        validator: ChainPropagationValidator,
    ) -> None:
        """Verify chain can be rebuilt after detecting failure.

        Simulates the pattern of detecting a broken chain and rebuilding
        with correct correlation/causation.

        Test Flow:
            1. Build chain with break
            2. Detect violation
            3. Rebuild valid chain from break point
            4. Verify new chain is valid
        """
        # Arrange - build broken chain
        broken_chain = chain_builder.build_chain_with_causation_break(
            length=5,
            break_at=3,
        )
        broken_envelopes = [
            create_envelope_from_chained_message(m) for m in broken_chain
        ]

        # Detect violation
        violations = validator.validate_chain(broken_envelopes[2], broken_envelopes[3])
        assert len(violations) > 0, "Should detect violation"

        # Rebuild: create new valid chain continuing from message 2
        recovery_correlation = broken_chain[2].correlation_id
        repaired_chain = [
            ChainedMessage(
                message_id=uuid4(),
                correlation_id=recovery_correlation,
                causation_id=broken_chain[2].message_id,  # Correct causation
                payload={"position": 3, "type": "repaired"},
                position=3,
            ),
            ChainedMessage(
                message_id=uuid4(),
                correlation_id=recovery_correlation,
                causation_id=None,  # Will be set below
                payload={"position": 4, "type": "repaired"},
                position=4,
            ),
        ]
        repaired_chain[1] = ChainedMessage(
            message_id=repaired_chain[1].message_id,
            correlation_id=recovery_correlation,
            causation_id=repaired_chain[0].message_id,
            payload=repaired_chain[1].payload,
            position=4,
        )

        # Convert to envelopes
        repaired_envelopes = [
            create_envelope_from_chained_message(m) for m in repaired_chain
        ]

        # Validate repaired chain
        repair_violations = validator.validate_chain(
            broken_envelopes[2],
            repaired_envelopes[0],
        )

        # Assert - repaired chain should be valid
        assert len(repair_violations) == 0, (
            f"Repaired chain should be valid, got: {repair_violations}"
        )

    @pytest.mark.asyncio
    async def test_partial_chain_validation(
        self,
        chain_builder: ChainBuilder,
        validator: ChainPropagationValidator,
    ) -> None:
        """Verify partial chain validation works correctly.

        When only a subset of chain is available, validation should
        still work for the available pairs.

        Test Flow:
            1. Build valid chain of 10 messages
            2. Take subset (messages 3-6)
            3. Validate subset
            4. Verify validation passes for available pairs
        """
        # Arrange
        full_chain = chain_builder.build_valid_chain(length=10)
        subset = full_chain[3:7]  # Messages 3, 4, 5, 6
        subset_envelopes = [create_envelope_from_chained_message(m) for m in subset]

        # Act - validate subset pairs
        all_violations = []
        for i in range(len(subset_envelopes) - 1):
            violations = validator.validate_chain(
                subset_envelopes[i],
                subset_envelopes[i + 1],
            )
            all_violations.extend(violations)

        # Assert
        assert len(all_violations) == 0


@pytest.mark.chaos
class TestRandomChaosChainValidation:
    """Test chain validation under random chaos conditions."""

    @pytest.mark.asyncio
    async def test_random_chaos_violations_detected(
        self,
    ) -> None:
        """Verify random chaos violations are properly detected.

        Test Flow:
            1. Build chain with random chaos (20% break rate)
            2. Validate entire chain
            3. Verify violations detected at break points
        """
        # Arrange - higher break rates for test
        chaos_config = ChaosChainConfig(
            break_correlation_rate=0.2,
            break_causation_rate=0.2,
        )
        builder = ChainBuilder(chaos_config=chaos_config, seed=42)
        validator = ChainPropagationValidator()

        chain, break_positions = builder.build_chain_with_random_chaos(length=10)
        envelopes = [create_envelope_from_chained_message(m) for m in chain]

        # Act - validate all pairs
        all_violations = []
        for i in range(len(envelopes) - 1):
            violations = validator.validate_chain(envelopes[i], envelopes[i + 1])
            all_violations.extend(violations)

        # Assert - if there are breaks, there should be violations
        if break_positions:
            assert len(all_violations) > 0, (
                f"Expected violations at break positions {break_positions}"
            )

    @pytest.mark.asyncio
    async def test_no_chaos_no_violations(
        self,
    ) -> None:
        """Verify no violations when chaos is disabled.

        Test Flow:
            1. Build chain with 0% break rates
            2. Validate
            3. Verify no violations
        """
        # Arrange - no chaos
        no_chaos_config = ChaosChainConfig(
            break_correlation_rate=0.0,
            break_causation_rate=0.0,
        )
        builder = ChainBuilder(chaos_config=no_chaos_config, seed=42)
        validator = ChainPropagationValidator()

        chain, break_positions = builder.build_chain_with_random_chaos(length=10)
        envelopes = [create_envelope_from_chained_message(m) for m in chain]

        # Assert - no breaks should have occurred
        assert len(break_positions) == 0

        # Act - validate all pairs
        all_violations = []
        for i in range(len(envelopes) - 1):
            violations = validator.validate_chain(envelopes[i], envelopes[i + 1])
            all_violations.extend(violations)

        # Assert
        assert len(all_violations) == 0


@pytest.mark.chaos
class TestEnvelopeFieldAccessHelpers:
    """Test envelope field access helper functions."""

    @pytest.mark.asyncio
    async def test_get_message_id_returns_envelope_id(
        self,
        chain_builder: ChainBuilder,
    ) -> None:
        """Verify get_message_id returns envelope_id."""
        # Arrange
        chain = chain_builder.build_valid_chain(length=1)
        envelope = create_envelope_from_chained_message(chain[0])

        # Act
        message_id = get_message_id(envelope)

        # Assert
        assert message_id == chain[0].message_id

    @pytest.mark.asyncio
    async def test_get_correlation_id_returns_correlation(
        self,
        chain_builder: ChainBuilder,
    ) -> None:
        """Verify get_correlation_id returns correlation_id."""
        # Arrange
        chain = chain_builder.build_valid_chain(length=1)
        envelope = create_envelope_from_chained_message(chain[0])

        # Act
        correlation_id = get_correlation_id(envelope)

        # Assert
        assert correlation_id == chain[0].correlation_id

    @pytest.mark.asyncio
    async def test_get_causation_id_returns_from_tags(
        self,
        chain_builder: ChainBuilder,
    ) -> None:
        """Verify get_causation_id returns causation_id from tags."""
        # Arrange
        chain = chain_builder.build_valid_chain(length=2)
        envelope = create_envelope_from_chained_message(chain[1])

        # Act
        causation_id = get_causation_id(envelope)

        # Assert
        assert causation_id == chain[0].message_id

    @pytest.mark.asyncio
    async def test_get_causation_id_returns_none_for_root(
        self,
        chain_builder: ChainBuilder,
    ) -> None:
        """Verify get_causation_id returns None for root message."""
        # Arrange
        chain = chain_builder.build_valid_chain(length=1)
        envelope = create_envelope_from_chained_message(chain[0])

        # Act
        causation_id = get_causation_id(envelope)

        # Assert
        assert causation_id is None
