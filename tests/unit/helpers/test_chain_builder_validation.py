# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for ChainBuilder argument validation.

This module tests that the ChainBuilder class properly validates its
arguments, ensuring that invalid inputs are rejected with clear error
messages.

Related:
    - tests/helpers/chaos_utils.py: ChainBuilder implementation
    - OMN-955: Chaos scenario tests
"""

from __future__ import annotations

import pytest

from tests.helpers.chaos_utils import ChainBuilder

# =============================================================================
# Length Validation Tests
# =============================================================================


@pytest.mark.unit
class TestChainBuilderLengthValidation:
    """Tests for chain length validation across all build methods."""

    @pytest.fixture
    def builder(self) -> ChainBuilder:
        """Create a ChainBuilder instance for testing."""
        return ChainBuilder(seed=42)

    def test_build_valid_chain_rejects_zero_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that build_valid_chain rejects length=0."""
        with pytest.raises(ValueError, match="Chain length must be positive"):
            builder.build_valid_chain(length=0)

    def test_build_valid_chain_rejects_negative_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that build_valid_chain rejects negative length."""
        with pytest.raises(ValueError, match="Chain length must be positive"):
            builder.build_valid_chain(length=-1)

    def test_build_valid_chain_accepts_positive_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that build_valid_chain accepts positive length."""
        chain = builder.build_valid_chain(length=3)
        assert len(chain) == 3

    def test_build_correlation_break_rejects_zero_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that build_chain_with_correlation_break rejects length=0."""
        with pytest.raises(ValueError, match="Chain length must be positive"):
            builder.build_chain_with_correlation_break(length=0, break_at=0)

    def test_build_causation_break_rejects_zero_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that build_chain_with_causation_break rejects length=0."""
        with pytest.raises(ValueError, match="Chain length must be positive"):
            builder.build_chain_with_causation_break(length=0, break_at=1)

    def test_build_random_chaos_rejects_zero_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that build_chain_with_random_chaos rejects length=0."""
        with pytest.raises(ValueError, match="Chain length must be positive"):
            builder.build_chain_with_random_chaos(length=0)

    def test_build_random_chaos_rejects_negative_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that build_chain_with_random_chaos rejects negative length."""
        with pytest.raises(ValueError, match="Chain length must be positive"):
            builder.build_chain_with_random_chaos(length=-5)


# =============================================================================
# break_at Validation Tests for Correlation Break
# =============================================================================


@pytest.mark.unit
class TestCorrelationBreakAtValidation:
    """Tests for break_at validation in correlation break chains."""

    @pytest.fixture
    def builder(self) -> ChainBuilder:
        """Create a ChainBuilder instance for testing."""
        return ChainBuilder(seed=42)

    def test_rejects_negative_break_at(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that negative break_at is rejected."""
        with pytest.raises(ValueError, match="break_at must be non-negative"):
            builder.build_chain_with_correlation_break(length=5, break_at=-1)

    def test_rejects_break_at_equal_to_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that break_at == length is rejected."""
        with pytest.raises(ValueError, match=r"break_at.*must be less than length"):
            builder.build_chain_with_correlation_break(length=5, break_at=5)

    def test_rejects_break_at_greater_than_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that break_at > length is rejected."""
        with pytest.raises(ValueError, match=r"break_at.*must be less than length"):
            builder.build_chain_with_correlation_break(length=5, break_at=10)

    def test_accepts_break_at_zero(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that break_at=0 is accepted (all messages have broken correlation)."""
        chain = builder.build_chain_with_correlation_break(length=5, break_at=0)
        assert len(chain) == 5
        # All messages should have the broken correlation_id
        first_correlation = chain[0].correlation_id
        for msg in chain:
            assert msg.correlation_id == first_correlation

    def test_accepts_valid_break_at(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that valid break_at positions are accepted."""
        chain = builder.build_chain_with_correlation_break(length=5, break_at=3)
        assert len(chain) == 5
        # First 3 messages should have original correlation
        # Last 2 should have different correlation
        original_correlation = chain[0].correlation_id
        broken_correlation = chain[3].correlation_id
        assert original_correlation != broken_correlation
        for i, msg in enumerate(chain):
            if i < 3:
                assert msg.correlation_id == original_correlation
            else:
                assert msg.correlation_id == broken_correlation


# =============================================================================
# break_at Validation Tests for Causation Break
# =============================================================================


@pytest.mark.unit
class TestCausationBreakAtValidation:
    """Tests for break_at validation in causation break chains."""

    @pytest.fixture
    def builder(self) -> ChainBuilder:
        """Create a ChainBuilder instance for testing."""
        return ChainBuilder(seed=42)

    def test_rejects_negative_break_at(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that negative break_at is rejected."""
        with pytest.raises(ValueError, match="break_at must be positive"):
            builder.build_chain_with_causation_break(length=5, break_at=-1)

    def test_rejects_break_at_zero(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that break_at=0 is rejected (root has no causation to break)."""
        with pytest.raises(ValueError, match="Cannot break causation at root"):
            builder.build_chain_with_causation_break(length=5, break_at=0)

    def test_rejects_break_at_equal_to_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that break_at == length is rejected."""
        with pytest.raises(ValueError, match=r"break_at.*must be less than length"):
            builder.build_chain_with_causation_break(length=5, break_at=5)

    def test_rejects_break_at_greater_than_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that break_at > length is rejected."""
        with pytest.raises(ValueError, match=r"break_at.*must be less than length"):
            builder.build_chain_with_causation_break(length=5, break_at=10)

    def test_accepts_break_at_one(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that break_at=1 is accepted (minimum valid position)."""
        chain = builder.build_chain_with_causation_break(length=5, break_at=1)
        assert len(chain) == 5
        # Message at position 1 should NOT have parent's message_id as causation
        assert chain[1].causation_id != chain[0].message_id

    def test_accepts_valid_break_at(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that valid break_at positions are accepted."""
        chain = builder.build_chain_with_causation_break(length=5, break_at=3)
        assert len(chain) == 5
        # Check causation chain is broken only at position 3
        assert chain[0].causation_id is None  # Root has no causation
        assert chain[1].causation_id == chain[0].message_id  # Valid
        assert chain[2].causation_id == chain[1].message_id  # Valid
        assert chain[3].causation_id != chain[2].message_id  # Broken
        assert chain[4].causation_id == chain[3].message_id  # Valid


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.unit
class TestChainBuilderEdgeCases:
    """Tests for edge cases in chain building."""

    @pytest.fixture
    def builder(self) -> ChainBuilder:
        """Create a ChainBuilder instance for testing."""
        return ChainBuilder(seed=42)

    def test_single_message_chain(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that a single-message chain is valid."""
        chain = builder.build_valid_chain(length=1)
        assert len(chain) == 1
        assert chain[0].causation_id is None  # Root has no causation
        assert chain[0].position == 0

    def test_correlation_break_at_zero_with_length_one(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test correlation break at 0 with single message chain."""
        chain = builder.build_chain_with_correlation_break(length=1, break_at=0)
        assert len(chain) == 1

    def test_causation_break_minimum_length(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test causation break requires at least length=2 for break_at=1."""
        chain = builder.build_chain_with_causation_break(length=2, break_at=1)
        assert len(chain) == 2
        # Position 1's causation should be broken
        assert chain[1].causation_id != chain[0].message_id

    def test_error_messages_are_descriptive(
        self,
        builder: ChainBuilder,
    ) -> None:
        """Test that error messages include actual values for debugging."""
        try:
            builder.build_chain_with_correlation_break(length=5, break_at=10)
        except ValueError as e:
            # Error message should include both values
            assert "10" in str(e)
            assert "5" in str(e)
