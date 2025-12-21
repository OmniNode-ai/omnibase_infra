# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelSequenceInfo.

Tests validate:
- Model instantiation with valid data
- Field validation (types, constraints)
- Default values for optional fields
- Staleness comparison logic
- Factory methods: from_kafka(), from_sequence()
- Immutability (frozen model)

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.projection import ModelSequenceInfo


class TestModelSequenceInfoInstantiation:
    """Tests for model instantiation with valid data."""

    def test_minimal_instantiation(self) -> None:
        """Test instantiation with only required sequence field."""
        seq = ModelSequenceInfo(sequence=100)
        assert seq.sequence == 100
        assert seq.partition is None
        assert seq.offset is None

    def test_full_instantiation(self) -> None:
        """Test instantiation with all fields."""
        seq = ModelSequenceInfo(
            sequence=1000,
            partition="0",
            offset=1000,
        )
        assert seq.sequence == 1000
        assert seq.partition == "0"
        assert seq.offset == 1000

    def test_kafka_style_instantiation(self) -> None:
        """Test instantiation mimicking Kafka message metadata."""
        seq = ModelSequenceInfo(
            sequence=12345,
            partition="3",
            offset=12345,
        )
        assert seq.sequence == 12345
        assert seq.partition == "3"
        assert seq.offset == 12345

    def test_generic_transport_instantiation(self) -> None:
        """Test instantiation for generic (non-Kafka) transports."""
        seq = ModelSequenceInfo(sequence=42)
        assert seq.sequence == 42
        assert seq.partition is None
        assert seq.offset is None


class TestModelSequenceInfoFieldValidation:
    """Tests for field validation and constraints."""

    def test_sequence_required(self) -> None:
        """Test that sequence is a required field."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSequenceInfo()  # type: ignore[call-arg]
        assert "sequence" in str(exc_info.value)

    def test_sequence_must_be_non_negative(self) -> None:
        """Test that sequence must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSequenceInfo(sequence=-1)
        assert "sequence" in str(exc_info.value)

    def test_sequence_zero_is_valid(self) -> None:
        """Test that sequence=0 is valid."""
        seq = ModelSequenceInfo(sequence=0)
        assert seq.sequence == 0

    def test_sequence_large_value(self) -> None:
        """Test that large sequence values are valid."""
        seq = ModelSequenceInfo(sequence=9999999999)
        assert seq.sequence == 9999999999

    def test_offset_must_be_non_negative(self) -> None:
        """Test that offset must be >= 0 if provided."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSequenceInfo(sequence=100, offset=-1)
        assert "offset" in str(exc_info.value)

    def test_offset_zero_is_valid(self) -> None:
        """Test that offset=0 is valid."""
        seq = ModelSequenceInfo(sequence=0, offset=0)
        assert seq.offset == 0

    def test_partition_string_type(self) -> None:
        """Test that partition must be a string."""
        seq = ModelSequenceInfo(sequence=100, partition="5")
        assert seq.partition == "5"

    def test_partition_empty_string_valid(self) -> None:
        """Test that empty partition string is valid."""
        seq = ModelSequenceInfo(sequence=100, partition="")
        assert seq.partition == ""

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSequenceInfo(sequence=100, unknown_field="value")  # type: ignore[call-arg]
        assert (
            "unknown_field" in str(exc_info.value).lower()
            or "extra" in str(exc_info.value).lower()
        )


class TestModelSequenceInfoImmutability:
    """Tests for model immutability (frozen=True)."""

    def test_sequence_is_immutable(self) -> None:
        """Test that sequence cannot be modified after creation."""
        seq = ModelSequenceInfo(sequence=100)
        with pytest.raises(ValidationError):
            seq.sequence = 200  # type: ignore[misc]

    def test_partition_is_immutable(self) -> None:
        """Test that partition cannot be modified after creation."""
        seq = ModelSequenceInfo(sequence=100, partition="0")
        with pytest.raises(ValidationError):
            seq.partition = "1"  # type: ignore[misc]

    def test_offset_is_immutable(self) -> None:
        """Test that offset cannot be modified after creation."""
        seq = ModelSequenceInfo(sequence=100, offset=100)
        with pytest.raises(ValidationError):
            seq.offset = 200  # type: ignore[misc]


class TestIsStaleComparedTo:
    """Tests for is_stale_compared_to() method."""

    def test_lower_sequence_is_stale(self) -> None:
        """Test that lower sequence is considered stale."""
        old = ModelSequenceInfo(sequence=10)
        new = ModelSequenceInfo(sequence=20)
        assert old.is_stale_compared_to(new) is True

    def test_higher_sequence_is_not_stale(self) -> None:
        """Test that higher sequence is not stale."""
        old = ModelSequenceInfo(sequence=10)
        new = ModelSequenceInfo(sequence=20)
        assert new.is_stale_compared_to(old) is False

    def test_same_sequence_is_not_stale(self) -> None:
        """Test that same sequence is not stale (no offset tiebreaker)."""
        seq1 = ModelSequenceInfo(sequence=10)
        seq2 = ModelSequenceInfo(sequence=10)
        assert seq1.is_stale_compared_to(seq2) is False

    def test_same_sequence_lower_offset_is_stale(self) -> None:
        """Test offset tiebreaker: lower offset is stale."""
        old = ModelSequenceInfo(sequence=10, partition="0", offset=50)
        new = ModelSequenceInfo(sequence=10, partition="0", offset=100)
        assert old.is_stale_compared_to(new) is True

    def test_same_sequence_higher_offset_is_not_stale(self) -> None:
        """Test offset tiebreaker: higher offset is not stale."""
        old = ModelSequenceInfo(sequence=10, partition="0", offset=50)
        new = ModelSequenceInfo(sequence=10, partition="0", offset=100)
        assert new.is_stale_compared_to(old) is False

    def test_same_sequence_same_offset_is_not_stale(self) -> None:
        """Test that identical sequences with same offset are not stale."""
        seq1 = ModelSequenceInfo(sequence=10, partition="0", offset=50)
        seq2 = ModelSequenceInfo(sequence=10, partition="0", offset=50)
        assert seq1.is_stale_compared_to(seq2) is False

    def test_offset_tiebreaker_requires_same_partition(self) -> None:
        """Test that offset tiebreaker only applies within same partition."""
        # Different partitions: offset should not be used as tiebreaker
        old = ModelSequenceInfo(sequence=10, partition="0", offset=50)
        new = ModelSequenceInfo(sequence=10, partition="1", offset=100)
        # Same sequence, different partition -> not stale (no tiebreaker applies)
        assert old.is_stale_compared_to(new) is False

    def test_offset_tiebreaker_requires_both_offsets(self) -> None:
        """Test that offset tiebreaker requires both sequences to have offsets."""
        old = ModelSequenceInfo(sequence=10, partition="0", offset=50)
        new = ModelSequenceInfo(sequence=10)  # No offset
        # Same sequence, one missing offset -> not stale
        assert old.is_stale_compared_to(new) is False

    def test_staleness_with_zero_sequence(self) -> None:
        """Test staleness comparison with sequence=0."""
        zero = ModelSequenceInfo(sequence=0)
        one = ModelSequenceInfo(sequence=1)
        assert zero.is_stale_compared_to(one) is True
        assert one.is_stale_compared_to(zero) is False


class TestIsNewerThan:
    """Tests for is_newer_than() method."""

    def test_higher_sequence_is_newer(self) -> None:
        """Test that higher sequence is considered newer."""
        old = ModelSequenceInfo(sequence=10)
        new = ModelSequenceInfo(sequence=20)
        assert new.is_newer_than(old) is True

    def test_lower_sequence_is_not_newer(self) -> None:
        """Test that lower sequence is not newer."""
        old = ModelSequenceInfo(sequence=10)
        new = ModelSequenceInfo(sequence=20)
        assert old.is_newer_than(new) is False

    def test_same_sequence_is_not_newer(self) -> None:
        """Test that same sequence is not newer."""
        seq1 = ModelSequenceInfo(sequence=10)
        seq2 = ModelSequenceInfo(sequence=10)
        assert seq1.is_newer_than(seq2) is False

    def test_is_newer_than_inverse_of_is_stale(self) -> None:
        """Test that is_newer_than is the inverse of is_stale_compared_to."""
        old = ModelSequenceInfo(sequence=10)
        new = ModelSequenceInfo(sequence=20)
        # new.is_newer_than(old) == old.is_stale_compared_to(new)
        assert new.is_newer_than(old) == old.is_stale_compared_to(new)

    def test_offset_tiebreaker_in_is_newer_than(self) -> None:
        """Test offset tiebreaker works with is_newer_than."""
        old = ModelSequenceInfo(sequence=10, partition="0", offset=50)
        new = ModelSequenceInfo(sequence=10, partition="0", offset=100)
        assert new.is_newer_than(old) is True
        assert old.is_newer_than(new) is False


class TestFromKafka:
    """Tests for from_kafka() factory method."""

    def test_from_kafka_basic(self) -> None:
        """Test from_kafka creates correct sequence info."""
        seq = ModelSequenceInfo.from_kafka(partition=0, offset=12345)
        assert seq.sequence == 12345
        assert seq.partition == "0"
        assert seq.offset == 12345

    def test_from_kafka_partition_converted_to_string(self) -> None:
        """Test that partition integer is converted to string."""
        seq = ModelSequenceInfo.from_kafka(partition=5, offset=100)
        assert seq.partition == "5"
        assert isinstance(seq.partition, str)

    def test_from_kafka_sequence_equals_offset(self) -> None:
        """Test that sequence equals offset for Kafka messages."""
        seq = ModelSequenceInfo.from_kafka(partition=2, offset=9999)
        assert seq.sequence == seq.offset

    def test_from_kafka_zero_values(self) -> None:
        """Test from_kafka with zero values."""
        seq = ModelSequenceInfo.from_kafka(partition=0, offset=0)
        assert seq.sequence == 0
        assert seq.partition == "0"
        assert seq.offset == 0

    def test_from_kafka_large_partition(self) -> None:
        """Test from_kafka with large partition number."""
        seq = ModelSequenceInfo.from_kafka(partition=100, offset=500)
        assert seq.partition == "100"


class TestFromSequence:
    """Tests for from_sequence() factory method."""

    def test_from_sequence_basic(self) -> None:
        """Test from_sequence creates correct sequence info."""
        seq = ModelSequenceInfo.from_sequence(42)
        assert seq.sequence == 42
        assert seq.partition is None
        assert seq.offset is None

    def test_from_sequence_zero(self) -> None:
        """Test from_sequence with zero value."""
        seq = ModelSequenceInfo.from_sequence(0)
        assert seq.sequence == 0

    def test_from_sequence_large_value(self) -> None:
        """Test from_sequence with large value."""
        seq = ModelSequenceInfo.from_sequence(9999999999)
        assert seq.sequence == 9999999999


class TestSequenceInfoEquality:
    """Tests for model equality comparison."""

    def test_same_values_are_equal(self) -> None:
        """Test that models with same values are equal."""
        seq1 = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        seq2 = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        assert seq1 == seq2

    def test_different_sequence_not_equal(self) -> None:
        """Test that different sequences are not equal."""
        seq1 = ModelSequenceInfo(sequence=100)
        seq2 = ModelSequenceInfo(sequence=200)
        assert seq1 != seq2

    def test_different_partition_not_equal(self) -> None:
        """Test that different partitions are not equal."""
        seq1 = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        seq2 = ModelSequenceInfo(sequence=100, partition="1", offset=100)
        assert seq1 != seq2

    def test_different_offset_not_equal(self) -> None:
        """Test that different offsets are not equal."""
        seq1 = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        seq2 = ModelSequenceInfo(sequence=100, partition="0", offset=200)
        assert seq1 != seq2


class TestSequenceInfoHashability:
    """Tests for model hashability (frozen models are hashable)."""

    def test_sequence_info_is_hashable(self) -> None:
        """Test that frozen model is hashable."""
        seq = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        # Should not raise
        hash_value = hash(seq)
        assert isinstance(hash_value, int)

    def test_equal_sequences_have_same_hash(self) -> None:
        """Test that equal models have the same hash."""
        seq1 = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        seq2 = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        assert hash(seq1) == hash(seq2)

    def test_can_be_used_in_set(self) -> None:
        """Test that sequence info can be used in sets."""
        seq1 = ModelSequenceInfo(sequence=100)
        seq2 = ModelSequenceInfo(sequence=100)  # Duplicate
        seq3 = ModelSequenceInfo(sequence=200)

        seq_set = {seq1, seq2, seq3}
        assert len(seq_set) == 2  # Deduplication

    def test_can_be_used_as_dict_key(self) -> None:
        """Test that sequence info can be used as dictionary key."""
        seq = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        cache: dict[ModelSequenceInfo, str] = {seq: "cached_value"}
        assert cache[seq] == "cached_value"


class TestSequenceInfoSerialization:
    """Tests for model serialization."""

    def test_model_dump(self) -> None:
        """Test serialization to dict."""
        seq = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        data = seq.model_dump()
        assert data == {"sequence": 100, "partition": "0", "offset": 100}

    def test_model_dump_minimal(self) -> None:
        """Test serialization with only required fields."""
        seq = ModelSequenceInfo(sequence=42)
        data = seq.model_dump()
        assert data == {"sequence": 42, "partition": None, "offset": None}

    def test_model_dump_json(self) -> None:
        """Test JSON serialization."""
        seq = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        json_str = seq.model_dump_json()
        assert '"sequence":100' in json_str
        assert '"partition":"0"' in json_str
        assert '"offset":100' in json_str

    def test_model_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {"sequence": 100, "partition": "0", "offset": 100}
        seq = ModelSequenceInfo.model_validate(data)
        assert seq.sequence == 100
        assert seq.partition == "0"
        assert seq.offset == 100


class TestOrderingEdgeCases:
    """Edge case tests for ordering semantics."""

    def test_mixed_partition_and_no_partition(self) -> None:
        """Test comparison between Kafka and non-Kafka sequences."""
        kafka_seq = ModelSequenceInfo(sequence=100, partition="0", offset=100)
        generic_seq = ModelSequenceInfo(sequence=100)
        # Same sequence, different transport - not stale
        assert not kafka_seq.is_stale_compared_to(generic_seq)
        assert not generic_seq.is_stale_compared_to(kafka_seq)

    def test_sequence_difference_takes_precedence(self) -> None:
        """Test that sequence difference overrides offset comparison."""
        # Lower sequence but higher offset should still be stale
        old = ModelSequenceInfo(sequence=10, partition="0", offset=1000)
        new = ModelSequenceInfo(sequence=20, partition="0", offset=1)
        assert old.is_stale_compared_to(new) is True

    def test_boundary_values(self) -> None:
        """Test with boundary values."""
        zero = ModelSequenceInfo(sequence=0, offset=0, partition="0")
        max_int = ModelSequenceInfo(sequence=2**63 - 1)  # Large but valid
        assert zero.is_stale_compared_to(max_int) is True
