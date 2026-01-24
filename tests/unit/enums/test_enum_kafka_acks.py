# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for EnumKafkaAcks.

Tests the to_aiokafka() method that converts enum values to the types
expected by aiokafka's AIOKafkaProducer.
"""

import pytest

from omnibase_infra.enums import EnumKafkaAcks
from omnibase_infra.enums.enum_kafka_acks import _AIOKAFKA_MAP


class TestEnumKafkaAcksValues:
    """Tests for EnumKafkaAcks enum values."""

    def test_all_value(self) -> None:
        """ALL enum value should be 'all'."""
        assert EnumKafkaAcks.ALL.value == "all"

    def test_none_value(self) -> None:
        """NONE enum value should be '0'."""
        assert EnumKafkaAcks.NONE.value == "0"

    def test_leader_value(self) -> None:
        """LEADER enum value should be '1'."""
        assert EnumKafkaAcks.LEADER.value == "1"

    def test_all_replicas_value(self) -> None:
        """ALL_REPLICAS enum value should be '-1'."""
        assert EnumKafkaAcks.ALL_REPLICAS.value == "-1"


class TestToAiokafka:
    """Tests for EnumKafkaAcks.to_aiokafka() method."""

    def test_all_returns_string(self) -> None:
        """ALL should return string 'all' for aiokafka."""
        result = EnumKafkaAcks.ALL.to_aiokafka()
        assert result == "all"
        assert isinstance(result, str)

    def test_none_returns_int_zero(self) -> None:
        """NONE should return integer 0 for aiokafka."""
        result = EnumKafkaAcks.NONE.to_aiokafka()
        assert result == 0
        assert isinstance(result, int)

    def test_leader_returns_int_one(self) -> None:
        """LEADER should return integer 1 for aiokafka."""
        result = EnumKafkaAcks.LEADER.to_aiokafka()
        assert result == 1
        assert isinstance(result, int)

    def test_all_replicas_returns_int_negative_one(self) -> None:
        """ALL_REPLICAS should return integer -1 for aiokafka."""
        result = EnumKafkaAcks.ALL_REPLICAS.to_aiokafka()
        assert result == -1
        assert isinstance(result, int)


class TestEnumKafkaAcksFromString:
    """Tests for creating EnumKafkaAcks from string values."""

    def test_from_string_all(self) -> None:
        """Should create ALL from string 'all'."""
        assert EnumKafkaAcks("all") == EnumKafkaAcks.ALL

    def test_from_string_zero(self) -> None:
        """Should create NONE from string '0'."""
        assert EnumKafkaAcks("0") == EnumKafkaAcks.NONE

    def test_from_string_one(self) -> None:
        """Should create LEADER from string '1'."""
        assert EnumKafkaAcks("1") == EnumKafkaAcks.LEADER

    def test_from_string_negative_one(self) -> None:
        """Should create ALL_REPLICAS from string '-1'."""
        assert EnumKafkaAcks("-1") == EnumKafkaAcks.ALL_REPLICAS

    def test_invalid_string_raises_value_error(self) -> None:
        """Should raise ValueError for invalid string values."""
        with pytest.raises(ValueError):
            EnumKafkaAcks("invalid")

    def test_invalid_numeric_string_raises_value_error(self) -> None:
        """Should raise ValueError for numeric strings that are not valid acks."""
        with pytest.raises(ValueError):
            EnumKafkaAcks("2")


class TestEnumKafkaAcksIsStrEnum:
    """Tests verifying EnumKafkaAcks is a string enum."""

    def test_is_string_subclass(self) -> None:
        """EnumKafkaAcks values should be string instances."""
        for member in EnumKafkaAcks:
            assert isinstance(member.value, str)

    def test_can_use_in_string_context(self) -> None:
        """EnumKafkaAcks should work in string contexts."""
        acks = EnumKafkaAcks.ALL
        assert f"acks={acks.value}" == "acks=all"


class TestAllMembersHaveToAiokafka:
    """Tests ensuring all enum members have valid to_aiokafka() behavior."""

    def test_all_members_return_valid_types(self) -> None:
        """All enum members should return int or str from to_aiokafka()."""
        for member in EnumKafkaAcks:
            result = member.to_aiokafka()
            assert isinstance(result, (int, str)), (
                f"{member.name}.to_aiokafka() returned {type(result)}"
            )

    def test_to_aiokafka_type_mapping(self) -> None:
        """Verify exact type mapping for all members."""
        expected_types = {
            EnumKafkaAcks.ALL: str,
            EnumKafkaAcks.NONE: int,
            EnumKafkaAcks.LEADER: int,
            EnumKafkaAcks.ALL_REPLICAS: int,
        }
        for member, expected_type in expected_types.items():
            result = member.to_aiokafka()
            assert isinstance(result, expected_type), (
                f"{member.name}.to_aiokafka() should return {expected_type.__name__}, "
                f"got {type(result).__name__}"
            )

    def test_aiokafka_map_covers_all_members(self) -> None:
        """_AIOKAFKA_MAP should have entry for every enum member."""
        for member in EnumKafkaAcks:
            assert member.value in _AIOKAFKA_MAP, (
                f"Missing _AIOKAFKA_MAP entry for {member.name}"
            )
