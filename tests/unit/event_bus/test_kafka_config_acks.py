# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelKafkaEventBusConfig acks handling.

Tests the EnumKafkaAcks integration with the config model,
including the acks_aiokafka computed field and environment variable parsing.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from omnibase_infra.enums import EnumKafkaAcks
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


class TestAcksFieldType:
    """Tests for the acks field using EnumKafkaAcks."""

    def test_default_acks_is_all(self) -> None:
        """Default acks should be EnumKafkaAcks.ALL."""
        config = ModelKafkaEventBusConfig()
        assert config.acks == EnumKafkaAcks.ALL

    def test_explicit_enum_value(self) -> None:
        """Should accept explicit EnumKafkaAcks values."""
        config = ModelKafkaEventBusConfig(acks=EnumKafkaAcks.LEADER)
        assert config.acks == EnumKafkaAcks.LEADER

    def test_string_coercion_all(self) -> None:
        """Should coerce string 'all' to EnumKafkaAcks.ALL."""
        config = ModelKafkaEventBusConfig(acks="all")
        assert config.acks == EnumKafkaAcks.ALL

    def test_string_coercion_zero(self) -> None:
        """Should coerce string '0' to EnumKafkaAcks.NONE."""
        config = ModelKafkaEventBusConfig(acks="0")
        assert config.acks == EnumKafkaAcks.NONE

    def test_string_coercion_one(self) -> None:
        """Should coerce string '1' to EnumKafkaAcks.LEADER."""
        config = ModelKafkaEventBusConfig(acks="1")
        assert config.acks == EnumKafkaAcks.LEADER

    def test_string_coercion_negative_one(self) -> None:
        """Should coerce string '-1' to EnumKafkaAcks.ALL_REPLICAS."""
        config = ModelKafkaEventBusConfig(acks="-1")
        assert config.acks == EnumKafkaAcks.ALL_REPLICAS

    def test_invalid_string_raises_error(self) -> None:
        """Should reject invalid string values."""
        with pytest.raises(ValueError):
            ModelKafkaEventBusConfig(acks="invalid")

    def test_invalid_numeric_string_raises_error(self) -> None:
        """Should reject numeric strings that are not valid acks."""
        with pytest.raises(ValueError):
            ModelKafkaEventBusConfig(acks="2")


class TestAcksAiokafkaComputedField:
    """Tests for the acks_aiokafka computed field."""

    def test_all_returns_string(self) -> None:
        """acks_aiokafka should return 'all' string for ALL."""
        config = ModelKafkaEventBusConfig(acks=EnumKafkaAcks.ALL)
        assert config.acks_aiokafka == "all"
        assert isinstance(config.acks_aiokafka, str)

    def test_none_returns_int_zero(self) -> None:
        """acks_aiokafka should return integer 0 for NONE."""
        config = ModelKafkaEventBusConfig(acks=EnumKafkaAcks.NONE)
        assert config.acks_aiokafka == 0
        assert isinstance(config.acks_aiokafka, int)

    def test_leader_returns_int_one(self) -> None:
        """acks_aiokafka should return integer 1 for LEADER."""
        config = ModelKafkaEventBusConfig(acks=EnumKafkaAcks.LEADER)
        assert config.acks_aiokafka == 1
        assert isinstance(config.acks_aiokafka, int)

    def test_all_replicas_returns_int_negative_one(self) -> None:
        """acks_aiokafka should return integer -1 for ALL_REPLICAS."""
        config = ModelKafkaEventBusConfig(acks=EnumKafkaAcks.ALL_REPLICAS)
        assert config.acks_aiokafka == -1
        assert isinstance(config.acks_aiokafka, int)

    def test_default_config_acks_aiokafka(self) -> None:
        """Default config should have acks_aiokafka as 'all' string."""
        config = ModelKafkaEventBusConfig.default()
        assert config.acks_aiokafka == "all"
        assert isinstance(config.acks_aiokafka, str)


class TestAcksEnvironmentOverride:
    """Tests for environment variable override of acks field."""

    def test_env_override_all(self) -> None:
        """Environment variable 'all' should set acks to ALL."""
        with patch.dict(os.environ, {"KAFKA_ACKS": "all"}, clear=False):
            config = ModelKafkaEventBusConfig.default()
            assert config.acks == EnumKafkaAcks.ALL
            assert config.acks_aiokafka == "all"

    def test_env_override_zero(self) -> None:
        """Environment variable '0' should set acks to NONE."""
        with patch.dict(os.environ, {"KAFKA_ACKS": "0"}, clear=False):
            config = ModelKafkaEventBusConfig.default()
            assert config.acks == EnumKafkaAcks.NONE
            assert config.acks_aiokafka == 0

    def test_env_override_one(self) -> None:
        """Environment variable '1' should set acks to LEADER."""
        with patch.dict(os.environ, {"KAFKA_ACKS": "1"}, clear=False):
            config = ModelKafkaEventBusConfig.default()
            assert config.acks == EnumKafkaAcks.LEADER
            assert config.acks_aiokafka == 1

    def test_env_override_negative_one(self) -> None:
        """Environment variable '-1' should set acks to ALL_REPLICAS."""
        with patch.dict(os.environ, {"KAFKA_ACKS": "-1"}, clear=False):
            config = ModelKafkaEventBusConfig.default()
            assert config.acks == EnumKafkaAcks.ALL_REPLICAS
            assert config.acks_aiokafka == -1

    def test_invalid_env_raises_error(self) -> None:
        """Invalid environment variable should raise ProtocolConfigurationError."""
        with patch.dict(os.environ, {"KAFKA_ACKS": "invalid"}, clear=False):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelKafkaEventBusConfig.default()
            assert "KAFKA_ACKS='invalid'" in str(exc_info.value)
            assert "Valid values are: all, 0, 1, -1" in str(exc_info.value)

    def test_numeric_invalid_env_raises_error(self) -> None:
        """Invalid numeric environment variable should raise ProtocolConfigurationError."""
        with patch.dict(os.environ, {"KAFKA_ACKS": "2"}, clear=False):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelKafkaEventBusConfig.default()
            assert "KAFKA_ACKS='2'" in str(exc_info.value)
            assert "Valid values are: all, 0, 1, -1" in str(exc_info.value)


class TestAcksModelDump:
    """Tests for model serialization with acks field."""

    def test_model_dump_includes_acks(self) -> None:
        """model_dump should include the acks field."""
        config = ModelKafkaEventBusConfig(acks=EnumKafkaAcks.LEADER)
        data = config.model_dump()
        assert "acks" in data
        assert data["acks"] == EnumKafkaAcks.LEADER

    def test_model_dump_includes_acks_aiokafka(self) -> None:
        """model_dump should include the computed acks_aiokafka field."""
        config = ModelKafkaEventBusConfig(acks=EnumKafkaAcks.LEADER)
        data = config.model_dump()
        assert "acks_aiokafka" in data
        assert data["acks_aiokafka"] == 1


class TestAcksIntegrationWithEventBus:
    """Integration tests verifying acks works correctly with EventBusKafka."""

    def test_config_acks_aiokafka_is_correct_type_for_producer(self) -> None:
        """acks_aiokafka should return types compatible with aiokafka producer."""
        # Test all enum variants
        test_cases = [
            (EnumKafkaAcks.ALL, "all", str),
            (EnumKafkaAcks.NONE, 0, int),
            (EnumKafkaAcks.LEADER, 1, int),
            (EnumKafkaAcks.ALL_REPLICAS, -1, int),
        ]
        for acks_enum, expected_value, expected_type in test_cases:
            config = ModelKafkaEventBusConfig(acks=acks_enum)
            result = config.acks_aiokafka
            assert result == expected_value, (
                f"Expected {expected_value} for {acks_enum.name}, got {result}"
            )
            assert isinstance(result, expected_type), (
                f"Expected {expected_type.__name__} for {acks_enum.name}, "
                f"got {type(result).__name__}"
            )
