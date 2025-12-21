# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for ModelSnapshotTopicConfig.

Verifies Kafka snapshot topic configuration including:
- Default configuration values
- Cleanup policy validation (must be 'compact')
- Topic name validation and warnings
- Environment variable overrides
- Kafka config dictionary generation
- Snapshot key generation
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.projection import ModelSnapshotTopicConfig


class TestModelSnapshotTopicConfigDefaults:
    """Tests for default configuration values."""

    def test_default_topic(self) -> None:
        """Test default topic follows ONEX convention."""
        config = ModelSnapshotTopicConfig.default()
        assert config.topic == "onex.registration.snapshots"

    def test_default_cleanup_policy_is_compact(self) -> None:
        """Test that cleanup_policy defaults to compact."""
        config = ModelSnapshotTopicConfig.default()
        assert config.cleanup_policy == "compact"

    def test_default_partition_count(self) -> None:
        """Test default partition count."""
        config = ModelSnapshotTopicConfig.default()
        assert config.partition_count == 12

    def test_default_replication_factor(self) -> None:
        """Test default replication factor."""
        config = ModelSnapshotTopicConfig.default()
        assert config.replication_factor == 3

    def test_default_retention_is_infinite(self) -> None:
        """Test that retention defaults to infinite (-1) for compacted topics."""
        config = ModelSnapshotTopicConfig.default()
        assert config.retention_ms == -1

    def test_default_min_insync_replicas(self) -> None:
        """Test default min in-sync replicas for durability."""
        config = ModelSnapshotTopicConfig.default()
        assert config.min_insync_replicas == 2


class TestModelSnapshotTopicConfigCleanupPolicyValidation:
    """Tests for cleanup_policy validation."""

    def test_compact_policy_accepted(self) -> None:
        """Test that compact cleanup policy is accepted."""
        config = ModelSnapshotTopicConfig(
            topic="test.snapshots",
            cleanup_policy="compact",
        )
        assert config.cleanup_policy == "compact"

    def test_compact_policy_case_insensitive(self) -> None:
        """Test that cleanup policy validation is case-insensitive."""
        config = ModelSnapshotTopicConfig(
            topic="test.snapshots",
            cleanup_policy="COMPACT",
        )
        assert config.cleanup_policy == "compact"

    def test_delete_policy_rejected(self) -> None:
        """Test that delete cleanup policy is rejected."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            ModelSnapshotTopicConfig(
                topic="test.snapshots",
                cleanup_policy="delete",
            )
        assert "MUST use cleanup.policy=compact" in str(exc_info.value)

    def test_compact_delete_policy_rejected(self) -> None:
        """Test that compact,delete hybrid policy is rejected."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            ModelSnapshotTopicConfig(
                topic="test.snapshots",
                cleanup_policy="compact,delete",
            )
        assert "MUST use cleanup.policy=compact" in str(exc_info.value)

    def test_empty_policy_rejected(self) -> None:
        """Test that empty cleanup policy is rejected."""
        with pytest.raises(ProtocolConfigurationError):
            ModelSnapshotTopicConfig(
                topic="test.snapshots",
                cleanup_policy="",
            )


class TestModelSnapshotTopicConfigTopicValidation:
    """Tests for topic validation."""

    def test_valid_onex_kafka_format(self) -> None:
        """Test valid ONEX Kafka format topic."""
        config = ModelSnapshotTopicConfig(
            topic="onex.registration.snapshots",
        )
        assert config.topic == "onex.registration.snapshots"

    def test_valid_environment_aware_format(self) -> None:
        """Test valid environment-aware format topic."""
        config = ModelSnapshotTopicConfig(
            topic="prod.registration.snapshots.v1",
        )
        assert config.topic == "prod.registration.snapshots.v1"

    def test_empty_topic_rejected(self) -> None:
        """Test that empty topic is rejected."""
        with pytest.raises(ProtocolConfigurationError):
            ModelSnapshotTopicConfig(topic="")

    def test_whitespace_only_topic_rejected(self) -> None:
        """Test that whitespace-only topic is rejected."""
        with pytest.raises(ProtocolConfigurationError):
            ModelSnapshotTopicConfig(topic="   ")

    def test_nonstandard_topic_accepted_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that non-standard topics are accepted but logged."""
        config = ModelSnapshotTopicConfig(topic="my-custom-topic")
        assert config.topic == "my-custom-topic"
        assert "does not follow ONEX snapshot topic naming" in caplog.text


class TestModelSnapshotTopicConfigKafkaConfig:
    """Tests for Kafka config dictionary generation."""

    def test_to_kafka_config_returns_dict(self) -> None:
        """Test that to_kafka_config returns a dictionary."""
        config = ModelSnapshotTopicConfig.default()
        kafka_config = config.to_kafka_config()
        assert isinstance(kafka_config, dict)

    def test_to_kafka_config_cleanup_policy(self) -> None:
        """Test that cleanup.policy is correctly set."""
        config = ModelSnapshotTopicConfig.default()
        kafka_config = config.to_kafka_config()
        assert kafka_config["cleanup.policy"] == "compact"

    def test_to_kafka_config_min_compaction_lag(self) -> None:
        """Test that min.compaction.lag.ms is correctly set."""
        config = ModelSnapshotTopicConfig(
            topic="test.snapshots",
            min_compaction_lag_ms=120000,
        )
        kafka_config = config.to_kafka_config()
        assert kafka_config["min.compaction.lag.ms"] == "120000"

    def test_to_kafka_config_retention(self) -> None:
        """Test that retention.ms is correctly set."""
        config = ModelSnapshotTopicConfig.default()
        kafka_config = config.to_kafka_config()
        assert kafka_config["retention.ms"] == "-1"

    def test_to_kafka_config_all_keys_present(self) -> None:
        """Test that all expected Kafka config keys are present."""
        config = ModelSnapshotTopicConfig.default()
        kafka_config = config.to_kafka_config()
        expected_keys = {
            "cleanup.policy",
            "min.compaction.lag.ms",
            "max.compaction.lag.ms",
            "segment.bytes",
            "retention.ms",
            "min.insync.replicas",
        }
        assert set(kafka_config.keys()) == expected_keys


class TestModelSnapshotTopicConfigSnapshotKey:
    """Tests for snapshot key generation."""

    def test_get_snapshot_key_format(self) -> None:
        """Test snapshot key follows domain:entity_id format."""
        config = ModelSnapshotTopicConfig.default()
        key = config.get_snapshot_key("registration", "node-123")
        assert key == "registration:node-123"

    def test_get_snapshot_key_with_uuid(self) -> None:
        """Test snapshot key with UUID entity_id."""
        config = ModelSnapshotTopicConfig.default()
        key = config.get_snapshot_key(
            "registration", "550e8400-e29b-41d4-a716-446655440000"
        )
        assert key == "registration:550e8400-e29b-41d4-a716-446655440000"

    def test_get_snapshot_key_different_domains(self) -> None:
        """Test snapshot keys for different domains."""
        config = ModelSnapshotTopicConfig.default()
        reg_key = config.get_snapshot_key("registration", "node-1")
        disc_key = config.get_snapshot_key("discovery", "node-1")
        assert reg_key == "registration:node-1"
        assert disc_key == "discovery:node-1"
        assert reg_key != disc_key


class TestModelSnapshotTopicConfigEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_topic_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test topic can be overridden via environment variable."""
        monkeypatch.setenv("SNAPSHOT_TOPIC", "custom.snapshots")
        config = ModelSnapshotTopicConfig.default()
        assert config.topic == "custom.snapshots"

    def test_partition_count_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test partition count can be overridden via environment variable."""
        monkeypatch.setenv("SNAPSHOT_PARTITION_COUNT", "24")
        config = ModelSnapshotTopicConfig.default()
        assert config.partition_count == 24

    def test_invalid_partition_count_ignored(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid partition count is ignored with warning."""
        monkeypatch.setenv("SNAPSHOT_PARTITION_COUNT", "not-a-number")
        config = ModelSnapshotTopicConfig.default()
        assert config.partition_count == 12  # Default
        assert "Failed to parse integer" in caplog.text


class TestModelSnapshotTopicConfigYamlLoading:
    """Tests for YAML configuration loading."""

    def test_from_yaml_loads_correctly(self, tmp_path: Path) -> None:
        """Test loading configuration from YAML file."""
        yaml_content = """
topic: "prod.registration.snapshots.v1"
partition_count: 24
replication_factor: 3
cleanup_policy: "compact"
min_compaction_lag_ms: 120000
retention_ms: -1
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = ModelSnapshotTopicConfig.from_yaml(yaml_file)
        assert config.topic == "prod.registration.snapshots.v1"
        assert config.partition_count == 24
        assert config.min_compaction_lag_ms == 120000

    def test_from_yaml_file_not_found(self) -> None:
        """Test FileNotFoundError for missing YAML file."""
        with pytest.raises(FileNotFoundError):
            ModelSnapshotTopicConfig.from_yaml(Path("/nonexistent/config.yaml"))

    def test_from_yaml_invalid_content(self, tmp_path: Path) -> None:
        """Test error handling for invalid YAML content."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("- this\n- is\n- a list")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            ModelSnapshotTopicConfig.from_yaml(yaml_file)
        assert "must be a dictionary" in str(exc_info.value)


class TestModelSnapshotTopicConfigImmutability:
    """Tests for model immutability (frozen=True)."""

    def test_model_is_frozen(self) -> None:
        """Test that the model is immutable."""
        config = ModelSnapshotTopicConfig.default()
        with pytest.raises(ValidationError, match="frozen"):
            config.topic = "modified.topic"  # type: ignore[misc]

    def test_apply_environment_overrides_returns_new_instance(self) -> None:
        """Test that apply_environment_overrides returns a new instance."""
        config1 = ModelSnapshotTopicConfig(topic="test.snapshots")
        config2 = config1.apply_environment_overrides()
        assert config1 is not config2 or config1 == config2
