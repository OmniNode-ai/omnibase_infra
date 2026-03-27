# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for post-merge consumer configuration.

Related Tickets:
    - OMN-6727: post-merge consumer chain
"""

from __future__ import annotations

import pytest

from omnibase_infra.services.post_merge.config import ConfigPostMergeConsumer
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_GITHUB_PR_MERGED


@pytest.mark.unit
class TestConfigPostMergeConsumer:
    """Tests for ConfigPostMergeConsumer defaults and validation."""

    def test_defaults(self) -> None:
        config = ConfigPostMergeConsumer()
        assert config.kafka_bootstrap_servers == "localhost:19092"
        assert config.kafka_group_id == "post-merge-consumer"
        assert config.input_topic == SUFFIX_GITHUB_PR_MERGED
        assert config.auto_offset_reset == "earliest"
        assert config.hostile_review_enabled is True
        assert config.contract_sweep_enabled is True
        assert config.integration_check_enabled is True
        assert config.dry_run is False
        assert config.health_check_port == 8088

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POST_MERGE_KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        monkeypatch.setenv("POST_MERGE_DRY_RUN", "true")
        monkeypatch.setenv("POST_MERGE_HOSTILE_REVIEW_ENABLED", "false")
        config = ConfigPostMergeConsumer()
        assert config.kafka_bootstrap_servers == "kafka:9092"
        assert config.dry_run is True
        assert config.hostile_review_enabled is False

    def test_extra_fields_ignored(self) -> None:
        # extra="ignore" should not raise
        config = ConfigPostMergeConsumer.model_validate(
            {"unknown_field": "value", "kafka_bootstrap_servers": "localhost:19092"}
        )
        assert config.kafka_bootstrap_servers == "localhost:19092"

    def test_auto_ticket_min_severity_default(self) -> None:
        config = ConfigPostMergeConsumer()
        assert config.auto_ticket_min_severity == "medium"
