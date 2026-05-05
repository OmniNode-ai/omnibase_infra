# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for TopicProvisioner contract extraction and partition capping.

Tests end-to-end behavior introduced in OMN-10534:
- Contract-driven topic extraction from a real directory structure
- Partition cap enforcement via ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS env var
- Provisioning priority sort order respected across extracted topics
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

pytestmark = [pytest.mark.integration]


def _bootstrap_servers() -> str:
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@pytest.fixture
def contracts_root_with_priority(tmp_path: Path) -> Path:
    """Contracts directory with two nodes at different provisioning priorities."""
    high_dir = tmp_path / "node_high"
    high_dir.mkdir()
    (high_dir / "contract.yaml").write_text(
        "name: node_high\n"
        "version: 1.0.0\n"
        "namespace: onex.stamped\n"
        "event_bus:\n"
        "  publish_topics:\n"
        "    - topic: onex.evt.test-high.ready.v1\n"
        "      provisioning_priority: 10\n"
    )

    low_dir = tmp_path / "node_low"
    low_dir.mkdir()
    (low_dir / "contract.yaml").write_text(
        "name: node_low\n"
        "version: 1.0.0\n"
        "namespace: onex.stamped\n"
        "event_bus:\n"
        "  publish_topics:\n"
        "    - topic: onex.evt.test-low.events.v1\n"
        "      provisioning_priority: 50\n"
    )
    return tmp_path


@pytest.fixture
def contracts_root_high_partition(tmp_path: Path) -> Path:
    """Contracts directory with a topic declaring many partitions."""
    node_dir = tmp_path / "node_partition"
    node_dir.mkdir()
    (node_dir / "contract.yaml").write_text(
        "name: node_partition\n"
        "version: 1.0.0\n"
        "namespace: onex.stamped\n"
        "event_bus:\n"
        "  publish_topics:\n"
        "    - topic: onex.evt.test-partition.data.v1\n"
    )
    return tmp_path


class TestTopicProvisionerContractExtraction:
    """End-to-end: contract extraction, sort order, and partition cap."""

    def test_extracts_topics_from_contracts_directory(
        self, contracts_root_with_priority: Path
    ) -> None:
        provisioner = TopicProvisioner(
            bootstrap_servers=_bootstrap_servers(),
            contracts_root=contracts_root_with_priority,
        )
        suffixes = [s.suffix for s in provisioner._topic_specs]
        assert "onex.evt.test-high.ready.v1" in suffixes
        assert "onex.evt.test-low.events.v1" in suffixes

    def test_topics_sorted_by_provisioning_priority(
        self, contracts_root_with_priority: Path
    ) -> None:
        provisioner = TopicProvisioner(
            bootstrap_servers=_bootstrap_servers(),
            contracts_root=contracts_root_with_priority,
        )
        suffixes = [s.suffix for s in provisioner._topic_specs]
        high_idx = next(i for i, s in enumerate(suffixes) if "test-high" in s)
        low_idx = next(i for i, s in enumerate(suffixes) if "test-low" in s)
        assert high_idx < low_idx, (
            "Higher-priority topic must sort before lower-priority"
        )

    def test_partition_cap_applied_when_env_set(
        self, contracts_root_high_partition: Path
    ) -> None:
        with patch.dict(os.environ, {"ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS": "2"}):
            provisioner = TopicProvisioner(
                bootstrap_servers=_bootstrap_servers(),
                contracts_root=contracts_root_high_partition,
            )
        for spec in provisioner._topic_specs:
            assert provisioner._creation_partitions(spec) <= 2

    def test_no_partition_cap_when_env_unset(
        self, contracts_root_high_partition: Path
    ) -> None:
        env = {
            k: v
            for k, v in os.environ.items()
            if k != "ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS"
        }
        with patch.dict(os.environ, env, clear=True):
            provisioner = TopicProvisioner(
                bootstrap_servers=_bootstrap_servers(),
                contracts_root=contracts_root_high_partition,
            )
        assert provisioner._topic_partition_cap is None
