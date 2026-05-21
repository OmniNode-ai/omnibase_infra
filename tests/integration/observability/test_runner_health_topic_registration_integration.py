# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration check that the runner health snapshot Kafka topic is registered
in the platform topic registry with the expected retention config (OMN-11276)."""

from __future__ import annotations

import pytest

from omnibase_infra.topics.platform_topic_suffixes import (
    ALL_OMNIBASE_INFRA_TOPIC_SPECS,
    SUFFIX_RUNNER_HEALTH_SNAPSHOT,
)


@pytest.mark.integration
def test_runner_health_snapshot_topic_registered_with_retention() -> None:
    """SUFFIX_RUNNER_HEALTH_SNAPSHOT must appear in the platform topic registry
    with the OMN-11276 retention config (7-day delete policy, 1 partition)."""
    matches = [
        spec
        for spec in ALL_OMNIBASE_INFRA_TOPIC_SPECS
        if spec.suffix == SUFFIX_RUNNER_HEALTH_SNAPSHOT
    ]
    assert len(matches) == 1, (
        f"Expected exactly one topic spec for SUFFIX_RUNNER_HEALTH_SNAPSHOT, "
        f"got {len(matches)}"
    )
    spec = matches[0]
    assert spec.partitions == 1
    assert spec.kafka_config == {
        "retention.ms": "604800000",
        "cleanup.policy": "delete",
    }
