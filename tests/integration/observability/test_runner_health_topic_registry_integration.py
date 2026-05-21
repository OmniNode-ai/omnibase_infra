# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runner health topic registration (OMN-11276)."""

from __future__ import annotations

import pytest

from omnibase_core.validation import validate_topic_suffix
from omnibase_infra import topics
from omnibase_infra.topics.platform_topic_suffixes import (
    ALL_OMNIBASE_INFRA_TOPIC_SPECS,
    SUFFIX_RUNNER_HEALTH_SNAPSHOT,
)


@pytest.mark.integration
def test_runner_health_snapshot_topic_is_exported_and_provisioned() -> None:
    """Runner health snapshots must be importable and provisioned at runtime."""
    assert topics.SUFFIX_RUNNER_HEALTH_SNAPSHOT == SUFFIX_RUNNER_HEALTH_SNAPSHOT
    assert validate_topic_suffix(SUFFIX_RUNNER_HEALTH_SNAPSHOT).is_valid

    specs_by_suffix = {spec.suffix: spec for spec in ALL_OMNIBASE_INFRA_TOPIC_SPECS}
    spec = specs_by_suffix[SUFFIX_RUNNER_HEALTH_SNAPSHOT]

    assert spec.partitions == 1
    assert spec.kafka_config is not None
    assert spec.kafka_config["retention.ms"] == "604800000"
    assert spec.kafka_config["cleanup.policy"] == "delete"
