# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration-level unit tests for topic registry consolidation.

Validates the end-to-end correctness of the ProtocolTopicRegistry
consolidation: all keys resolvable, no legacy TOPIC_* imports remain,
and the registry is the single source of truth for topic strings.

Related:
    - OMN-5839: Topic registry consolidation epic
    - OMN-5848: Final verification ticket

.. versionadded:: 0.24.0
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.unit
class TestTopicRegistryIntegration:
    """End-to-end verification of topic registry consolidation."""

    def test_all_keys_resolvable(self) -> None:
        """Every topic key in __all__ resolves to a valid onex.* topic."""
        registry = ServiceTopicRegistry.from_defaults()
        for key_name in topic_keys.__all__:
            key = getattr(topic_keys, key_name)
            topic = registry.resolve(key)
            assert topic.startswith("onex."), (
                f"Topic for {key} doesn't match naming: {topic}"
            )

    def test_no_topic_constant_imports_remain_in_src(self) -> None:
        """No src/ file imports TOPIC_* from topic_constants."""
        result = subprocess.run(
            [
                "grep",
                "-rn",
                r"from omnibase_infra.event_bus.topic_constants import.*TOPIC_",
                "src/",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.stdout == "", (
            f"Remaining TOPIC_* imports in src/:\n{result.stdout}"
        )

    def test_no_wiring_health_monitored_imports_remain_in_src(self) -> None:
        """No src/ file imports WIRING_HEALTH_MONITORED_TOPICS."""
        result = subprocess.run(
            [
                "grep",
                "-rn",
                r"from omnibase_infra.event_bus.topic_constants import.*WIRING_HEALTH",
                "src/",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.stdout == "", (
            f"Remaining WIRING_HEALTH imports in src/:\n{result.stdout}"
        )

    def test_topic_constants_only_has_dlq(self) -> None:
        """topic_constants.py __all__ contains only DLQ items and functions."""
        from omnibase_infra.event_bus.topic_constants import __all__ as tc_all

        for name in tc_all:
            is_dlq = name.startswith(("DLQ_", "_DLQ_"))
            is_func = name in {
                "build_dlq_topic",
                "parse_dlq_topic",
                "is_dlq_topic",
                "get_dlq_topic_for_original",
                "derive_dlq_topic_for_event_type",
            }
            assert is_dlq or is_func, f"Non-DLQ item in topic_constants.__all__: {name}"

    def test_registry_key_count_matches_topic_keys(self) -> None:
        """Registry has exactly as many entries as topic_keys.__all__."""
        registry = ServiceTopicRegistry.from_defaults()
        assert len(registry.all_keys()) == len(topic_keys.__all__)

    def test_all_topic_strings_are_unique(self) -> None:
        """No two keys resolve to the same topic string."""
        registry = ServiceTopicRegistry.from_defaults()
        topics = [registry.resolve(k) for k in registry.all_keys()]
        assert len(set(topics)) == len(topics), (
            "Duplicate topic strings detected in registry"
        )
