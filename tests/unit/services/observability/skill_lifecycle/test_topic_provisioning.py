# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for skill lifecycle topic provisioning (OMN-2934).

Verifies that onex.evt.omniclaude.skill-started.v1 and
onex.evt.omniclaude.skill-completed.v1 are registered in the
omnibase_infra topic provisioning registry.
"""

from __future__ import annotations

import pytest

from omnibase_infra.topics.platform_topic_suffixes import (
    _OMNICLAUDE_SKILL_TOPIC_SUFFIXES,
    ALL_OMNICLAUDE_TOPIC_SPECS,
)

SKILL_STARTED_TOPIC = "onex.evt.omniclaude.skill-started.v1"
SKILL_COMPLETED_TOPIC = "onex.evt.omniclaude.skill-completed.v1"


class TestSkillLifecycleTopicProvisioning:
    """Verify skill lifecycle topics are registered for Kafka provisioning."""

    @pytest.mark.unit
    def test_skill_started_in_suffixes(self) -> None:
        """onex.evt.omniclaude.skill-started.v1 is in _OMNICLAUDE_SKILL_TOPIC_SUFFIXES."""
        assert SKILL_STARTED_TOPIC in _OMNICLAUDE_SKILL_TOPIC_SUFFIXES, (
            f"{SKILL_STARTED_TOPIC!r} is missing from _OMNICLAUDE_SKILL_TOPIC_SUFFIXES. "
            "This topic was introduced in OMN-2934 to provision the skill-lifecycle consumer."
        )

    @pytest.mark.unit
    def test_skill_completed_in_suffixes(self) -> None:
        """onex.evt.omniclaude.skill-completed.v1 is in _OMNICLAUDE_SKILL_TOPIC_SUFFIXES."""
        assert SKILL_COMPLETED_TOPIC in _OMNICLAUDE_SKILL_TOPIC_SUFFIXES, (
            f"{SKILL_COMPLETED_TOPIC!r} is missing from _OMNICLAUDE_SKILL_TOPIC_SUFFIXES. "
            "This topic was introduced in OMN-2934 to provision the skill-lifecycle consumer."
        )

    @pytest.mark.unit
    def test_skill_started_in_topic_specs(self) -> None:
        """onex.evt.omniclaude.skill-started.v1 appears in ALL_OMNICLAUDE_TOPIC_SPECS."""
        provisioned = {spec.suffix for spec in ALL_OMNICLAUDE_TOPIC_SPECS}
        assert SKILL_STARTED_TOPIC in provisioned, (
            f"{SKILL_STARTED_TOPIC!r} not found in ALL_OMNICLAUDE_TOPIC_SPECS. "
            "Topic will not be auto-created at broker startup."
        )

    @pytest.mark.unit
    def test_skill_completed_in_topic_specs(self) -> None:
        """onex.evt.omniclaude.skill-completed.v1 appears in ALL_OMNICLAUDE_TOPIC_SPECS."""
        provisioned = {spec.suffix for spec in ALL_OMNICLAUDE_TOPIC_SPECS}
        assert SKILL_COMPLETED_TOPIC in provisioned, (
            f"{SKILL_COMPLETED_TOPIC!r} not found in ALL_OMNICLAUDE_TOPIC_SPECS. "
            "Topic will not be auto-created at broker startup."
        )

    @pytest.mark.unit
    def test_no_duplicate_suffixes(self) -> None:
        """_OMNICLAUDE_SKILL_TOPIC_SUFFIXES contains no duplicate entries."""
        suffix_list = list(_OMNICLAUDE_SKILL_TOPIC_SUFFIXES)
        suffix_set = set(suffix_list)
        assert len(suffix_list) == len(suffix_set), (
            f"Duplicate topics found in _OMNICLAUDE_SKILL_TOPIC_SUFFIXES: "
            f"{[s for s in suffix_list if suffix_list.count(s) > 1]}"
        )

    @pytest.mark.unit
    def test_skill_lifecycle_topics_are_evt_kind(self) -> None:
        """Skill lifecycle topics use the 'evt' kind prefix."""
        assert SKILL_STARTED_TOPIC.startswith("onex.evt.")
        assert SKILL_COMPLETED_TOPIC.startswith("onex.evt.")

    @pytest.mark.unit
    def test_skill_lifecycle_topics_use_omniclaude_producer(self) -> None:
        """Skill lifecycle topics use the 'omniclaude' producer segment."""
        assert "omniclaude" in SKILL_STARTED_TOPIC
        assert "omniclaude" in SKILL_COMPLETED_TOPIC
