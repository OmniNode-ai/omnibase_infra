# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for resolution event topic registration in platform topic registry.

Verifies that ``onex.evt.platform.resolution-decided.v1`` is declared
in the platform topic registry so ``TopicProvisioner`` creates it with
correct settings (retention, partition count) rather than auto-creating
on first write.

Related:
    - OMN-4325: Add onex.evt.platform.resolution-decided.v1 to platform topic registry
    - OMN-2895: Resolution Event Ledger
    - SUFFIX_RESOLUTION_DECIDED: The resolution event topic suffix constant
"""

from __future__ import annotations

import pytest

from omnibase_infra.event_bus.topic_constants import TOPIC_RESOLUTION_DECIDED
from omnibase_infra.topics import ALL_PLATFORM_SUFFIXES, ALL_PLATFORM_TOPIC_SPECS
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_RESOLUTION_DECIDED


@pytest.mark.unit
class TestResolutionTopicRegistered:
    """Resolution event topic is declared in the platform topic registry."""

    def test_suffix_resolution_decided_constant_value(self) -> None:
        """SUFFIX_RESOLUTION_DECIDED has the expected topic suffix value."""
        assert SUFFIX_RESOLUTION_DECIDED == "onex.evt.platform.resolution-decided.v1"

    def test_topic_constant_matches_suffix(self) -> None:
        """TOPIC_RESOLUTION_DECIDED in topic_constants matches SUFFIX_RESOLUTION_DECIDED."""
        assert TOPIC_RESOLUTION_DECIDED == SUFFIX_RESOLUTION_DECIDED

    def test_resolution_topic_in_platform_suffixes(self) -> None:
        """Resolution event topic is declared in ALL_PLATFORM_SUFFIXES."""
        assert SUFFIX_RESOLUTION_DECIDED in ALL_PLATFORM_SUFFIXES

    def test_resolution_topic_has_spec(self) -> None:
        """Resolution event topic has a ModelTopicSpec in ALL_PLATFORM_TOPIC_SPECS."""
        matching_specs = [
            spec
            for spec in ALL_PLATFORM_TOPIC_SPECS
            if spec.suffix == SUFFIX_RESOLUTION_DECIDED
        ]
        assert len(matching_specs) == 1, (
            f"Expected exactly 1 spec for {SUFFIX_RESOLUTION_DECIDED!r}, "
            f"found {len(matching_specs)}"
        )

    def test_resolution_topic_spec_has_partitions(self) -> None:
        """Resolution event topic spec has at least 1 partition configured."""
        matching_specs = [
            spec
            for spec in ALL_PLATFORM_TOPIC_SPECS
            if spec.suffix == SUFFIX_RESOLUTION_DECIDED
        ]
        assert len(matching_specs) == 1
        spec = matching_specs[0]
        assert spec.partitions >= 1, (
            f"Expected at least 1 partition for {SUFFIX_RESOLUTION_DECIDED!r}, "
            f"got {spec.partitions}"
        )
