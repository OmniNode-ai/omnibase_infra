# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for build_topic_router_from_contract() utility.

Validates that the utility correctly maps Python event class names to their
declared Kafka topics from a contract YAML dict.
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.contract_topic_router import (
    build_topic_router_from_contract,
)


@pytest.mark.unit
def test_builds_router_from_published_events() -> None:
    """Maps two events correctly from published_events list."""
    contract_data = {
        "published_events": [
            {
                "event_type": "NodeRegistrationAccepted",
                "topic": "onex.evt.platform.node-registration-accepted.v1",
            },
            {
                "event_type": "NodeBecameActive",
                "topic": "onex.evt.platform.node-became-active.v1",
            },
        ]
    }
    router = build_topic_router_from_contract(contract_data)
    assert router == {
        "ModelNodeRegistrationAccepted": "onex.evt.platform.node-registration-accepted.v1",
        "ModelNodeBecameActive": "onex.evt.platform.node-became-active.v1",
    }


@pytest.mark.unit
def test_returns_empty_dict_for_missing_published_events_key() -> None:
    """Empty dict input returns empty router."""
    router = build_topic_router_from_contract({})
    assert router == {}


@pytest.mark.unit
def test_returns_empty_dict_for_empty_published_events() -> None:
    """Empty published_events list returns empty router."""
    router = build_topic_router_from_contract({"published_events": []})
    assert router == {}


@pytest.mark.unit
def test_skips_entries_missing_event_type_or_topic() -> None:
    """Entries missing event_type or topic are silently skipped."""
    contract_data = {
        "published_events": [
            {
                "event_type": "NodeRegistrationAccepted",
                "topic": "onex.evt.platform.node-registration-accepted.v1",
            },
            {"event_type": "NodeOrphan"},  # missing topic
            {"topic": "onex.evt.platform.other.v1"},  # missing event_type
        ]
    }
    router = build_topic_router_from_contract(contract_data)
    assert len(router) == 1
    assert "ModelNodeRegistrationAccepted" in router
