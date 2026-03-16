# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for health_published_events_map health check function.

OMN-5159
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.health.health_published_events_map import (
    check_published_events_map_health,
)


@pytest.mark.unit
class TestCheckPublishedEventsMapHealth:
    """Tests for check_published_events_map_health."""

    def test_returns_unhealthy_when_none(self) -> None:
        result = check_published_events_map_health(None)
        assert result.status == "unhealthy"
        assert result.name == "published_events_map"
        assert result.error is not None
        assert "empty or not loaded" in result.error

    def test_returns_unhealthy_when_empty(self) -> None:
        result = check_published_events_map_health({})
        assert result.status == "unhealthy"
        assert result.name == "published_events_map"
        assert result.error is not None
        assert "empty or not loaded" in result.error

    def test_returns_healthy_with_entries(self) -> None:
        result = check_published_events_map_health(
            {"ModelNodeRegistered": "onex.evt.platform.node-registered.v1"}
        )
        assert result.status == "healthy"
        assert result.name == "published_events_map"
        assert result.error is None
        assert result.details is not None
        assert result.details["entry_count"] == 1

    def test_returns_healthy_with_multiple_entries(self) -> None:
        result = check_published_events_map_health(
            {
                "ModelNodeRegistered": "onex.evt.platform.node-registered.v1",
                "ModelNodeBecameActive": "onex.evt.platform.node-became-active.v1",
                "ModelTopicCatalogResponse": "onex.evt.platform.topic-catalog-response.v1",
            }
        )
        assert result.status == "healthy"
        assert result.details is not None
        assert result.details["entry_count"] == 3

    def test_includes_contract_path_in_unhealthy_error(self) -> None:
        result = check_published_events_map_health(
            None, contract_path="/path/to/contract.yaml"
        )
        assert result.status == "unhealthy"
        assert result.error is not None
        assert "/path/to/contract.yaml" in result.error

    def test_no_contract_path_in_unhealthy_error(self) -> None:
        result = check_published_events_map_health(None, contract_path=None)
        assert result.status == "unhealthy"
        assert result.error is not None
        assert "contract:" not in result.error
