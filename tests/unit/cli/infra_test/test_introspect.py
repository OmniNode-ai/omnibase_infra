# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for introspection event payload construction."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import click
import pytest

from omnibase_infra.cli.infra_test.introspect import _build_introspection_payload


@pytest.mark.unit
class TestBuildIntrospectionPayload:
    """Test introspection event payload builder."""

    def test_auto_generates_node_id(self) -> None:
        """Payload auto-generates a valid UUID for node_id."""
        payload = _build_introspection_payload()
        assert UUID(str(payload["node_id"]))

    def test_uses_provided_node_id(self) -> None:
        """Payload uses the provided node_id."""
        nid = "12345678-1234-1234-1234-123456789abc"
        payload = _build_introspection_payload(node_id=nid)
        assert payload["node_id"] == nid

    def test_default_node_type(self) -> None:
        """Default node type is EFFECT."""
        payload = _build_introspection_payload()
        assert payload["node_type"] == "EFFECT"

    def test_custom_node_type(self) -> None:
        """Custom node type is set correctly."""
        payload = _build_introspection_payload(node_type="ORCHESTRATOR")
        assert payload["node_type"] == "ORCHESTRATOR"

    def test_has_correlation_id(self) -> None:
        """Payload includes a correlation_id UUID."""
        payload = _build_introspection_payload()
        assert UUID(str(payload["correlation_id"]))

    def test_has_timestamp(self) -> None:
        """Payload includes a valid ISO 8601 timestamp."""
        payload = _build_introspection_payload()
        ts = str(payload["timestamp"])
        assert ts
        # Verify it's parseable as ISO 8601
        parsed = datetime.fromisoformat(ts)
        assert parsed.year >= 2024

    def test_has_version_semver(self) -> None:
        """Payload includes node_version as semver dict."""
        payload = _build_introspection_payload()
        version = payload["node_version"]
        assert isinstance(version, dict)
        assert version["major"] == 1
        assert version["minor"] == 0
        assert version["patch"] == 0

    def test_custom_version(self) -> None:
        """Custom version string is parsed into node_version."""
        payload = _build_introspection_payload(version="2.3.4")
        version = payload["node_version"]
        assert isinstance(version, dict)
        assert version["major"] == 2
        assert version["minor"] == 3
        assert version["patch"] == 4

    def test_has_endpoints(self) -> None:
        """Payload includes health endpoint."""
        payload = _build_introspection_payload()
        endpoints = payload["endpoints"]
        assert isinstance(endpoints, dict)
        assert "health" in endpoints

    def test_reason_is_startup(self) -> None:
        """Default reason is STARTUP."""
        payload = _build_introspection_payload()
        assert payload["reason"] == "STARTUP"

    def test_partial_version_major_only(self) -> None:
        """Partial version with major only fills minor and patch with 0."""
        payload = _build_introspection_payload(version="1")
        version = payload["node_version"]
        assert isinstance(version, dict)
        assert version["major"] == 1
        assert version["minor"] == 0
        assert version["patch"] == 0

    def test_partial_version_major_minor(self) -> None:
        """Partial version with major.minor fills patch with 0."""
        payload = _build_introspection_payload(version="1.2")
        version = payload["node_version"]
        assert isinstance(version, dict)
        assert version["major"] == 1
        assert version["minor"] == 2
        assert version["patch"] == 0

    def test_extra_segments_raises(self) -> None:
        """Version with more than 3 segments raises click.BadParameter."""
        with pytest.raises(click.BadParameter, match="at most 3 segments"):
            _build_introspection_payload(version="1.2.3.4")

    def test_negative_version_raises(self) -> None:
        """Negative version segment raises click.BadParameter."""
        with pytest.raises(click.BadParameter, match="must not be negative"):
            _build_introspection_payload(version="-1.0.0")

    def test_empty_version_raises(self) -> None:
        """Empty version string raises click.BadParameter."""
        with pytest.raises(click.BadParameter, match="must not be empty"):
            _build_introspection_payload(version="")

    def test_invalid_version_raises(self) -> None:
        """Non-numeric version string raises click.BadParameter."""
        with pytest.raises(click.BadParameter, match="Invalid version"):
            _build_introspection_payload(version="abc")
