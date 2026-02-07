# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for introspection event payload construction."""

from __future__ import annotations

from uuid import UUID

from omnibase_infra.cli.infra_test.introspect import _build_introspection_payload


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
        """Payload includes a timestamp."""
        payload = _build_introspection_payload()
        assert payload["timestamp"]

    def test_has_version_semver(self) -> None:
        """Payload includes node_version as semver dict."""
        payload = _build_introspection_payload()
        version = payload["node_version"]
        assert isinstance(version, dict)
        assert version["major"] == 1
        assert version["minor"] == 0
        assert version["patch"] == 0

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
