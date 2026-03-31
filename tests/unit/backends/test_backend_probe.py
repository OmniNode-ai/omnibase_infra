# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for backend health probes (OMN-7075).

Tests the 4-state probe model and probe functions for Kafka and Postgres
backends registered via onex.backends entry points.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.backends.backend_probe import (
    EnumProbeState,
    ModelProbeResult,
    probe_kafka,
    probe_postgres,
)

pytestmark = pytest.mark.unit


class TestModelProbeResult:
    """Test the ModelProbeResult Pydantic model."""

    def test_all_states_valid(self) -> None:
        """All 4 probe states can be constructed."""
        for state in EnumProbeState:
            result = ModelProbeResult(
                state=state,
                reason="test",
                backend_label="test_backend",
            )
            assert result.state == state
            assert result.reason == "test"

    def test_frozen_model(self) -> None:
        """ModelProbeResult is frozen (immutable)."""
        result = ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason="test",
            backend_label="test_backend",
        )
        with pytest.raises(Exception):
            result.state = EnumProbeState.HEALTHY  # type: ignore[misc]


class TestKafkaBackendDiscoverable:
    """Test Kafka backend discovery via entry points."""

    def test_kafka_backend_discoverable(self) -> None:
        """Kafka backend is discoverable via onex.backends entry point."""
        backends = entry_points(group="onex.backends")
        names = [ep.name for ep in backends]
        assert "event_bus_kafka" in names

    def test_state_postgres_discoverable(self) -> None:
        """Postgres backend is discoverable via onex.backends entry point."""
        backends = entry_points(group="onex.backends")
        names = [ep.name for ep in backends]
        assert "state_postgres" in names


class TestKafkaProbe:
    """Test Kafka probe function."""

    def test_no_bootstrap_servers_returns_discovered(self) -> None:
        """When KAFKA_BOOTSTRAP_SERVERS is not set, probe returns DISCOVERED."""
        with patch.dict("os.environ", {}, clear=True):
            result = probe_kafka(bootstrap_servers="")
            assert result.state == EnumProbeState.DISCOVERED
            assert result.backend_label == "event_bus_kafka"

    def test_unreachable_broker_returns_discovered(self) -> None:
        """When broker is unreachable, probe returns DISCOVERED."""
        result = probe_kafka(
            bootstrap_servers="localhost:59999",
            timeout=0.5,
        )
        assert result.state == EnumProbeState.DISCOVERED
        assert "TCP connect" in result.reason

    def test_invalid_broker_address_returns_discovered(self) -> None:
        """When broker address is unparseable, probe returns DISCOVERED."""
        result = probe_kafka(bootstrap_servers="no-port-here")
        assert result.state == EnumProbeState.DISCOVERED

    def test_reachable_but_topic_list_fails(self) -> None:
        """When TCP works but topic listing fails, returns REACHABLE."""
        with (
            patch(
                "omnibase_infra.backends.backend_probe._tcp_reachable",
                return_value=True,
            ),
            patch(
                "confluent_kafka.admin.AdminClient",
                side_effect=RuntimeError("Connection refused"),
            ),
        ):
            result = probe_kafka(bootstrap_servers="localhost:9092")
            assert result.state == EnumProbeState.REACHABLE

    def test_auth_failure_returns_reachable(self) -> None:
        """When auth fails, probe returns REACHABLE (not HEALTHY)."""
        with (
            patch(
                "omnibase_infra.backends.backend_probe._tcp_reachable",
                return_value=True,
            ),
            patch(
                "confluent_kafka.admin.AdminClient",
                side_effect=RuntimeError("SASL authentication failed"),
            ),
        ):
            result = probe_kafka(bootstrap_servers="localhost:9092")
            assert result.state == EnumProbeState.REACHABLE
            assert "Auth failure" in result.reason

    def test_healthy_with_broker_match_returns_authoritative(self) -> None:
        """When topics listed and brokers match, returns AUTHORITATIVE."""
        mock_broker = MagicMock()
        mock_broker.host = "localhost"

        mock_metadata = MagicMock()
        mock_metadata.topics = {"topic1": None, "topic2": None}
        mock_metadata.brokers = {0: mock_broker}

        mock_admin = MagicMock()
        mock_admin.list_topics.return_value = mock_metadata

        with (
            patch(
                "omnibase_infra.backends.backend_probe._tcp_reachable",
                return_value=True,
            ),
            patch(
                "confluent_kafka.admin.AdminClient",
                return_value=mock_admin,
            ),
        ):
            result = probe_kafka(bootstrap_servers="localhost:9092")
            assert result.state == EnumProbeState.AUTHORITATIVE


class TestPostgresProbe:
    """Test Postgres probe function."""

    def test_unreachable_returns_discovered(self) -> None:
        """When Postgres is not running, probe returns DISCOVERED."""
        result = probe_postgres(host="localhost", port=59999, timeout=0.5)
        assert result.state == EnumProbeState.DISCOVERED
        assert "TCP connect" in result.reason

    def test_auth_failure_returns_reachable(self) -> None:
        """When TCP works but auth fails, returns REACHABLE."""
        with patch(
            "omnibase_infra.backends.backend_probe._tcp_reachable",
            return_value=True,
        ):
            import psycopg2

            with patch(
                "psycopg2.connect",
                side_effect=psycopg2.OperationalError("password authentication failed"),
            ):
                result = probe_postgres(host="localhost", port=5432)
                assert result.state == EnumProbeState.REACHABLE
                assert "Auth failure" in result.reason

    def test_select_succeeds_missing_tables_returns_healthy(self) -> None:
        """When SELECT 1 works but tables missing, returns HEALTHY."""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        # First call: SELECT 1
        # Second call: schema check returns no tables
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.fetchall.return_value = []

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "omnibase_infra.backends.backend_probe._tcp_reachable",
                return_value=True,
            ),
            patch("psycopg2.connect", return_value=mock_conn),
        ):
            result = probe_postgres(
                host="localhost",
                port=5432,
                required_tables=("snapshots", "projections"),
            )
            assert result.state == EnumProbeState.HEALTHY
            assert "Missing required tables" in result.reason

    def test_all_tables_present_returns_authoritative(self) -> None:
        """When all required tables exist, returns AUTHORITATIVE."""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.fetchall.return_value = [
            ("snapshots",),
            ("projections",),
        ]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "omnibase_infra.backends.backend_probe._tcp_reachable",
                return_value=True,
            ),
            patch("psycopg2.connect", return_value=mock_conn),
        ):
            result = probe_postgres(
                host="localhost",
                port=5432,
                required_tables=("snapshots", "projections"),
            )
            assert result.state == EnumProbeState.AUTHORITATIVE
