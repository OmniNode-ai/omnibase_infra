# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the platform capability probe (OMN-5265).

All network calls are mocked -- no real connections are made.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.probes.capability_probe import (
    http_health_check,
    kafka_reachable,
    probe_platform_tier,
    read_capabilities_cached,
    run_probe,
    socket_check,
    write_capabilities_atomic,
)


class TestSocketCheck:
    """Tests for socket_check()."""

    @patch("omnibase_infra.probes.capability_probe.socket.create_connection")
    def test_returns_true_on_success(self, mock_conn: MagicMock) -> None:
        mock_conn.return_value.__enter__ = MagicMock()
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        assert socket_check("localhost", 5432) is True

    @patch("omnibase_infra.probes.capability_probe.socket.create_connection")
    def test_returns_false_on_os_error(self, mock_conn: MagicMock) -> None:
        mock_conn.side_effect = OSError("Connection refused")
        assert socket_check("localhost", 5432) is False


class TestKafkaReachable:
    """Tests for kafka_reachable()."""

    @patch("omnibase_infra.probes.capability_probe.socket_check")
    def test_returns_true_when_one_host_reachable(self, mock_check: MagicMock) -> None:
        mock_check.side_effect = [False, True]
        assert kafka_reachable("bad:9092,good:9092") is True

    @patch("omnibase_infra.probes.capability_probe.socket_check")
    def test_returns_false_when_none_reachable(self, mock_check: MagicMock) -> None:
        mock_check.return_value = False
        assert kafka_reachable("bad:9092") is False

    def test_returns_false_for_empty_string(self) -> None:
        assert kafka_reachable("") is False

    def test_returns_false_for_whitespace(self) -> None:
        assert kafka_reachable("   ") is False

    def test_handles_malformed_entries(self) -> None:
        assert kafka_reachable("no-port,also-no-port") is False

    def test_handles_non_numeric_port(self) -> None:
        assert kafka_reachable("host:abc") is False


class TestHttpHealthCheck:
    """Tests for http_health_check()."""

    @patch("omnibase_infra.probes.capability_probe.urllib.request.urlopen")
    def test_returns_true_on_200(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value = mock_response
        assert http_health_check("http://localhost:8053/health") is True

    @patch("omnibase_infra.probes.capability_probe.urllib.request.urlopen")
    def test_returns_false_on_500(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 500
        mock_urlopen.return_value = mock_response
        assert http_health_check("http://localhost:8053/health") is False

    @patch("omnibase_infra.probes.capability_probe.urllib.request.urlopen")
    def test_returns_false_on_exception(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = Exception("Connection refused")
        assert http_health_check("http://localhost:8053/health") is False


class TestProbePlatformTier:
    """Tests for probe_platform_tier()."""

    @patch("omnibase_infra.probes.capability_probe.http_health_check")
    @patch("omnibase_infra.probes.capability_probe.kafka_reachable")
    def test_standalone_when_kafka_unreachable(
        self, mock_kafka: MagicMock, mock_http: MagicMock
    ) -> None:
        mock_kafka.return_value = False
        assert probe_platform_tier(kafka_servers="localhost:9092") == "standalone"
        mock_http.assert_not_called()

    @patch("omnibase_infra.probes.capability_probe.http_health_check")
    @patch("omnibase_infra.probes.capability_probe.kafka_reachable")
    def test_event_bus_when_kafka_up_intel_down(
        self, mock_kafka: MagicMock, mock_http: MagicMock
    ) -> None:
        mock_kafka.return_value = True
        mock_http.return_value = False
        assert probe_platform_tier(kafka_servers="localhost:9092") == "event_bus"

    @patch("omnibase_infra.probes.capability_probe.http_health_check")
    @patch("omnibase_infra.probes.capability_probe.kafka_reachable")
    def test_full_onex_when_both_up(
        self, mock_kafka: MagicMock, mock_http: MagicMock
    ) -> None:
        mock_kafka.return_value = True
        mock_http.return_value = True
        assert probe_platform_tier(kafka_servers="localhost:9092") == "full_onex"

    def test_standalone_with_empty_servers(self) -> None:
        assert probe_platform_tier(kafka_servers="") == "standalone"


class TestWriteCapabilitiesAtomic:
    """Tests for write_capabilities_atomic()."""

    def test_writes_json_to_file(self, tmp_path: Path) -> None:
        target = tmp_path / "caps.json"
        data: dict[str, object] = {
            "tier": "standalone",
            "probed_at": "2026-01-01T00:00:00+00:00",
        }
        write_capabilities_atomic(data, capabilities_file=target)
        assert target.exists()
        loaded = json.loads(target.read_text())
        assert loaded["tier"] == "standalone"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "subdir" / "caps.json"
        write_capabilities_atomic({"tier": "standalone"}, capabilities_file=target)
        assert target.exists()


class TestReadCapabilitiesCached:
    """Tests for read_capabilities_cached()."""

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "nonexistent.json"
        assert read_capabilities_cached(capabilities_file=target) is None

    def test_returns_data_when_fresh(self, tmp_path: Path) -> None:
        target = tmp_path / "caps.json"
        now = datetime.now(tz=UTC)
        data = {"tier": "event_bus", "probed_at": now.isoformat()}
        target.write_text(json.dumps(data))
        result = read_capabilities_cached(capabilities_file=target)
        assert result is not None
        assert result["tier"] == "event_bus"

    def test_returns_none_when_stale(self, tmp_path: Path) -> None:
        target = tmp_path / "caps.json"
        # 10 minutes ago -- exceeds 5 minute TTL
        old = datetime(2020, 1, 1, tzinfo=UTC)
        data = {"tier": "standalone", "probed_at": old.isoformat()}
        target.write_text(json.dumps(data))
        assert read_capabilities_cached(capabilities_file=target) is None

    def test_returns_none_when_missing_probed_at(self, tmp_path: Path) -> None:
        target = tmp_path / "caps.json"
        target.write_text(json.dumps({"tier": "standalone"}))
        assert read_capabilities_cached(capabilities_file=target) is None


class TestRunProbe:
    """Tests for run_probe()."""

    @patch("omnibase_infra.probes.capability_probe.probe_platform_tier")
    def test_writes_result_and_returns_tier(
        self, mock_probe: MagicMock, tmp_path: Path
    ) -> None:
        mock_probe.return_value = "event_bus"
        target = tmp_path / "caps.json"
        tier = run_probe(
            kafka_servers="localhost:9092",
            intelligence_url="http://localhost:8053",
            capabilities_file=target,
        )
        assert tier == "event_bus"
        assert target.exists()
        loaded = json.loads(target.read_text())
        assert loaded["tier"] == "event_bus"
        assert "probed_at" in loaded
