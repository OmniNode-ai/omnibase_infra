# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for MonitorAlertEmitter (OMN-8885).

Covers:
- Payload shape: all required fields present in the emitted event
- Graceful degradation: no confluent-kafka → emitter disabled, no raise
- Graceful degradation: KAFKA_BOOTSTRAP_SERVERS unset → emitter disabled
- Dry-run: prints payload instead of calling producer
- Both error and warning severity paths emit to Kafka
- Topic name is loaded from contract YAML, not hardcoded
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C12345")


def _make_mock_producer() -> MagicMock:
    producer = MagicMock()
    producer.produce = MagicMock()
    producer.flush = MagicMock(return_value=0)
    return producer


@pytest.mark.unit
class TestMonitorAlertEmitterPayloadShape:
    """The emitted event must contain all required OMN-8885 fields."""

    def test_payload_contains_required_fields(self) -> None:
        from monitor_logs import MonitorAlertEmitter

        mock_producer = _make_mock_producer()
        with patch("confluent_kafka.Producer", return_value=mock_producer):
            emitter = MonitorAlertEmitter(dry_run=False)
            emitter.emit(
                container="omninode-runtime",
                severity="ERROR",
                pattern_matched="ERROR_PATTERN",
                full_message_text="RuntimeError: something failed",
                raw_log_excerpt="RuntimeError: something failed",
                exit_code=1,
                restart_count=3,
            )

        assert mock_producer.produce.called
        call_kwargs: dict[str, Any] = mock_producer.produce.call_args[1]
        payload = json.loads(call_kwargs["value"])

        required_fields = {
            "alert_id",
            "source",
            "severity",
            "pattern_matched",
            "container",
            "exit_code",
            "restart_count",
            "full_message_text",
            "raw_log_excerpt",
            "detected_at",
            "host",
        }
        assert required_fields <= set(payload.keys()), (
            f"Missing fields: {required_fields - set(payload.keys())}"
        )

    def test_payload_field_values(self) -> None:
        from monitor_logs import MonitorAlertEmitter

        mock_producer = _make_mock_producer()
        with patch("confluent_kafka.Producer", return_value=mock_producer):
            emitter = MonitorAlertEmitter(dry_run=False)
            emitter.emit(
                container="omninode-runner-1",
                severity="CRITICAL",
                pattern_matched="OOMKilled",
                full_message_text="OOMKilled: container exceeded memory limit",
                raw_log_excerpt="OOMKilled",
                exit_code=137,
                restart_count=5,
            )

        payload = json.loads(mock_producer.produce.call_args[1]["value"])
        assert payload["container"] == "omninode-runner-1"
        assert payload["source"] == "omninode-runner-1"
        assert payload["severity"] == "CRITICAL"
        assert payload["pattern_matched"] == "OOMKilled"
        assert payload["exit_code"] == 137
        assert payload["restart_count"] == 5
        assert payload["alert_id"]  # non-empty UUID string
        assert payload["detected_at"]  # ISO8601 string


@pytest.mark.unit
class TestMonitorAlertEmitterGracefulDegrade:
    """If confluent-kafka is absent or env var missing, emitter disables cleanly."""

    def test_no_confluent_kafka_disables_emitter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")

        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
            if name == "confluent_kafka":
                raise ImportError("No module named 'confluent_kafka'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from monitor_logs import MonitorAlertEmitter

            emitter = MonitorAlertEmitter(dry_run=False)
            assert not emitter._init_ok
            # Must not raise even though producer is None
            emitter.emit(
                container="test",
                severity="ERROR",
                pattern_matched="x",
                full_message_text="x",
                raw_log_excerpt="x",
            )

    def test_missing_kafka_bootstrap_servers_disables_emitter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        from monitor_logs import MonitorAlertEmitter

        with patch("confluent_kafka.Producer") as mock_producer_cls:
            emitter = MonitorAlertEmitter(dry_run=False)
            mock_producer_cls.assert_not_called()
            assert not emitter._init_ok
            # Must not raise even with no producer
            emitter.emit(
                container="test",
                severity="ERROR",
                pattern_matched="x",
                full_message_text="x",
                raw_log_excerpt="x",
            )


@pytest.mark.unit
class TestMonitorAlertEmitterDryRun:
    """Dry-run prints payload without calling Kafka producer."""

    def test_dry_run_prints_and_does_not_produce(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from monitor_logs import MonitorAlertEmitter

        mock_producer = _make_mock_producer()
        with patch("confluent_kafka.Producer", return_value=mock_producer):
            emitter = MonitorAlertEmitter(dry_run=True)
            emitter.emit(
                container="omninode-runtime",
                severity="WARNING",
                pattern_matched="stale-registration",
                full_message_text="Heartbeat received for non-active node",
                raw_log_excerpt="Heartbeat received",
            )

        mock_producer.produce.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "omninode-runtime" in captured.out


@pytest.mark.unit
class TestMonitorAlertEmitterTopicFromContract:
    """Topic name must be loaded from monitor_alert_contract.yaml, not hardcoded."""

    def test_topic_constant_matches_contract_yaml(self) -> None:
        import yaml

        contract_path = Path(_SCRIPTS_DIR) / "monitor_alert_contract.yaml"
        assert contract_path.exists(), f"Contract YAML missing: {contract_path}"

        data = yaml.safe_load(contract_path.read_text())
        declared_topics = data.get("event_bus", {}).get("publish_topics", [])
        assert declared_topics, (
            "No publish_topics declared in monitor_alert_contract.yaml"
        )

        from monitor_logs import _TOPIC_MONITOR_ALERT_V1

        assert declared_topics[0] == _TOPIC_MONITOR_ALERT_V1, (
            f"_TOPIC_MONITOR_ALERT_V1={_TOPIC_MONITOR_ALERT_V1!r} "
            f"does not match contract YAML topic={declared_topics[0]!r}"
        )

    def test_topic_used_in_produce_call(self) -> None:
        from monitor_logs import _TOPIC_MONITOR_ALERT_V1, MonitorAlertEmitter

        mock_producer = _make_mock_producer()
        with patch("confluent_kafka.Producer", return_value=mock_producer):
            emitter = MonitorAlertEmitter(dry_run=False)
            emitter.emit(
                container="test-container",
                severity="ERROR",
                pattern_matched="ERROR_PATTERN",
                full_message_text="some error",
                raw_log_excerpt="some error",
            )

        actual_topic = mock_producer.produce.call_args[1]["topic"]
        assert actual_topic == _TOPIC_MONITOR_ALERT_V1


@pytest.mark.unit
class TestContainerTailerDualEmit:
    """Both _maybe_alert and _maybe_warning_alert call MonitorAlertEmitter.emit."""

    def test_maybe_alert_calls_alert_emitter(self) -> None:
        import time

        from monitor_logs import ContainerTailer

        mock_emitter = MagicMock()
        stop = __import__("threading").Event()

        with (
            patch("monitor_logs._cooldown_read", return_value=(0.0, 0)),
            patch("monitor_logs._cooldown_write"),
            patch("monitor_logs.post_slack"),
        ):
            tailer = ContainerTailer(
                container="test-container",
                bot_token="xoxb-test",
                channel_id="C12345",
                cooldown=300,
                dry_run=False,
                stop_event=stop,
                alert_emitter=mock_emitter,
            )
            tailer._maybe_alert(["[ERROR] something bad happened"])

        mock_emitter.emit.assert_called_once()
        kwargs = mock_emitter.emit.call_args[1]
        assert kwargs["severity"] == "ERROR"
        assert kwargs["container"] == "test-container"

    def test_maybe_warning_alert_calls_alert_emitter(self) -> None:
        from monitor_logs import ContainerTailer

        mock_emitter = MagicMock()
        stop = __import__("threading").Event()

        with (
            patch("monitor_logs._cooldown_read", return_value=(0.0, 0)),
            patch("monitor_logs._cooldown_write"),
            patch("monitor_logs.post_slack"),
        ):
            tailer = ContainerTailer(
                container="test-container",
                bot_token="xoxb-test",
                channel_id="C12345",
                cooldown=300,
                dry_run=False,
                stop_event=stop,
                alert_emitter=mock_emitter,
            )
            tailer._maybe_warning_alert(
                "stale-registration", ["[WARNING] Heartbeat for non-active"]
            )

        mock_emitter.emit.assert_called_once()
        kwargs = mock_emitter.emit.call_args[1]
        assert kwargs["severity"] == "WARNING"
        assert kwargs["pattern_matched"] == "stale-registration"
