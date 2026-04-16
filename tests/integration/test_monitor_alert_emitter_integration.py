# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for MonitorAlertEmitter (OMN-8885).

Verifies end-to-end integration aspects:
  - Topic name loaded from actual monitor_alert_contract.yaml file
  - Emitter initialization with real environment configuration
  - Graceful degradation behavior in realistic scenarios
  - Payload serialization and Kafka compatibility
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path for imports
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


@pytest.mark.integration
def test_topic_loaded_from_actual_contract_yaml() -> None:
    """Topic name must be loaded from monitor_alert_contract.yaml, not hardcoded."""
    from monitor_logs import _TOPIC_MONITOR_ALERT_V1

    # Verify the contract file exists
    contract_path = Path(_SCRIPTS_DIR) / "monitor_alert_contract.yaml"
    assert contract_path.exists(), (
        f"Contract file not found: {contract_path}. "
        "Topic names must be contract-driven, not hardcoded."
    )

    # Load the contract and verify topic matches
    import yaml

    with contract_path.open() as f:
        contract_data = yaml.safe_load(f)

    published_topics = contract_data.get("event_bus", {}).get("publish_topics", [])
    assert len(published_topics) > 0, (
        "Contract must declare at least one published topic"
    )

    expected_topic = published_topics[0]
    assert expected_topic == _TOPIC_MONITOR_ALERT_V1, (
        f"Topic name mismatch: code uses '{_TOPIC_MONITOR_ALERT_V1}' but "
        f"contract declares '{expected_topic}'. Update code to load from contract."
    )


@pytest.mark.integration
def test_emitter_initializes_with_kafka_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emitter should successfully initialize when KAFKA_BOOTSTRAP_SERVERS is set."""
    monkeypatch.setenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"
    )  # kafka-fallback-ok: test fixture

    mock_producer = MagicMock()
    with patch("confluent_kafka.Producer", return_value=mock_producer):
        from monitor_logs import MonitorAlertEmitter

        emitter = MonitorAlertEmitter(dry_run=False)
        assert emitter._init_ok, (
            "Emitter should initialize successfully with valid env vars"
        )
        assert emitter._producer is not None


@pytest.mark.integration
def test_emitter_disabled_when_kafka_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emitter should gracefully disable when KAFKA_BOOTSTRAP_SERVERS is unset."""
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

    from monitor_logs import MonitorAlertEmitter

    emitter = MonitorAlertEmitter(dry_run=False)
    assert not emitter._init_ok, "Emitter should self-disable when env var missing"
    assert emitter._producer is None


@pytest.mark.integration
def test_emit_produces_kafka_compatible_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emitted payload must be valid JSON that Kafka can serialize."""
    monkeypatch.setenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"
    )  # kafka-fallback-ok: test fixture

    mock_producer = MagicMock()
    with patch("confluent_kafka.Producer", return_value=mock_producer):
        from monitor_logs import MonitorAlertEmitter

        emitter = MonitorAlertEmitter(dry_run=False)
        emitter.emit(
            container="omninode-runtime",
            severity="ERROR",
            pattern_matched="RuntimeError",
            full_message_text="Full error message here",
            raw_log_excerpt="Raw log excerpt (first 500 chars)",
            exit_code=1,
            restart_count=2,
        )

    assert mock_producer.produce.called, "Producer should be called"
    call_kwargs = mock_producer.produce.call_args[1]

    # Verify payload is valid JSON
    payload_bytes = call_kwargs["value"]
    payload = json.loads(payload_bytes)  # Will raise if not valid JSON

    # Verify required fields for Kafka consumers
    assert "alert_id" in payload
    assert "detected_at" in payload
    assert "severity" in payload
    assert payload["severity"] == "ERROR"


@pytest.mark.integration
def test_emit_uses_correct_topic_from_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Producer must use the topic declared in monitor_alert_contract.yaml."""
    monkeypatch.setenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"
    )  # kafka-fallback-ok: test fixture

    # Load expected topic from contract
    contract_path = Path(_SCRIPTS_DIR) / "monitor_alert_contract.yaml"
    import yaml

    with contract_path.open() as f:
        contract_data = yaml.safe_load(f)
    expected_topic = contract_data["event_bus"]["publish_topics"][0]

    mock_producer = MagicMock()
    with patch("confluent_kafka.Producer", return_value=mock_producer):
        from monitor_logs import MonitorAlertEmitter

        emitter = MonitorAlertEmitter(dry_run=False)
        emitter.emit(
            container="test-container",
            severity="WARNING",
            pattern_matched="test",
            full_message_text="test",
            raw_log_excerpt="test",
        )

    # Verify produce() was called with the contract-declared topic
    call_kwargs = mock_producer.produce.call_args[1]
    assert call_kwargs["topic"] == expected_topic, (
        f"Producer must use topic from contract ({expected_topic}), "
        f"got: {call_kwargs['topic']}"
    )


@pytest.mark.integration
def test_dry_run_mode_does_not_call_producer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dry-run mode should print payload but not call producer.produce()."""
    monkeypatch.setenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"
    )  # kafka-fallback-ok: test fixture

    mock_producer = MagicMock()
    with patch("confluent_kafka.Producer", return_value=mock_producer):
        from monitor_logs import MonitorAlertEmitter

        emitter = MonitorAlertEmitter(dry_run=True)
        emitter.emit(
            container="test",
            severity="INFO",
            pattern_matched="test",
            full_message_text="test",
            raw_log_excerpt="test",
        )

    # Producer should not be called in dry-run mode
    assert not mock_producer.produce.called, "Dry-run should not call producer"

    # Should print payload to stdout
    captured = capsys.readouterr()
    assert "monitor-alert" in captured.out or "alert_id" in captured.out


@pytest.mark.integration
def test_emitter_with_real_kafka_config_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emitter should use production-ready Kafka config when initialized."""
    monkeypatch.setenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"
    )  # kafka-fallback-ok: test fixture

    captured_config = {}

    def capture_config(config: dict[str, str | int]) -> MagicMock:
        captured_config.update(config)
        return MagicMock()

    with patch("confluent_kafka.Producer", side_effect=capture_config):
        from monitor_logs import MonitorAlertEmitter

        MonitorAlertEmitter(dry_run=False)

    # Verify production-ready config
    assert (
        captured_config["bootstrap.servers"] == "localhost:19092"
    )  # kafka-fallback-ok: assertion
    assert captured_config["acks"] == "all"
    assert captured_config["enable.idempotence"] == "true"
    assert captured_config["retries"] == 5


@pytest.mark.integration
def test_emitter_handles_empty_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emitter should handle None/missing optional fields gracefully."""
    monkeypatch.setenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"
    )  # kafka-fallback-ok: test fixture

    mock_producer = MagicMock()
    with patch("confluent_kafka.Producer", return_value=mock_producer):
        from monitor_logs import MonitorAlertEmitter

        emitter = MonitorAlertEmitter(dry_run=False)
        # Emit with optional fields as None or default values
        emitter.emit(
            container="test",
            severity="WARNING",
            pattern_matched="test",
            full_message_text="test",
            raw_log_excerpt="test",
            exit_code=None,  # type: ignore[arg-type]
            restart_count=None,  # type: ignore[arg-type]
        )

    # Should not raise, should produce valid JSON
    assert mock_producer.produce.called
    payload = json.loads(mock_producer.produce.call_args[1]["value"])
    assert "alert_id" in payload
