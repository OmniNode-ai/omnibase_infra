# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for runtime error classification and Kafka emission in monitor_logs.py.

Tests the RuntimeErrorEmitter and supporting classification functions added
in OMN-5649.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# monitor_logs.py is a script, not a package module. Add scripts/ to sys.path
# so we can import it directly.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide minimal env vars so module-level code doesn't fail."""
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C12345")


# ---- Classification tests ----


@pytest.mark.unit
class TestClassifyRuntimeError:
    """Test _classify_runtime_error categorisation regexes."""

    def test_schema_mismatch_column(self) -> None:
        from monitor_logs import _classify_runtime_error

        result = _classify_runtime_error(
            '2026-03-21 13:22:42 [ERROR] omnibase_infra.foo: column "pattern_name" does not exist'
        )
        assert result == "SCHEMA_MISMATCH"

    def test_schema_mismatch_relation(self) -> None:
        from monitor_logs import _classify_runtime_error

        result = _classify_runtime_error(
            '[ERROR] relation "node_registrations" does not exist'
        )
        assert result == "SCHEMA_MISMATCH"

    def test_missing_topic(self) -> None:
        from monitor_logs import _classify_runtime_error

        result = _classify_runtime_error(
            "[ERROR] MISSING_TOPIC: Required topic 'onex.evt.platform.feature-flag-changed.v1' not in broker"
        )
        assert result == "MISSING_TOPIC"

    def test_connection_error(self) -> None:
        from monitor_logs import _classify_runtime_error

        result = _classify_runtime_error(
            "[ERROR] ConnectionRefusedError: [Errno 61] Connection refused"
        )
        assert result == "CONNECTION"

    def test_timeout_error(self) -> None:
        from monitor_logs import _classify_runtime_error

        result = _classify_runtime_error(
            "[ERROR] TimeoutError: timed out waiting for response"
        )
        assert result == "TIMEOUT"

    def test_oom_error(self) -> None:
        from monitor_logs import _classify_runtime_error

        result = _classify_runtime_error(
            "[CRITICAL] MemoryError: Cannot allocate memory"
        )
        assert result == "OOM"

    def test_authentication_error(self) -> None:
        from monitor_logs import _classify_runtime_error

        result = _classify_runtime_error(
            "[ERROR] AuthenticationError: permission denied for user"
        )
        assert result == "AUTHENTICATION"

    def test_unknown_error(self) -> None:
        from monitor_logs import _classify_runtime_error

        result = _classify_runtime_error(
            "[ERROR] Something completely unexpected happened"
        )
        assert result == "UNKNOWN"


# ---- Fingerprint tests ----


@pytest.mark.unit
class TestComputeRuntimeFingerprint:
    """Test _compute_runtime_fingerprint produces consistent SHA-256 hashes."""

    def test_fingerprint_is_64_chars(self) -> None:
        from monitor_logs import _compute_runtime_fingerprint

        fp = _compute_runtime_fingerprint(
            "omninode-runtime", "SCHEMA_MISMATCH", "column does not exist"
        )
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_deterministic(self) -> None:
        from monitor_logs import _compute_runtime_fingerprint

        fp1 = _compute_runtime_fingerprint("container", "CONNECTION", "refused")
        fp2 = _compute_runtime_fingerprint("container", "CONNECTION", "refused")
        assert fp1 == fp2

    def test_different_inputs_different_fingerprint(self) -> None:
        from monitor_logs import _compute_runtime_fingerprint

        fp1 = _compute_runtime_fingerprint("container-a", "CONNECTION", "refused")
        fp2 = _compute_runtime_fingerprint("container-b", "CONNECTION", "refused")
        assert fp1 != fp2


# ---- Emitter tests ----


@pytest.mark.unit
class TestRuntimeErrorEmitter:
    """Test RuntimeErrorEmitter.maybe_emit produces Kafka events."""

    @patch("monitor_logs.RuntimeErrorEmitter._init_clients", return_value=True)
    def test_runtime_error_emitted_to_kafka(self, mock_init: MagicMock) -> None:
        """Verify that a container ERROR log line produces a Kafka event."""
        from monitor_logs import RuntimeErrorEmitter

        emitter = RuntimeErrorEmitter(dry_run=False)
        mock_producer = MagicMock()
        emitter._producer = mock_producer
        emitter._init_ok = True

        line = '2026-03-21 13:22:42 [ERROR] omnibase_infra.foo: column "pattern_name" does not exist'
        emitter.maybe_emit("omninode-runtime-effects", line)

        assert mock_producer.produce.called
        call_kwargs = mock_producer.produce.call_args
        assert call_kwargs[1]["topic"] == "onex.evt.omnibase-infra.runtime-error.v1"

        import json

        event = json.loads(call_kwargs[1]["value"].decode("utf-8"))
        assert event["error_category"] == "SCHEMA_MISMATCH"
        assert event["container"] == "omninode-runtime-effects"
        assert "fingerprint" in event
        assert len(event["fingerprint"]) == 64
        assert event["log_level"] == "ERROR"
        assert event["severity"] == "HIGH"

    @patch("monitor_logs.RuntimeErrorEmitter._init_clients", return_value=True)
    def test_missing_topic_extracts_topic_name(self, mock_init: MagicMock) -> None:
        """Verify that MISSING_TOPIC errors parse the topic name."""
        from monitor_logs import RuntimeErrorEmitter

        emitter = RuntimeErrorEmitter(dry_run=False)
        mock_producer = MagicMock()
        emitter._producer = mock_producer
        emitter._init_ok = True

        line = "[ERROR] MISSING_TOPIC: Required topic 'onex.evt.platform.feature-flag-changed.v1' not in broker"
        emitter.maybe_emit("omninode-runtime-effects", line)

        import json

        event = json.loads(mock_producer.produce.call_args[1]["value"].decode("utf-8"))
        assert event["error_category"] == "MISSING_TOPIC"
        assert (
            event["missing_topic_name"] == "onex.evt.platform.feature-flag-changed.v1"
        )

    @patch("monitor_logs.RuntimeErrorEmitter._init_clients", return_value=True)
    def test_schema_mismatch_extracts_relation_name(self, mock_init: MagicMock) -> None:
        """Verify that SCHEMA_MISMATCH errors parse the relation/column name."""
        from monitor_logs import RuntimeErrorEmitter

        emitter = RuntimeErrorEmitter(dry_run=False)
        mock_producer = MagicMock()
        emitter._producer = mock_producer
        emitter._init_ok = True

        line = '[ERROR] column "pattern_name" does not exist'
        emitter.maybe_emit("omninode-runtime", line)

        import json

        event = json.loads(mock_producer.produce.call_args[1]["value"].decode("utf-8"))
        assert event["error_category"] == "SCHEMA_MISMATCH"
        assert event["missing_relation_name"] == "pattern_name"

    @patch("monitor_logs.RuntimeErrorEmitter._init_clients", return_value=True)
    def test_oom_classified_as_critical(self, mock_init: MagicMock) -> None:
        """Verify that OOM errors get CRITICAL severity."""
        from monitor_logs import RuntimeErrorEmitter

        emitter = RuntimeErrorEmitter(dry_run=False)
        mock_producer = MagicMock()
        emitter._producer = mock_producer
        emitter._init_ok = True

        line = "[CRITICAL] MemoryError: Cannot allocate memory"
        emitter.maybe_emit("omninode-runtime", line)

        import json

        event = json.loads(mock_producer.produce.call_args[1]["value"].decode("utf-8"))
        assert event["error_category"] == "OOM"
        assert event["severity"] == "CRITICAL"
        assert event["log_level"] == "CRITICAL"

    @patch("monitor_logs.RuntimeErrorEmitter._init_clients", return_value=True)
    def test_dry_run_does_not_produce(self, mock_init: MagicMock) -> None:
        """Verify that dry_run mode does not call Kafka produce."""
        from monitor_logs import RuntimeErrorEmitter

        emitter = RuntimeErrorEmitter(dry_run=True)
        mock_producer = MagicMock()
        emitter._producer = mock_producer
        emitter._init_ok = True

        line = "[ERROR] some error"
        emitter.maybe_emit("container", line)

        assert not mock_producer.produce.called
