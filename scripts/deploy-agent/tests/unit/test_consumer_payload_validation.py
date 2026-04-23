# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for signed deploy command validation in DeployConsumer."""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("kafka", types.SimpleNamespace(KafkaConsumer=object))

from deploy_agent.consumer import DeployConsumer
from deploy_agent.job_state import JobStore

_SECRET = "test-consumer-secret"


def _sign(envelope: dict) -> dict:
    body = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    sig = hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {**envelope, "_signature": sig}


class _FakeKafkaConsumer:
    def __init__(self) -> None:
        self.commit_calls = 0

    def commit(self) -> None:
        self.commit_calls += 1


def test_signed_trigger_payload_is_accepted(tmp_path) -> None:
    """Signed trigger envelopes must validate after auth strips transport metadata."""
    payload = _sign(
        {
            "correlation_id": "aaaaaaaa-0000-0000-0000-000000000001",
            "git_ref": "origin/main",
            "reason": "manual trigger by operator",
            "requested_at": "2026-04-21T12:00:00+00:00",
            "requested_by": "operator-manual",
            "scope": "runtime",
            "services": [],
        }
    )
    consumer = object.__new__(DeployConsumer)
    consumer.consumer = _FakeKafkaConsumer()
    consumer.job_store = JobStore(tmp_path)

    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": _SECRET}):
        cmd, reason = consumer._process_message(SimpleNamespace(value=payload))

    assert reason is None
    assert cmd is not None
    assert str(cmd.correlation_id) == payload["correlation_id"]
    assert cmd.reason == payload["reason"]
    assert cmd.requested_at is not None
    assert cmd.build_source.value == "release"

    stored = consumer.job_store.load(cmd.correlation_id)
    assert stored is not None
    assert "_signature" not in stored.command
    assert stored.command["reason"] == payload["reason"]
    assert consumer.consumer.commit_calls == 1
