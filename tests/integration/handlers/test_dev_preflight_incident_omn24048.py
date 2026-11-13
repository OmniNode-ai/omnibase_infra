# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev deploy that fails in preflight publishes a manifest and an incident (OMN-24048).

The guard's ``run_preflight`` is exercised end to end with the real
``DevRollEventsPublisher`` and ``ModelDevRollIncident``.  No docker and no
broker are required: the publisher's ``_ensure_topic`` and ``_produce`` are
monkeypatched to record sends, ``KAFKA_BOOTSTRAP_SERVERS`` points at a
placeholder, and the executor's ``preflight`` raises ``RuntimeError``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest
from deploy_agent import executor as executor_module
from deploy_agent import publisher as publisher_mod
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelRebuildRequested,
    Scope,
)

from omnibase_infra.models.model_dev_roll_incident import ModelDevRollIncident

pytestmark = pytest.mark.integration

TOPIC_MANIFEST = "onex.evt.omnimarket.dev-roll-manifest.v1"
TOPIC_INCIDENT = "onex.evt.omnimarket.dev-roll-incident.v1"


def _dev_cmd() -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="gha/omnimarket/pr-5942",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )


def _stability_cmd() -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="gha/omnimarket/pr-5942",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.STABILITY_TEST,
    )


def _wire_sender(
    monkeypatch: pytest.MonkeyPatch, *, sender_raises: bool = False
) -> list[tuple[str, str, dict[str, Any]]]:
    """Replace the publisher's ``_ensure_topic`` and ``_produce`` to record sends."""
    sent: list[tuple[str, str, dict[str, Any]]] = []

    def produce(topic: str, key: str, payload: dict[str, Any], config: object) -> bool:
        if sender_raises:
            raise OSError("broker down")
        sent.append((topic, key, payload))
        return True

    monkeypatch.setattr(publisher_mod, "_ensure_topic", lambda topic, config: None)
    monkeypatch.setattr(publisher_mod, "_produce", produce)
    return sent


def _set_tracking_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")


def test_preflight_failure_publishes_manifest_then_incident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_tracking_ref(monkeypatch)
    cmd = _dev_cmd()
    sent = _wire_sender(monkeypatch)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    executor = Mock()
    executor.preflight.side_effect = RuntimeError("Docker unavailable")

    with pytest.raises(RuntimeError, match="Docker unavailable"):
        executor_module.load_dev_boot_guard().run_preflight(executor, cmd, Mock())

    assert [topic for topic, _key, _payload in sent] == [TOPIC_MANIFEST, TOPIC_INCIDENT]
    incident = ModelDevRollIncident.model_validate(sent[1][2])
    assert incident.issuer == cmd.requested_by
    assert incident.failed_phase == "preflight"
    assert incident.recovery_verified is False


def test_preflight_failure_sender_error_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_tracking_ref(monkeypatch)
    cmd = _dev_cmd()
    sent = _wire_sender(monkeypatch, sender_raises=True)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    executor = Mock()
    executor.preflight.side_effect = RuntimeError("Docker unavailable")

    with pytest.raises(RuntimeError, match="Docker unavailable"):
        executor_module.load_dev_boot_guard().run_preflight(executor, cmd, Mock())

    assert sent == []


def test_stability_test_preflight_failure_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_tracking_ref(monkeypatch)
    cmd = _stability_cmd()
    sent = _wire_sender(monkeypatch)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    executor = Mock()
    executor.preflight.side_effect = RuntimeError("Docker unavailable")

    with pytest.raises(RuntimeError, match="Docker unavailable"):
        executor_module.load_dev_boot_guard().run_preflight(executor, cmd, Mock())

    assert sent == []
