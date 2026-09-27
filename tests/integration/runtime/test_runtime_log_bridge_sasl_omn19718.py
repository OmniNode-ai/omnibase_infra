# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the kernel runtime-log bridge SASL seam."""

from __future__ import annotations

from typing import ClassVar

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.observability.runtime_log_event_bridge import RuntimeLogEventBridge
from omnibase_infra.runtime.service_kernel import _create_runtime_log_bridge_producer

pytestmark = pytest.mark.integration


class _RecordingProducer:
    """Broker-free producer stand-in that records the kernel's constructor call."""

    constructor_calls: ClassVar[list[dict[str, object]]] = []
    started: ClassVar[list[_RecordingProducer]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        type(self).constructor_calls.append(kwargs)

    async def start(self) -> None:
        type(self).started.append(self)


@pytest.fixture(autouse=True)
def reset_recording_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the auth environment and the fake producer's observations."""
    import aiokafka

    _RecordingProducer.constructor_calls = []
    _RecordingProducer.started = []
    monkeypatch.setattr(aiokafka, "AIOKafkaProducer", _RecordingProducer)
    for name in (
        "KAFKA_SECURITY_PROTOCOL",
        "KAFKA_SASL_MECHANISM",
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_enabled_bridge_constructs_a_sasl_authenticated_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The enabled kernel bridge receives the runtime SASL client settings."""
    monkeypatch.setenv("ENABLE_RUNTIME_LOG_BRIDGE", "true")
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "runtime-log-bridge")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", "test-password")

    assert RuntimeLogEventBridge.is_enabled()
    producer = await _create_runtime_log_bridge_producer("lab-broker:9092")

    assert isinstance(producer, _RecordingProducer)
    assert _RecordingProducer.constructor_calls == [
        {
            "bootstrap_servers": "lab-broker:9092",
            "security_protocol": "SASL_PLAINTEXT",
            "sasl_mechanism": "SCRAM-SHA-256",
            "sasl_plain_username": "runtime-log-bridge",
            "sasl_plain_password": "test-password",
        }
    ]
    assert _RecordingProducer.started == [producer]


@pytest.mark.asyncio
async def test_disabled_bridge_does_not_construct_a_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bootstrap guard leaves the dedicated producer unconstructed when off."""
    monkeypatch.setenv("ENABLE_RUNTIME_LOG_BRIDGE", "false")

    if RuntimeLogEventBridge.is_enabled():
        await _create_runtime_log_bridge_producer("lab-broker:9092")

    assert _RecordingProducer.constructor_calls == []
    assert _RecordingProducer.started == []


@pytest.mark.asyncio
async def test_declared_sasl_without_password_fails_before_producer_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete SASL declaration cannot fall back to an unauthenticated client."""
    monkeypatch.setenv("ENABLE_RUNTIME_LOG_BRIDGE", "true")
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "runtime-log-bridge")

    with pytest.raises(
        ProtocolConfigurationError,
        match="requires non-empty credential fields: sasl_plain_password",
    ):
        await _create_runtime_log_bridge_producer("lab-broker:9092")

    assert _RecordingProducer.constructor_calls == []
    assert _RecordingProducer.started == []
