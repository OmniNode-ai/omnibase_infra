# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18012: the synchronous confluent clients must honour the lane transport.

``build_aiokafka_auth_kwargs`` already resolved transport + credentials for
the aiokafka data plane. The synchronous ``confluent_kafka`` admin clients --
the runtime health monitor's ``consumer_coverage`` probe and the backend
Kafka probe -- were still constructed with ``bootstrap.servers`` and two
timeouts only, i.e. implicit PLAINTEXT.

Live evidence (dev lane, 2026-09-08T06:35Z): the aiokafka data plane in
``omninode-runtime`` authenticated to the SASL_PLAINTEXT / SCRAM-SHA-256
broker from ``KAFKA_SECURITY_PROTOCOL`` / ``KAFKA_SASL_*`` while the admin
client in the same process failed with ``Admin client error:
InfraConnectionError`` on every cycle, so ``/health`` reported degraded and
docker marked ``omninode-runtime``, ``omninode-runtime-effects`` and
``omnibase-infra-runtime-worker-1`` unhealthy.

All credentials in this module are synthetic test constants.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.kafka_auth import (
    build_confluent_auth_config,
    build_confluent_auth_config_from_env,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

# Synthetic, never a real credential.
_SYNTHETIC_USER = "omn18012-synthetic-user"
_SYNTHETIC_PASSWORD = "omn18012-synthetic-password"

_SCRAM_MECHANISMS = ("PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512")


def _sasl_config(mechanism: str, **overrides: object) -> ModelKafkaEventBusConfig:
    kwargs: dict[str, object] = {
        "bootstrap_servers": "localhost:19092",
        "security_protocol": "SASL_PLAINTEXT",
        "sasl_mechanism": mechanism,
        "sasl_plain_username": _SYNTHETIC_USER,
        "sasl_plain_password": _SYNTHETIC_PASSWORD,
    }
    kwargs.update(overrides)
    return ModelKafkaEventBusConfig(**kwargs)


def _clear_kafka_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "KAFKA_SECURITY_PROTOCOL",
        "KAFKA_SASL_MECHANISM",
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
        "KAFKA_SSL_CA_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
@pytest.mark.parametrize("mechanism", _SCRAM_MECHANISMS)
def test_sasl_lane_carries_transport_and_credentials(mechanism: str) -> None:
    """A SASL declaration yields the four librdkafka auth entries."""
    entries = build_confluent_auth_config(_sasl_config(mechanism))

    assert entries["security.protocol"] == "SASL_PLAINTEXT"
    assert entries["sasl.mechanism"] == mechanism
    assert entries["sasl.username"] == _SYNTHETIC_USER
    assert entries["sasl.password"] == _SYNTHETIC_PASSWORD


@pytest.mark.unit
def test_plaintext_lane_omits_every_auth_entry() -> None:
    """A PLAINTEXT lane resolves to no entries, so its construction is unchanged."""
    config = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:19092",
        security_protocol="PLAINTEXT",
    )

    assert build_confluent_auth_config(config) == {}


@pytest.mark.unit
def test_plaintext_env_omits_every_auth_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no SASL env declared, the env builder resolves to no entries."""
    _clear_kafka_env(monkeypatch)

    assert build_confluent_auth_config_from_env() == {}


@pytest.mark.unit
def test_env_declaration_reaches_the_confluent_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dev-lane env vars resolve to the librdkafka auth entries."""
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
    monkeypatch.setenv("KAFKA_SASL_USERNAME", _SYNTHETIC_USER)
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _SYNTHETIC_PASSWORD)

    entries = build_confluent_auth_config_from_env()

    assert entries["security.protocol"] == "SASL_PLAINTEXT"
    assert entries["sasl.mechanism"] == "SCRAM-SHA-256"
    assert entries["sasl.username"] == _SYNTHETIC_USER
    assert entries["sasl.password"] == _SYNTHETIC_PASSWORD


@pytest.mark.unit
def test_missing_credential_is_refused_and_does_not_echo_the_password() -> None:
    """A mechanism without credentials fails loudly; the message carries no secret.

    ``ModelKafkaEventBusConfig`` already refuses this shape at validation, so
    the builder's own guard is defence in depth for a config assembled without
    validation. ``model_construct`` is how that path is reached deliberately.
    """
    config = ModelKafkaEventBusConfig.model_construct(
        bootstrap_servers="localhost:19092",
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism="SCRAM-SHA-256",
        sasl_plain_username=None,
        sasl_plain_password=_SYNTHETIC_PASSWORD,
    )

    with pytest.raises(ProtocolConfigurationError) as excinfo:
        build_confluent_auth_config(config)

    assert _SYNTHETIC_PASSWORD not in str(excinfo.value)
    assert _SYNTHETIC_PASSWORD not in repr(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize("mechanism", ["OAUTHBEARER", "AWS_MSK_IAM"])
def test_token_callback_mechanisms_are_refused_not_downgraded(mechanism: str) -> None:
    """A token-callback mechanism raises rather than silently opening PLAINTEXT."""
    overrides: dict[str, object] = {
        "sasl_mechanism": mechanism,
        "sasl_plain_username": None,
        "sasl_plain_password": None,
        "security_protocol": "SASL_SSL",
        "sasl_oauthbearer_token_endpoint_url": "https://example.invalid/token",
        "sasl_oauthbearer_client_id": "synthetic-client-id",
        "sasl_oauthbearer_client_secret": "synthetic-client-secret",
    }
    config = _sasl_config(mechanism, **overrides)

    with pytest.raises(ProtocolConfigurationError):
        build_confluent_auth_config(config)


class _FakeGroup:
    def __init__(self, group_id: str, state: str) -> None:
        self.group_id = group_id
        self.state = state


class _FakeResult:
    errors: tuple[object, ...] = ()

    def __init__(self, groups: list[_FakeGroup]) -> None:
        self.valid = groups


class _FakeFuture:
    def __init__(self, result: _FakeResult) -> None:
        self._result = result

    def result(self, timeout: float) -> _FakeResult:
        return self._result


class _RecordingAdminClient:
    """Captures the config dict the health monitor hands to confluent-kafka."""

    seen_config: dict[str, Any] = {}

    def __init__(self, config: dict[str, Any]) -> None:
        _RecordingAdminClient.seen_config = dict(config)

    def list_consumer_groups(self, request_timeout: float) -> _FakeFuture:
        return _FakeFuture(_FakeResult([_FakeGroup("group-a", "STABLE")]))


@pytest.fixture
def _stub_confluent_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a confluent_kafka.admin stub — the real package is not a test dep."""
    module = types.ModuleType("confluent_kafka.admin")
    module.AdminClient = _RecordingAdminClient  # type: ignore[attr-defined]
    parent = sys.modules.get("confluent_kafka") or types.ModuleType("confluent_kafka")
    monkeypatch.setitem(sys.modules, "confluent_kafka", parent)
    monkeypatch.setitem(sys.modules, "confluent_kafka.admin", module)


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_confluent_admin")
def test_health_monitor_admin_client_carries_the_sasl_entries(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The runtime health monitor's admin client is built for the declared lane."""
    from omnibase_infra.services.service_runtime_health_monitor import (
        _list_consumer_group_snapshots,
    )

    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
    monkeypatch.setenv("KAFKA_SASL_USERNAME", _SYNTHETIC_USER)
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _SYNTHETIC_PASSWORD)

    with caplog.at_level(logging.DEBUG):
        snapshots = _list_consumer_group_snapshots("redpanda:9092", 5000)

    assert [snapshot.group_id for snapshot in snapshots] == ["group-a"]

    config = _RecordingAdminClient.seen_config
    assert config["bootstrap.servers"] == "redpanda:9092"
    assert config["security.protocol"] == "SASL_PLAINTEXT"
    assert config["sasl.mechanism"] == "SCRAM-SHA-256"
    assert config["sasl.username"] == _SYNTHETIC_USER
    assert config["sasl.password"] == _SYNTHETIC_PASSWORD

    # The password is threaded to the client and nowhere else: it must not
    # reach a log record or the repr of anything the probe returns.
    assert _SYNTHETIC_PASSWORD not in caplog.text
    assert _SYNTHETIC_PASSWORD not in repr(snapshots)


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_confluent_admin")
def test_health_monitor_admin_client_is_plaintext_on_a_plaintext_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PLAINTEXT lane keeps exactly the three pre-existing config entries."""
    from omnibase_infra.services.service_runtime_health_monitor import (
        _list_consumer_group_snapshots,
    )

    _clear_kafka_env(monkeypatch)

    _list_consumer_group_snapshots("redpanda:9092", 5000)

    assert set(_RecordingAdminClient.seen_config) == {
        "bootstrap.servers",
        "socket.timeout.ms",
        "request.timeout.ms",
    }
