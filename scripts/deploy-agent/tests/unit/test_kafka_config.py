# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deploy-agent Kafka config must use one explicit, DECLARED control bus.

OMN-18012: the transport half of the bus config is a lane DECLARATION, exactly
as the CI publishers read it out of the checked-in overlay (omnimarket
``config/ci_bus_lanes.yaml`` keys ``security_protocol`` / ``sasl_mechanism``).
The deploy agent's equivalent declaration surface is its systemd unit, so the
same two names are read from the environment and never derived from the shape
of it. Credential PRESENCE is not a statement about transport.
"""

from __future__ import annotations

import pytest
from deploy_agent.kafka_config import (
    ENV_SASL_ENV_PREFIX,
    ENV_SASL_MECHANISM,
    ENV_SASL_PASSWORD,
    ENV_SASL_USERNAME,
    ENV_SECURITY_PROTOCOL,
    ModelDeployAgentKafkaConfig,
    load_deploy_agent_kafka_config_from_env,
)
from pydantic import ValidationError

_TEST_PASSWORD = "test-secret"

_CREDENTIAL_ENV_NAMES = (
    ENV_SASL_USERNAME,
    ENV_SASL_PASSWORD,
    f"DEV_{ENV_SASL_USERNAME}",
    f"DEV_{ENV_SASL_PASSWORD}",
)


@pytest.fixture(autouse=True)
def _clear_bus_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every case from an undeclared bus, never the developer's shell."""
    for name in (
        "KAFKA_BOOTSTRAP_SERVERS",
        ENV_SECURITY_PROTOCOL,
        ENV_SASL_MECHANISM,
        ENV_SASL_ENV_PREFIX,
        *_CREDENTIAL_ENV_NAMES,
    ):
        monkeypatch.delenv(name, raising=False)


def test_missing_bootstrap_servers_fails_without_localhost_fallback() -> None:
    with pytest.raises(RuntimeError, match="no localhost fallback"):
        load_deploy_agent_kafka_config_from_env()


def test_plaintext_config_uses_explicit_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:9092")
    monkeypatch.setenv(ENV_SECURITY_PROTOCOL, "PLAINTEXT")

    config = load_deploy_agent_kafka_config_from_env()

    assert config.bootstrap_servers == "redpanda.example:9092"
    assert config.security_protocol == "PLAINTEXT"
    assert config.consumer_kwargs() == {
        "bootstrap_servers": "redpanda.example:9092",
        "security_protocol": "PLAINTEXT",
    }


def test_sasl_ssl_config_uses_kafka_python_cloud_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "pkc.example.confluent.cloud:9092")
    monkeypatch.setenv(ENV_SECURITY_PROTOCOL, "SASL_SSL")
    monkeypatch.setenv(ENV_SASL_MECHANISM, "PLAIN")
    monkeypatch.setenv(ENV_SASL_USERNAME, "key")
    monkeypatch.setenv(ENV_SASL_PASSWORD, _TEST_PASSWORD)

    config = load_deploy_agent_kafka_config_from_env()

    assert config.security_protocol == "SASL_SSL"
    assert config.consumer_kwargs() == {
        "bootstrap_servers": "pkc.example.confluent.cloud:9092",
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "PLAIN",
        "sasl_plain_username": "key",
        "sasl_plain_password": _TEST_PASSWORD,
    }
    # OMN-18057: the producer adds one delivery guarantee the consumer has no
    # notion of. It is asserted here rather than left as "producer == consumer",
    # because that equality is exactly what allowed the duplicate terminal
    # events measured on the dev bus on 2026-09-08 (offsets 96-101, 102-105).
    assert config.producer_kwargs() == {
        **config.consumer_kwargs(),
        "enable_idempotence": True,
    }
    assert "enable_idempotence" not in config.consumer_kwargs()


def test_dev_lane_sasl_plaintext_scram_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live .201 dev control bus: SASL over PLAINTEXT, SCRAM-SHA-256.

    Authenticated, NOT encrypted — there is no TLS on :19092. This is the exact
    shape the agent bootstrap-looped against while its config admitted only
    PLAINTEXT and SASL_SSL/PLAIN.
    """
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:19092")
    monkeypatch.setenv(ENV_SECURITY_PROTOCOL, "SASL_PLAINTEXT")
    monkeypatch.setenv(ENV_SASL_MECHANISM, "SCRAM-SHA-256")
    monkeypatch.setenv(ENV_SASL_USERNAME, "deploy-agent")
    monkeypatch.setenv(ENV_SASL_PASSWORD, _TEST_PASSWORD)

    config = load_deploy_agent_kafka_config_from_env()

    assert config.consumer_kwargs() == {
        "bootstrap_servers": "redpanda.example:19092",
        "security_protocol": "SASL_PLAINTEXT",
        "sasl_mechanism": "SCRAM-SHA-256",
        "sasl_plain_username": "deploy-agent",
        "sasl_plain_password": _TEST_PASSWORD,
    }


@pytest.mark.parametrize("protocol", ["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"])
def test_every_librdkafka_security_protocol_is_declarable(
    monkeypatch: pytest.MonkeyPatch, protocol: str
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:9092")
    monkeypatch.setenv(ENV_SECURITY_PROTOCOL, protocol)
    if protocol.startswith("SASL_"):
        monkeypatch.setenv(ENV_SASL_MECHANISM, "SCRAM-SHA-256")
        monkeypatch.setenv(ENV_SASL_USERNAME, "deploy-agent")
        monkeypatch.setenv(ENV_SASL_PASSWORD, _TEST_PASSWORD)

    assert load_deploy_agent_kafka_config_from_env().security_protocol == protocol


@pytest.mark.parametrize("mechanism", ["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"])
def test_scram_mechanisms_are_declarable(
    monkeypatch: pytest.MonkeyPatch, mechanism: str
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:19092")
    monkeypatch.setenv(ENV_SECURITY_PROTOCOL, "SASL_PLAINTEXT")
    monkeypatch.setenv(ENV_SASL_MECHANISM, mechanism)
    monkeypatch.setenv(ENV_SASL_USERNAME, "deploy-agent")
    monkeypatch.setenv(ENV_SASL_PASSWORD, _TEST_PASSWORD)

    config = load_deploy_agent_kafka_config_from_env()

    assert config.sasl_mechanism == mechanism
    assert config.consumer_kwargs()["sasl_mechanism"] == mechanism


def test_transport_is_never_inferred_from_credential_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credentials in the environment must not select a protocol.

    The pre-OMN-18012 loader defaulted to SASL_SSL whenever credentials were
    present, which picked TLS against a listener that speaks none. An
    undeclared transport is now a refusal, not a guess.
    """
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:19092")
    monkeypatch.setenv(ENV_SASL_USERNAME, "deploy-agent")
    monkeypatch.setenv(ENV_SASL_PASSWORD, _TEST_PASSWORD)

    with pytest.raises(RuntimeError, match=ENV_SECURITY_PROTOCOL):
        load_deploy_agent_kafka_config_from_env()


def test_undeclared_security_protocol_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:9092")

    with pytest.raises(RuntimeError, match="Refusing to guess the transport"):
        load_deploy_agent_kafka_config_from_env()


def test_unknown_security_protocol_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:9092")
    monkeypatch.setenv(ENV_SECURITY_PROTOCOL, "SASL_TLS")

    with pytest.raises(ValidationError, match="librdkafka security protocol"):
        load_deploy_agent_kafka_config_from_env()


def test_sasl_lane_without_credentials_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:19092")
    monkeypatch.setenv(ENV_SECURITY_PROTOCOL, "SASL_PLAINTEXT")
    monkeypatch.setenv(ENV_SASL_MECHANISM, "SCRAM-SHA-256")

    with pytest.raises(ValidationError, match=ENV_SASL_USERNAME):
        load_deploy_agent_kafka_config_from_env()


def test_sasl_protocol_requires_a_declared_mechanism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:19092")
    monkeypatch.setenv(ENV_SECURITY_PROTOCOL, "SASL_PLAINTEXT")
    monkeypatch.setenv(ENV_SASL_USERNAME, "deploy-agent")
    monkeypatch.setenv(ENV_SASL_PASSWORD, _TEST_PASSWORD)

    with pytest.raises(ValidationError, match=ENV_SASL_MECHANISM):
        load_deploy_agent_kafka_config_from_env()


def test_mechanism_beside_non_sasl_protocol_is_a_contradiction() -> None:
    with pytest.raises(ValidationError, match="carries no SASL"):
        ModelDeployAgentKafkaConfig(
            bootstrap_servers="redpanda.example:9092",
            security_protocol="PLAINTEXT",
            sasl_mechanism="SCRAM-SHA-256",
        )


def test_credentials_beside_non_sasl_protocol_are_rejected() -> None:
    with pytest.raises(ValidationError, match=ENV_SECURITY_PROTOCOL):
        ModelDeployAgentKafkaConfig(
            bootstrap_servers="pkc.example.confluent.cloud:9092",
            security_protocol="PLAINTEXT",
            sasl_username="key",
            sasl_password=_TEST_PASSWORD,
        )


def test_half_a_credential_pair_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must be set together"):
        ModelDeployAgentKafkaConfig(
            bootstrap_servers="redpanda.example:19092",
            security_protocol="SASL_PLAINTEXT",
            sasl_mechanism="SCRAM-SHA-256",
            sasl_username="deploy-agent",
        )


def test_declared_env_prefix_selects_the_lane_scoped_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The .201 operator env file holds DEV_-prefixed SCRAM credentials.

    systemd cannot expand one Environment= value into another, so the unit
    DECLARES which prefix its credentials live under rather than duplicating a
    secret into the unit file.
    """
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:19092")
    monkeypatch.setenv(ENV_SECURITY_PROTOCOL, "SASL_PLAINTEXT")
    monkeypatch.setenv(ENV_SASL_MECHANISM, "SCRAM-SHA-256")
    monkeypatch.setenv(ENV_SASL_ENV_PREFIX, "DEV_")
    monkeypatch.setenv(f"DEV_{ENV_SASL_USERNAME}", "dev-lane-principal")
    monkeypatch.setenv(f"DEV_{ENV_SASL_PASSWORD}", _TEST_PASSWORD)

    config = load_deploy_agent_kafka_config_from_env()

    assert config.sasl_username == "dev-lane-principal"
    assert config.sasl_password == _TEST_PASSWORD


def test_declared_env_prefix_does_not_fall_back_to_unprefixed_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared prefix is the whole statement; silence is not a fallback."""
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:19092")
    monkeypatch.setenv(ENV_SECURITY_PROTOCOL, "SASL_PLAINTEXT")
    monkeypatch.setenv(ENV_SASL_MECHANISM, "SCRAM-SHA-256")
    monkeypatch.setenv(ENV_SASL_ENV_PREFIX, "DEV_")
    monkeypatch.setenv(ENV_SASL_USERNAME, "wrong-lane-principal")
    monkeypatch.setenv(ENV_SASL_PASSWORD, _TEST_PASSWORD)

    with pytest.raises(ValidationError, match=f"DEV_{ENV_SASL_USERNAME}"):
        load_deploy_agent_kafka_config_from_env()
