# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18012 gate 0 (prerequisite): SASL PLAIN / SCRAM-SHA-* credential threading.

``ModelKafkaEventBusConfig.sasl_mechanism`` has always accepted
``PLAIN``/``SCRAM-SHA-256``/``SCRAM-SHA-512`` (the field pattern lists them),
but ``build_aiokafka_auth_kwargs`` handled only ``PLAINTEXT`` (returns ``{}``),
``OAUTHBEARER`` and ``AWS_MSK_IAM``. A client configured for a
username/password broker was therefore constructed with a mechanism and no
credentials -- the platform could not authenticate to a username/password
broker at all. That is the blocker for the OMN-18012 customer-path golden
chain, whose hermetic Redpanda listener is SCRAM.

RESIDUAL, stated because the harness these tests unblock is *not* the
production topology: staging and prod are MSK with ``AWS_MSK_IAM`` +
``SASL_SSL``. The hermetic harness is SCRAM. What SCRAM proves is
"this client did not silently open PLAINTEXT against an auth-required
listener" -- NOT "this client speaks IAM".

All credentials in this module are synthetic test constants.
"""

from __future__ import annotations

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

# Synthetic, never a real credential.
_SYNTHETIC_USER = "omn18012-synthetic-user"
_SYNTHETIC_PASSWORD = "omn18012-synthetic-not-a-real-secret"

_SCRAM_MECHANISMS = ("PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512")


def _config(mechanism: str, **overrides: object) -> ModelKafkaEventBusConfig:
    kwargs: dict[str, object] = {
        "bootstrap_servers": "localhost:19092",
        "security_protocol": "SASL_PLAINTEXT",
        "sasl_mechanism": mechanism,
        "sasl_plain_username": _SYNTHETIC_USER,
        "sasl_plain_password": _SYNTHETIC_PASSWORD,
    }
    kwargs.update(overrides)
    return ModelKafkaEventBusConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("mechanism", _SCRAM_MECHANISMS)
def test_username_password_mechanisms_thread_credentials(mechanism: str) -> None:
    """The built aiokafka kwargs carry the credentials, not just the mechanism."""
    kwargs = build_aiokafka_auth_kwargs(_config(mechanism))

    assert kwargs["security_protocol"] == "SASL_PLAINTEXT"
    assert kwargs["sasl_mechanism"] == mechanism
    # The defect: these two keys were absent, so aiokafka got a mechanism and
    # no credentials.
    assert kwargs["sasl_plain_username"] == _SYNTHETIC_USER
    assert kwargs["sasl_plain_password"] == _SYNTHETIC_PASSWORD


@pytest.mark.unit
@pytest.mark.parametrize("mechanism", _SCRAM_MECHANISMS)
def test_env_overrides_supply_the_credentials(
    mechanism: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KAFKA_SASL_USERNAME / KAFKA_SASL_PASSWORD reach the built client."""
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", mechanism)
    monkeypatch.setenv("KAFKA_SASL_USERNAME", _SYNTHETIC_USER)
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _SYNTHETIC_PASSWORD)

    config = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:19092"
    ).apply_environment_overrides()

    assert config.sasl_plain_username == _SYNTHETIC_USER
    assert config.sasl_plain_password == _SYNTHETIC_PASSWORD

    kwargs = build_aiokafka_auth_kwargs(config)
    assert kwargs["sasl_plain_username"] == _SYNTHETIC_USER
    assert kwargs["sasl_plain_password"] == _SYNTHETIC_PASSWORD


@pytest.mark.unit
@pytest.mark.parametrize("mechanism", _SCRAM_MECHANISMS)
def test_missing_credentials_are_refused_not_silently_dropped(mechanism: str) -> None:
    """A username/password mechanism with no credentials must fail loudly.

    The pre-fix behaviour built a client anyway; that is the shape of a
    boundary defect that only shows up against a real auth-required listener.
    """
    with pytest.raises(ProtocolConfigurationError):
        _config(mechanism, sasl_plain_username=None, sasl_plain_password=None)


@pytest.mark.unit
def test_plaintext_still_returns_no_auth_kwargs() -> None:
    """Positive control: the PLAINTEXT path is unchanged."""
    config = ModelKafkaEventBusConfig(bootstrap_servers="localhost:19092")
    assert build_aiokafka_auth_kwargs(config) == {}
