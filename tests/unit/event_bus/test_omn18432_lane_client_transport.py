# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18432 AC1/AC4/AC5: the declared lane transport reaches the built client.

OMN-16871 moved the broker ADDRESS out of the ambient environment and into the
lane declaration. It left the other three quarters of the statement behind: the
CLI reads ``security_protocol`` and ``sasl_mechanism`` off the same declaration,
logs them, and forwards only the address, because ``omnibase_core``'s accepted
backend-override key set is closed and carries no transport key. So a client
addressed at a SASL listener is still assembled entirely from
``KAFKA_SECURITY_PROTOCOL`` / ``KAFKA_SASL_*``.

That is the OMN-18012 failure mode with the arrow reversed. There, credential
PRESENCE was used to infer a transport and picked TLS against a listener that
speaks none. Here the lane DECLARES the transport, the process READS it, and
then discards it in favour of whatever the shell exported.

These tests pin the seam that closes it: a typed, address-matched, explicitly
scoped binding that the process entry point establishes and the bus factory
consults. The address match is the safety property worth stating plainly -- a
binding for one lane can never be applied to a different broker, so the failure
this closes cannot be re-introduced as "the last lane's credential leaked onto
the next connection".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from omnibase_infra.event_bus.lane_client_transport_binding import (
    LaneClientTransportBindingError,
    bind_lane_client_transport,
    resolve_lane_client_transport,
)
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)

pytestmark = pytest.mark.unit

_DECLARED_IN = Path("omnimarket/config/ci_bus_lanes.yaml")


def _sasl_transport(
    *, bootstrap: str = "broker.invalid:19092"
) -> ModelLaneClientTransport:
    return ModelLaneClientTransport(
        lane="dev",
        bootstrap_servers=bootstrap,
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism="SCRAM-SHA-256",
        sasl_username="dev-cli-host",
        sasl_password=SecretStr("s3kr3t-value"),
        declared_in=_DECLARED_IN,
    )


def test_no_binding_resolves_to_nothing() -> None:
    """AC5: with no binding established the seam is inert, not a default."""
    assert resolve_lane_client_transport("broker.invalid:19092") is None


def test_a_binding_resolves_only_for_its_own_broker_address() -> None:
    """The safety property: a lane's credential cannot reach another broker."""
    with bind_lane_client_transport(_sasl_transport()):
        assert resolve_lane_client_transport("broker.invalid:19092") is not None
        assert resolve_lane_client_transport("other-broker.invalid:39092") is None


def test_the_binding_is_gone_once_its_scope_ends() -> None:
    """Scoped, not process-lifetime: the value does not outlive the dispatch."""
    with bind_lane_client_transport(_sasl_transport()):
        pass
    assert resolve_lane_client_transport("broker.invalid:19092") is None


def test_the_binding_is_cleared_even_when_the_dispatch_raises() -> None:
    with pytest.raises(RuntimeError):
        with bind_lane_client_transport(_sasl_transport()):
            raise RuntimeError("dispatch blew up")
    assert resolve_lane_client_transport("broker.invalid:19092") is None


def test_a_second_binding_is_refused_rather_than_stacked() -> None:
    """Two lanes in one process is not a selection; it is a mistake."""
    with bind_lane_client_transport(_sasl_transport()):
        with pytest.raises(LaneClientTransportBindingError):
            with bind_lane_client_transport(
                _sasl_transport(bootstrap="other-broker.invalid:39092")
            ):
                pass


def test_client_config_overrides_carry_the_declared_transport_and_credential() -> None:
    """AC1: what the declaration says is what the client is built with."""
    overrides = _sasl_transport().as_client_config_overrides()

    assert overrides["security_protocol"] == "SASL_PLAINTEXT"
    assert overrides["sasl_mechanism"] == "SCRAM-SHA-256"
    assert overrides["sasl_plain_username"] == "dev-cli-host"
    assert overrides["sasl_plain_password"] == "s3kr3t-value"


def test_a_plaintext_lane_carries_no_credential_fields() -> None:
    """A non-SASL lane declares a protocol and nothing else; no blank creds."""
    transport = ModelLaneClientTransport(
        lane="lab",
        bootstrap_servers="plain.invalid:19092",
        security_protocol="PLAINTEXT",
        declared_in=_DECLARED_IN,
    )

    overrides = transport.as_client_config_overrides()

    assert overrides == {"security_protocol": "PLAINTEXT"}


def test_a_sasl_protocol_without_a_credential_is_not_constructible() -> None:
    """AC4 at the type level: there is no half-built SASL transport to pass on."""
    with pytest.raises(ValidationError):
        ModelLaneClientTransport(
            lane="dev",
            bootstrap_servers="broker.invalid:19092",
            security_protocol="SASL_PLAINTEXT",
            sasl_mechanism="SCRAM-SHA-256",
            declared_in=_DECLARED_IN,
        )


def test_a_sasl_protocol_without_a_mechanism_is_not_constructible() -> None:
    with pytest.raises(ValidationError):
        ModelLaneClientTransport(
            lane="dev",
            bootstrap_servers="broker.invalid:19092",
            security_protocol="SASL_PLAINTEXT",
            sasl_username="dev-cli-host",
            sasl_password=SecretStr("s3kr3t-value"),
            declared_in=_DECLARED_IN,
        )


def test_a_non_sasl_protocol_carrying_a_credential_is_not_constructible() -> None:
    """A credential beside a plaintext protocol is a contradiction, not a spare."""
    with pytest.raises(ValidationError):
        ModelLaneClientTransport(
            lane="lab",
            bootstrap_servers="plain.invalid:19092",
            security_protocol="PLAINTEXT",
            sasl_username="lab-cli-host",
            sasl_password=SecretStr("s3kr3t-value"),
            declared_in=_DECLARED_IN,
        )


def test_the_secret_does_not_appear_in_the_model_repr() -> None:
    """A transport object lands in log lines and tracebacks; the value must not."""
    rendered = repr(_sasl_transport())
    assert "s3kr3t-value" not in rendered


class TestBusFactoryHonoursTheBinding:
    """``EventBusKafka.from_bootstrap`` is the only seam core can reach."""

    def test_the_built_config_carries_the_bound_transport(self) -> None:
        """AC1 end to end: address from core, transport from the declaration."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

        with bind_lane_client_transport(_sasl_transport()):
            bus = EventBusKafka.from_bootstrap("broker.invalid:19092")

        assert bus.config.bootstrap_servers == "broker.invalid:19092"
        assert bus.config.security_protocol == "SASL_PLAINTEXT"
        assert bus.config.sasl_mechanism == "SCRAM-SHA-256"
        assert bus.config.sasl_plain_username == "dev-cli-host"
        assert bus.config.sasl_plain_password == "s3kr3t-value"

    def test_an_unbound_address_is_built_exactly_as_before(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC5: a surface with no binding keeps the environment-sourced answer."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

        monkeypatch.delenv("KAFKA_SECURITY_PROTOCOL", raising=False)
        monkeypatch.delenv("KAFKA_SASL_MECHANISM", raising=False)
        monkeypatch.delenv("KAFKA_SASL_USERNAME", raising=False)
        monkeypatch.delenv("KAFKA_SASL_PASSWORD", raising=False)

        unbound = EventBusKafka.from_bootstrap("broker.invalid:19092").config
        with bind_lane_client_transport(
            _sasl_transport(bootstrap="somewhere-else.invalid:19092")
        ):
            while_bound_elsewhere = EventBusKafka.from_bootstrap(
                "broker.invalid:19092"
            ).config

        assert while_bound_elsewhere.model_dump() == unbound.model_dump()
