# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19452 AC1: a broker that refuses the SASL login is reported as that.

Measured before this change: ``onex delegate --bus kafka --lane dev`` with a
wrong lane credential ended in "Fix the broker address, or pass --bus inmemory",
advice that sends the operator to a broker address that was never wrong.
aiokafka logs the broker's verdict (``[Error 58] SaslAuthenticationFailed``) on
its own logger and then raises a generic ``KafkaConnectionError: Unable to
bootstrap``, so the exception the probe sees carries no cause at all.

These tests connect the REAL aiokafka admin client to a listener that refuses
the SASL login on the wire, so the classification is proved against the client
production builds rather than against an exception a test constructed.

All names below are synthetic.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import yaml

from omnibase_infra.backends.backend_probe import (
    ConsumerGroupLivenessUnknownError,
    ConsumerGroupSaslRefusedError,
    live_consumer_groups,
)
from omnibase_infra.cli.delegate_locus import (
    DelegateLocusRefusedError,
    DelegateLocusSaslRefusedError,
    resolve_delegate_locus,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string
from tests.helpers.fake_sasl_refusing_broker import (
    SASL_MECHANISM,
    serve_sasl_refusing_broker,
)

_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_PRINCIPAL = "dev-cli-synthetic-host"
_BROKER_ADVICE = "Fix the broker address"

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _sasl_identity_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "KAFKA_SECURITY_PROTOCOL",
        "KAFKA_SASL_MECHANISM",
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", SASL_MECHANISM)
    monkeypatch.setenv("KAFKA_SASL_USERNAME", _PRINCIPAL)
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", "synthetic-not-real")


class TestTheProbeNamesASaslRefusal:
    def test_a_refused_login_is_a_typed_sasl_refusal_naming_the_principal(
        self,
    ) -> None:
        with serve_sasl_refusing_broker() as broker:
            with pytest.raises(ConsumerGroupSaslRefusedError) as raised:
                live_consumer_groups(
                    topic=_TOPIC, bootstrap_servers=broker, timeout=5.0
                )

        refusal = raised.value
        assert refusal.principal == _PRINCIPAL
        assert refusal.bootstrap_servers == broker
        assert _PRINCIPAL in str(refusal)
        assert _BROKER_ADVICE not in str(refusal)

    def test_the_sasl_refusal_is_still_an_unknown_liveness_answer(self) -> None:
        """UNKNOWN is not permission: every fail-closed caller keeps refusing."""
        assert issubclass(
            ConsumerGroupSaslRefusedError, ConsumerGroupLivenessUnknownError
        )

    def test_the_message_survives_the_receipt_sanitizer(self) -> None:
        """The receipt stores the message through ``sanitize_error_string``.

        Any credential-shaped word redacts the whole message, which would turn
        the finding into ``[REDACTED - potentially sensitive data]``.
        """
        with serve_sasl_refusing_broker() as broker:
            with pytest.raises(ConsumerGroupSaslRefusedError) as raised:
                live_consumer_groups(
                    topic=_TOPIC, bootstrap_servers=broker, timeout=5.0
                )

        assert sanitize_error_string(str(raised.value)) == str(raised.value)

    def test_the_secret_never_reaches_the_message(self) -> None:
        with serve_sasl_refusing_broker() as broker:
            with pytest.raises(ConsumerGroupSaslRefusedError) as raised:
                live_consumer_groups(
                    topic=_TOPIC, bootstrap_servers=broker, timeout=5.0
                )

        assert "synthetic-not-real" not in str(raised.value)

    def test_positive_control_a_closed_port_is_not_called_a_sasl_refusal(
        self,
    ) -> None:
        """The classification is not 'every bootstrap failure is SASL'."""
        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            closed = f"127.0.0.1:{held.getsockname()[1]}"

        with pytest.raises(ConsumerGroupLivenessUnknownError) as raised:
            live_consumer_groups(topic=_TOPIC, bootstrap_servers=closed, timeout=2.0)

        assert not isinstance(raised.value, ConsumerGroupSaslRefusedError)


class TestTheLocusRefusalNamesIt:
    def test_the_locus_gate_raises_a_sasl_refusal_without_the_broker_advice(
        self, tmp_path: Path
    ) -> None:
        packaged = tmp_path / "contract.yaml"
        packaged.write_text(
            yaml.safe_dump(
                {
                    "name": "node_delegate_skill_orchestrator",
                    "event_bus": {"subscribe_topics": [_TOPIC]},
                }
            ),
            encoding="utf-8",
        )
        with serve_sasl_refusing_broker() as broker:
            with pytest.raises(DelegateLocusSaslRefusedError) as raised:
                resolve_delegate_locus(
                    requested=EnumDelegateLocus.DEPLOYED_LANE,
                    bus="kafka",
                    kafka_bootstrap=broker,
                    contract_path=packaged,
                    shared_bus_value="kafka",
                )

        refusal = raised.value
        assert isinstance(refusal, DelegateLocusRefusedError)
        assert refusal.principal == _PRINCIPAL
        assert refusal.broker == broker
        assert _BROKER_ADVICE not in str(refusal)
