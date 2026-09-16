# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18418: the delegate lane probe uses the runtime's own Kafka client family.

``live_consumer_groups`` is the fail-closed gate in front of every dispatched
delegation: ``resolve_delegate_locus`` refuses the run unless this function can
name a live consumer group on the exact command topic. It was the one bus
caller in this repository built on the SYNCHRONOUS ``confluent_kafka`` family,
while every producer, consumer and admin client the runtime actually wires is
``aiokafka`` via :func:`build_aiokafka_auth_kwargs`.

That split is not cosmetic. ``AWS_MSK_IAM`` and ``OAUTHBEARER`` authenticate
through a token *callback* on the client, which librdkafka cannot take as a
flat config entry, so :func:`build_confluent_auth_config` refuses both by
design. On onex-dev -- the only ``AWS_MSK_IAM`` lane -- that refusal reached the
CLI as ``ONEX_CORE_041_INVALID_CONFIGURATION`` and the gate reported it as a
fact about the lane:

    cannot confirm a deployed orchestrator is consuming
    'onex.cmd.omnimarket.delegate-skill.v1' ... sasl_mechanism='AWS_MSK_IAM'
    needs a token callback on the confluent client

Measured in-cluster 2026-09-15T23:38Z on the dev-system cluster, in pod
``omninode-runtime-7946cbb694-wv8v7``, as the deployed container: every
``onex delegate --bus kafka`` refused there, before publishing, while the
runtime in the same pod consumed that topic happily.

OMN-17304 had already hit this class once and fixed it by threading confluent
credentials into the same construction, which is why it reproduced the moment a
mechanism arrived that the confluent family cannot express at all. The fix is
one client resolution path, not a second set of credentials: the probe resolves
its client from the same :class:`ModelKafkaEventBusConfig` and the same auth
builder the runtime wires, so the MSK token callback has exactly one
implementation.

The three lane shapes below are the whole contract of that construction:
PLAINTEXT opens unauthenticated, SCRAM threads its credentials, and IAM
installs the token provider instead of dying. The two PLAINTEXT/SCRAM cases
moved here from ``tests/unit/event_bus/test_omn18012_confluent_admin_transport.py``
when the family changed; their property is unchanged, only the client they
assert against.

All credentials in this module are synthetic test constants.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from omnibase_infra.backends.backend_probe import (
    ConsumerGroupLivenessUnknownError,
    live_consumer_groups,
)

# Synthetic, never a real credential.
_SYNTHETIC_USER = "omn18418-synthetic-user"
_SYNTHETIC_PASSWORD = "omn18418-synthetic-password"

_DELEGATE_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_BROKER = "lane-broker.invalid:19092"

_KAFKA_ENV_VARS = (
    "KAFKA_SECURITY_PROTOCOL",
    "KAFKA_SASL_MECHANISM",
    "KAFKA_SASL_USERNAME",
    "KAFKA_SASL_PASSWORD",
    "KAFKA_SSL_CA_FILE",
    "KAFKA_MSK_REGION",
)


def _clear_kafka_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _KAFKA_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _bound_group_id() -> str:
    from omnibase_core.event_bus.util_consumer_group import TOPIC_SCOPE_INFIX

    return (
        "onex-dev.omnimarket.node_delegate_skill_orchestrator.consume.1.1.0"
        f".__i.runtime-effects{TOPIC_SCOPE_INFIX}{_DELEGATE_COMMAND_TOPIC}"
    )


class _FakeDescribeResponse:
    """The raw ``DescribeGroupsResponse`` shape aiokafka hands back.

    Each group is the wire tuple ``(error_code, group_id, state,
    protocol_type, protocol, members)``. The state is Kafka's own spelling,
    ``"Stable"`` -- not the confluent enum's ``STABLE`` -- which is the one
    detail a family swap silently gets wrong.
    """

    def __init__(self, groups: list[tuple[Any, ...]]) -> None:
        self.groups = groups


class _RecordingAdminClient:
    """Captures the kwargs ``live_consumer_groups`` constructs its client with."""

    seen_kwargs: dict[str, Any] = {}
    listed_groups: list[tuple[str, str]] = []
    described_states: dict[str, str] = {}
    describe_calls: list[list[str]] = []
    closed: bool = False

    def __init__(self, **kwargs: Any) -> None:
        _RecordingAdminClient.seen_kwargs = dict(kwargs)
        _RecordingAdminClient.closed = False

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        _RecordingAdminClient.closed = True

    async def describe_cluster(self) -> dict[str, Any]:
        # The metadata question, asked first and deliberately: an unreachable
        # broker must raise here rather than resolve to an empty group listing.
        return {"brokers": [{"node_id": 1}]}

    async def list_consumer_groups(self) -> list[tuple[str, str]]:
        return list(_RecordingAdminClient.listed_groups)

    async def describe_consumer_groups(
        self, group_ids: list[str]
    ) -> list[_FakeDescribeResponse]:
        _RecordingAdminClient.describe_calls.append(list(group_ids))
        return [
            _FakeDescribeResponse(
                [
                    (
                        0,
                        group_id,
                        _RecordingAdminClient.described_states.get(group_id, "Stable"),
                        "consumer",
                        "",
                        [],
                    )
                    for group_id in group_ids
                ]
            )
        ]


@pytest.fixture
def _stub_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the aiokafka admin client the probe imports inside the call."""
    import aiokafka.admin

    bound = _bound_group_id()
    _RecordingAdminClient.seen_kwargs = {}
    _RecordingAdminClient.listed_groups = [
        (bound, "consumer"),
        ("some.other.group.__t.onex.cmd.unrelated.v1", "consumer"),
    ]
    _RecordingAdminClient.described_states = {}
    _RecordingAdminClient.describe_calls = []
    monkeypatch.setattr(aiokafka.admin, "AIOKafkaAdminClient", _RecordingAdminClient)


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_admin")
def test_msk_iam_lane_resolves_liveness_instead_of_refusing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The onex-dev lane shape answers the question instead of dying on it.

    This is the whole defect. On the confluent client this call raised
    ``ONEX_CORE_041_INVALID_CONFIGURATION`` before a single byte reached the
    broker, and the delegate CLI reported it as "cannot confirm a deployed
    orchestrator is consuming <topic>".
    """
    _clear_kafka_env(monkeypatch)
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "AWS_MSK_IAM")
    monkeypatch.setenv("KAFKA_MSK_REGION", "us-east-2")

    groups = live_consumer_groups(
        topic=_DELEGATE_COMMAND_TOPIC,
        bootstrap_servers=_BROKER,
    )

    assert groups == (_bound_group_id(),)

    kwargs = _RecordingAdminClient.seen_kwargs
    assert kwargs["bootstrap_servers"] == _BROKER
    assert kwargs["security_protocol"] == "SASL_SSL"
    # AWS_MSK_IAM travels on the wire as OAUTHBEARER carrying a SigV4 token,
    # so the mechanism is rewritten and a provider is installed. One
    # implementation of that callback, shared with the runtime.
    assert kwargs["sasl_mechanism"] == "OAUTHBEARER"
    assert kwargs["sasl_oauth_token_provider"].__class__.__name__ == "MSKTokenProvider"
    assert _RecordingAdminClient.closed is True


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_admin")
def test_scram_lane_threads_its_credentials_and_never_logs_them(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A username/password lane keeps authenticating (OMN-17304's property)."""
    _clear_kafka_env(monkeypatch)
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
    monkeypatch.setenv("KAFKA_SASL_USERNAME", _SYNTHETIC_USER)
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _SYNTHETIC_PASSWORD)

    with caplog.at_level(logging.DEBUG):
        groups = live_consumer_groups(
            topic=_DELEGATE_COMMAND_TOPIC,
            bootstrap_servers=_BROKER,
        )

    assert groups == (_bound_group_id(),)

    kwargs = _RecordingAdminClient.seen_kwargs
    assert kwargs["security_protocol"] == "SASL_PLAINTEXT"
    assert kwargs["sasl_mechanism"] == "SCRAM-SHA-256"
    assert kwargs["sasl_plain_username"] == _SYNTHETIC_USER
    assert kwargs["sasl_plain_password"] == _SYNTHETIC_PASSWORD

    assert _SYNTHETIC_PASSWORD not in caplog.text
    assert _SYNTHETIC_PASSWORD not in repr(groups)


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_admin")
def test_plaintext_lane_acquires_no_auth_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PLAINTEXT lane opens unauthenticated, exactly as before."""
    _clear_kafka_env(monkeypatch)

    live_consumer_groups(topic=_DELEGATE_COMMAND_TOPIC, bootstrap_servers=_BROKER)

    assert set(_RecordingAdminClient.seen_kwargs) == {
        "bootstrap_servers",
        "request_timeout_ms",
    }


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_admin")
def test_a_group_that_is_not_stable_is_not_a_live_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebalancing or empty is not "something will pick this up"."""
    _clear_kafka_env(monkeypatch)
    _RecordingAdminClient.described_states = {_bound_group_id(): "Empty"}

    assert (
        live_consumer_groups(topic=_DELEGATE_COMMAND_TOPIC, bootstrap_servers=_BROKER)
        == ()
    )


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_admin")
def test_a_per_group_error_is_unknown_not_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial answer to "is anything consuming this" is not an answer."""
    _clear_kafka_env(monkeypatch)

    class _ErroringAdminClient(_RecordingAdminClient):
        async def describe_consumer_groups(
            self, group_ids: list[str]
        ) -> list[_FakeDescribeResponse]:
            return [
                _FakeDescribeResponse(
                    [
                        (16, group_id, "Dead", "consumer", "", [])
                        for group_id in group_ids
                    ]
                )
            ]

    import aiokafka.admin

    monkeypatch.setattr(aiokafka.admin, "AIOKafkaAdminClient", _ErroringAdminClient)

    with pytest.raises(ConsumerGroupLivenessUnknownError):
        live_consumer_groups(topic=_DELEGATE_COMMAND_TOPIC, bootstrap_servers=_BROKER)


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_admin")
def test_an_unreachable_broker_is_unknown_not_an_empty_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fail-closed property the delegate gate depends on is unchanged."""
    _clear_kafka_env(monkeypatch)

    class _UnreachableAdminClient(_RecordingAdminClient):
        async def start(self) -> None:
            raise OSError("connection refused")

    import aiokafka.admin

    monkeypatch.setattr(aiokafka.admin, "AIOKafkaAdminClient", _UnreachableAdminClient)

    with pytest.raises(ConsumerGroupLivenessUnknownError):
        live_consumer_groups(topic=_DELEGATE_COMMAND_TOPIC, bootstrap_servers=_BROKER)


@pytest.mark.unit
def test_no_broker_address_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to ask is still UNKNOWN, never an empty answer."""
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

    with pytest.raises(ConsumerGroupLivenessUnknownError):
        live_consumer_groups(topic=_DELEGATE_COMMAND_TOPIC)


@pytest.mark.unit
@pytest.mark.usefixtures("_stub_admin")
def test_each_candidate_is_described_on_its_own_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One group per describe -- the batched form does not decode against MSK.

    ``describe_consumer_groups`` batches every group sharing a coordinator into
    a single ``DescribeGroupsRequest``. Measured in-cluster on onex-dev
    2026-09-16, that batched form died with ``ValueError: Buffer underrun
    decoding string`` and took the broker connection with it, while the SAME
    three groups described one at a time returned cleanly (Stable/1 member,
    Empty/0, Stable/1). A batched describe here is therefore a lane probe that
    refuses a healthy lane -- exactly the failure this ticket exists to remove
    -- so the shape is pinned rather than left to whichever call the next edit
    finds tidier.
    """
    from omnibase_core.event_bus.util_consumer_group import TOPIC_SCOPE_INFIX

    _clear_kafka_env(monkeypatch)
    # Three groups on the SAME topic scope, which is the live onex-dev shape:
    # the orchestrator's two rolling versions plus the ledger audit tap.
    scoped = sorted(
        f"onex-dev.omn18418.node-{n}.consume.1.{n}.0"
        f"{TOPIC_SCOPE_INFIX}{_DELEGATE_COMMAND_TOPIC}"
        for n in (1, 2, 3)
    )
    _RecordingAdminClient.listed_groups = [(group, "consumer") for group in scoped]

    found = live_consumer_groups(
        topic=_DELEGATE_COMMAND_TOPIC, bootstrap_servers=_BROKER
    )

    assert set(found) == set(scoped)
    assert _RecordingAdminClient.describe_calls == [[group] for group in scoped], (
        "candidates were not described one per request: "
        f"{_RecordingAdminClient.describe_calls!r}"
    )
