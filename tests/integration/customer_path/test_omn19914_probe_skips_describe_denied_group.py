# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19914 -- real broker ACLs distinguish a hidden group from no group.

The unit proof drives ``GroupAuthorizationFailedError`` through an admin fake,
but that cannot establish the wire shape that caused the incident: a principal
with CLUSTER DESCRIBE can list every group on a shared broker and then receive
Kafka error 30 while finding a particular group's coordinator.  This module
creates that principal and its ACLs through authenticated ``rpk``, then asks
the production probe through an aiokafka client authenticated as that user.

The two cases deliberately share no topic or group names.  A visible Stable
group proves that a hidden candidate is safely ignorable; a topic for which
the only candidate is hidden proves that the probe still refuses CLOSED and
names the grant needed to answer the liveness question.

RESIDUAL: the harness is SCRAM-SHA-256 over ``SASL_PLAINTEXT``, whereas the
affected lane uses IAM over TLS.  The authenticated ListGroups -> per-group
DescribeGroups authorization round trip, including broker-generated error 30,
is the behavior under test and is the same boundary the production probe uses.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from omnibase_core.event_bus.util_consumer_group import TOPIC_SCOPE_INFIX
from omnibase_infra.backends.backend_probe import (
    ConsumerGroupDescribeDeniedError,
    live_consumer_groups,
)

from . import redpanda_sasl_harness as harness
from .test_omn18418_lane_probe_against_a_real_broker import (
    _JoinedConsumer,
    _seed_one_record,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    # The session broker's 120s readiness budget belongs inside this module's
    # watchdog too; without it a busy Docker host reports a watchdog fact,
    # rather than an ACL/probe fact.
    pytest.mark.timeout(600),
    pytest.mark.xdist_group("omn18012_redpanda_sasl"),
]

_RESTRICTED_PASSWORD = "omn19914-synthetic-not-a-real-secret"


def _create_restricted_principal(
    broker: harness.RedpandaSasl,
    *,
    topic: str,
    visible_group: str | None,
) -> tuple[str, str]:
    """Create one user that can list all groups but describe only one named group."""
    username = f"omn19914-{uuid.uuid4().hex[:8]}"
    principal = f"User:{username}"
    broker.rpk(
        "acl",
        "user",
        "create",
        username,
        "-p",
        _RESTRICTED_PASSWORD,
        "--mechanism",
        "SCRAM-SHA-256",
    )

    def allow(*, operation: str, resource: str, name: str | None = None) -> None:
        args = [
            "acl",
            "create",
            "--allow-principal",
            principal,
            "--allow-host",
            "*",
            "--operation",
            operation,
        ]
        if resource == "cluster":
            args.append("--cluster")
        elif resource == "topic":
            args.extend(("--topic", topic))
        elif resource == "group" and name is not None:
            args.extend(("--group", name))
        else:
            raise AssertionError(f"unsupported ACL resource: {resource!r}")
        broker.rpk(*args)

    # CLUSTER DESCRIBE makes ListGroups reveal the hidden candidate.  The
    # topic grants are intentionally exact even though this probe only issues
    # admin calls today: they prevent a topic-metadata preflight from masking
    # the GROUP authorization result this test is meant to prove.
    allow(operation="describe", resource="cluster")
    allow(operation="describe", resource="topic")
    allow(operation="read", resource="topic")
    if visible_group is not None:
        allow(operation="describe", resource="group", name=visible_group)
    return username, _RESTRICTED_PASSWORD


def _authenticate_as(
    monkeypatch: pytest.MonkeyPatch, *, username: str, password: str
) -> None:
    """Select the probe identity for one synchronous call without env leakage."""
    monkeypatch.setenv("KAFKA_SASL_USERNAME", username)
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", password)


def _group_id(run: str, role: str, topic: str) -> str:
    """Return a topic-scoped group id that the probe will select as a candidate."""
    return f"omn19914.{run}.{role}.consume.1.0.0{TOPIC_SCOPE_INFIX}{topic}"


def test_a_group_this_principal_cannot_describe_is_skipped_once_another_is_proven_live(
    kafka_auth_env: harness.RedpandaSasl,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real error-30 hidden group cannot overturn a visible Stable group."""
    broker = kafka_auth_env
    run = uuid.uuid4().hex[:8]
    topic = f"omn19914.probe.{run}"
    visible_group = _group_id(run, "visible", topic)
    hidden_group = _group_id(run, "hidden", topic)
    broker.create_topic(topic)
    asyncio.run(_seed_one_record(topic, broker.bootstrap))
    username, password = _create_restricted_principal(
        broker, topic=topic, visible_group=visible_group
    )

    # Both consumers authenticate as the fixture's superuser.  The probe is
    # switched only after they are joined, so their membership cannot be
    # confused with the restricted identity's administrative visibility.
    with (
        _JoinedConsumer(
            topic=topic, group_id=visible_group, bootstrap=broker.bootstrap
        ),
        _JoinedConsumer(topic=topic, group_id=hidden_group, bootstrap=broker.bootstrap),
    ):
        _authenticate_as(monkeypatch, username=username, password=password)
        restricted_found = live_consumer_groups(
            topic=topic, bootstrap_servers=broker.bootstrap
        )
        assert visible_group in restricted_found
        assert hidden_group not in restricted_found

        # Discriminating control: the same real groups are both visible to the
        # superuser, proving the restricted answer arose from its withheld
        # GROUP DESCRIBE grant rather than a bad join or probe name filter.
        _authenticate_as(
            monkeypatch, username=broker.username, password=broker.password
        )
        assert set(
            live_consumer_groups(topic=topic, bootstrap_servers=broker.bootstrap)
        ) == {visible_group, hidden_group}


def test_when_every_group_is_describe_denied_the_probe_raises_the_typed_refusal(
    kafka_auth_env: harness.RedpandaSasl,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No visible Stable group leaves liveness unknown with its missing grant named."""
    broker = kafka_auth_env
    run = uuid.uuid4().hex[:8]
    topic = f"omn19914.denied.{run}"
    hidden_group = _group_id(run, "hidden", topic)
    broker.create_topic(topic)
    asyncio.run(_seed_one_record(topic, broker.bootstrap))
    username, password = _create_restricted_principal(
        broker, topic=topic, visible_group=None
    )

    with _JoinedConsumer(
        topic=topic, group_id=hidden_group, bootstrap=broker.bootstrap
    ):
        _authenticate_as(monkeypatch, username=username, password=password)
        with pytest.raises(ConsumerGroupDescribeDeniedError) as caught:
            live_consumer_groups(topic=topic, bootstrap_servers=broker.bootstrap)

    message = str(caught.value)
    assert hidden_group in message
    assert "DESCRIBE on GROUP" in message
