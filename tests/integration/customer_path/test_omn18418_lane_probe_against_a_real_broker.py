# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18418 -- the lane probe answers correctly against a REAL broker.

The unit tests in ``tests/unit/backends/test_omn18418_lane_probe_client_family.py``
pin what ``live_consumer_groups`` CONSTRUCTS: on an ``AWS_MSK_IAM`` lane it now
installs the runtime's own token provider instead of dying inside the confluent
config builder. A fake admin client cannot prove the other half -- that the new
client actually gets the right ANSWER off a live cluster -- and that half is
where a client-family swap goes wrong quietly:

  * the group STATE is not carried on aiokafka's listing at all, so the probe
    has to narrow by name and then describe;
  * the broker spells the state ``Stable`` where the confluent enum spelled it
    ``STABLE``, and a literal comparison there returns "nothing is consuming
    this" for a healthy lane -- indistinguishable from a real wiring death, and
    it fails the delegate gate CLOSED;
  * the describe is a per-group coordinator round trip, which a fake never
    makes.

So this suite runs the real function against the real auth-required Redpanda
the customer-path harness boots, with a real consumer joined to a real group.

Three readings, and the two controls are what make the positive mean anything:

  * NEGATIVE CONTROL -- the same live group, probed for a DIFFERENT topic,
    returns empty. Without it a probe that answered "yes" to everything would
    read as a pass.
  * POSITIVE -- the joined group is found for its own topic.
  * DISCRIMINATING CONTROL -- after the consumer leaves, the SAME group id
    still exists on the broker but is no longer ``Stable``, and the probe stops
    reporting it. That is the difference between reading wiring truth and
    matching a string.

RESIDUAL, stated: this harness is SCRAM-SHA-256 over ``SASL_PLAINTEXT``. It is
NOT ``AWS_MSK_IAM`` over ``SASL_SSL``, which is what onex-dev runs and what
this ticket's defect is about, and no IAM broker exists to test against here.
The mechanism differs; the client family, the describe round trip and the state
spelling -- everything this module actually asserts -- do not.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest

from omnibase_core.event_bus.util_consumer_group import TOPIC_SCOPE_INFIX
from omnibase_infra.backends.backend_probe import live_consumer_groups
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_from_env

from . import redpanda_sasl_harness as harness

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    # The module-scoped harness boots a REAL container against a 120s readiness
    # budget and pytest-timeout counts fixture setup against the first test, so
    # the 60s default watchdog would report a fact about the watchdog rather
    # than about the tree. Same reasoning as the OMN-18012 module beside this
    # one.
    pytest.mark.timeout(600),
    # Share the session broker rather than booting a second one under
    # ``-n4 --dist=loadgroup``.
    pytest.mark.xdist_group("omn18012_redpanda_sasl"),
]

_JOIN_TIMEOUT_S = 60.0
_LEAVE_TIMEOUT_S = 60.0
_POLL_INTERVAL_S = 1.0


class _JoinedConsumer:
    """A real aiokafka consumer, joined to a real group, on a background loop.

    ``live_consumer_groups`` is synchronous by contract -- the delegate CLI's
    locus gate is -- and drives its own ``asyncio.run``. Calling it from inside
    a running loop would nest one, so the consumer owns a loop on its own
    thread and the probe is called from the main thread, exactly as the CLI
    calls it.
    """

    def __init__(self, *, topic: str, group_id: str, bootstrap: str) -> None:
        self._topic = topic
        self._group_id = group_id
        self._bootstrap = bootstrap
        self._joined = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _JoinedConsumer:
        self._thread.start()
        if not self._joined.wait(timeout=_JOIN_TIMEOUT_S):
            raise AssertionError(
                f"consumer never joined {self._group_id!r} within "
                f"{_JOIN_TIMEOUT_S}s: {self._error!r}"
            )
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=_LEAVE_TIMEOUT_S)
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        try:
            asyncio.run(self._consume())
        except BaseException as exc:  # noqa: BLE001 — surfaced on __exit__
            self._error = exc
            self._joined.set()

    async def _consume(self) -> None:
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            auto_offset_reset="earliest",
            **build_aiokafka_auth_kwargs_from_env(),
        )
        await consumer.start()
        try:
            # Consume the seeded record, so the group has COMMITTED an offset
            # by the time it leaves. A group that never committed can be
            # reaped the moment its last member goes, which would delete the
            # subject of the discriminating control below and let that control
            # pass for the wrong reason.
            deadline = time.monotonic() + _JOIN_TIMEOUT_S
            while time.monotonic() < deadline:
                batch = await consumer.getmany(timeout_ms=1000, max_records=1)
                if any(records for records in batch.values()):
                    break
            await consumer.commit()
            self._joined.set()
            while not self._stop.is_set():
                await consumer.getmany(timeout_ms=500, max_records=1)
        finally:
            await consumer.stop()


async def _seed_one_record(topic: str, bootstrap: str) -> None:
    """Put one record on the topic so the consumer has something to commit."""
    from aiokafka import AIOKafkaProducer

    producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap,
        request_timeout_ms=15000,
        **build_aiokafka_auth_kwargs_from_env(),
    )
    await producer.start()
    try:
        await producer.send_and_wait(topic, uuid.uuid4().hex.encode())
    finally:
        await producer.stop()


def _await_groups(topic: str, bootstrap: str, *, want: bool) -> tuple[str, ...]:
    """Poll the probe until it reports (or stops reporting) a live group.

    Rebalance is not instantaneous in either direction, so a single reading is
    a race, not a result. Every call here is the real function.
    """
    deadline = time.monotonic() + _JOIN_TIMEOUT_S
    groups: tuple[str, ...] = ()
    while time.monotonic() < deadline:
        groups = live_consumer_groups(topic=topic, bootstrap_servers=bootstrap)
        if bool(groups) is want:
            return groups
        time.sleep(_POLL_INTERVAL_S)
    return groups


def test_liveness_reads_a_real_joined_group_and_stops_when_it_leaves(
    kafka_auth_env: harness.RedpandaSasl,
) -> None:
    """The whole contract of the probe, against a live authenticated broker."""
    broker = kafka_auth_env
    run = uuid.uuid4().hex[:8]
    topic = f"omn18418.probe.{run}"
    other_topic = f"omn18418.other.{run}"
    broker.create_topic(topic)
    broker.create_topic(other_topic)
    asyncio.run(_seed_one_record(topic, broker.bootstrap))

    # Topic-scoped by the canonical grammar: the probe's whole narrowing rule
    # is that a group id ending in this suffix is bound to this topic.
    group_id = f"omn18418.{run}.consume.1.0.0{TOPIC_SCOPE_INFIX}{topic}"

    with _JoinedConsumer(topic=topic, group_id=group_id, bootstrap=broker.bootstrap):
        # POSITIVE -- the joined group is wiring truth for its own topic.
        found = _await_groups(topic, broker.bootstrap, want=True)
        assert group_id in found, (
            f"the probe did not see the live group {group_id!r} on {topic!r}; "
            f"it reported {found!r}"
        )

        # NEGATIVE CONTROL -- the same live group, a different topic. A probe
        # that answered yes to everything would have passed the line above.
        assert (
            live_consumer_groups(topic=other_topic, bootstrap_servers=broker.bootstrap)
            == ()
        ), "the probe reported a consumer for a topic nothing is bound to"

    # DISCRIMINATING CONTROL -- the group id still exists on the broker, and is
    # no longer Stable. A name match would still report it here; reading the
    # state does not.
    assert broker.rpk("group", "describe", group_id, check=False).returncode == 0, (
        "the group vanished entirely, so the assertion below would pass for "
        "the wrong reason"
    )
    assert _await_groups(topic, broker.bootstrap, want=False) == (), (
        "the probe still reports a group whose only member has left -- it is "
        "matching the name, not reading the state"
    )


def test_an_unanswerable_broker_is_unknown_not_an_empty_answer(
    kafka_auth_env: harness.RedpandaSasl,
) -> None:
    """Fail-closed, proven against a real socket rather than a stub.

    The delegate gate refuses on UNKNOWN by design. The failure this guards is
    the opposite: a client that cannot reach the cluster answering the group
    question EMPTY, which reads as "nobody is bound" and lets a command be
    published into silence.
    """
    from omnibase_infra.backends.backend_probe import ConsumerGroupLivenessUnknownError

    host, _sep, _port = kafka_auth_env.bootstrap.rpartition(":")
    # A port nothing is listening on, on the same host the live broker proves
    # is reachable -- so this is a statement about the port, not the network.
    with pytest.raises(ConsumerGroupLivenessUnknownError):
        live_consumer_groups(
            topic="omn18418.unreachable.v1", bootstrap_servers=f"{host}:1"
        )
