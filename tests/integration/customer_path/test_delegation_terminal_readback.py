# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18012 gate 2 -- the terminal readback must survive a truncated partition.

The escape
----------
``_scan_topics_for_correlation`` (node_chain_canary_effect) computed its
backward seek as ``max(0, end - per_partition)``. Offset 0 is a valid floor
only for a partition that has never been truncated. Once retention advanced a
terminal topic's log start above 0, that seek was OFFSET_OUT_OF_RANGE: the
broker refused the fetch and the consumer's ``auto_offset_reset="latest"``
silently repositioned it to the high watermark, past every record already
written -- including the terminal the scan exists to find. Nothing raised and
nothing was logged, so the run reported NOT_FOUND, which reads identically to
a chain that never emitted. Every 2h canary run of 2026-09-06 was RED on a
terminal that was sitting on the bus. Fixed in omnibase_infra#3254
(``130876d9``); parent ``10017ee52``.

Why a unit test was not enough
------------------------------
The landed unit test (``tests/unit/nodes/node_chain_canary_effect/
test_handler_chain_canary_scan_log_start.py``) mocks the consumer, so it
encodes what we *believe* the broker does with an out-of-range seek. This
test asserts it against a real broker with a real trimmed partition: the log
start is advanced with ``rpk topic trim-prefix`` and then RE-ASSERTED through
aiokafka's own ``beginning_offsets`` before the scan runs, so a trim that
silently did not take cannot make this test vacuously green.

The boundary this proves, and the one it does not
-------------------------------------------------
RESIDUAL: staging and prod are MSK with ``AWS_MSK_IAM`` + ``SASL_SSL``; this
harness is SCRAM-SHA-256 over ``SASL_PLAINTEXT``. The scan reaches the broker
through ``build_aiokafka_auth_kwargs_from_env()``, so this module also proves
"this client did not silently open PLAINTEXT against an auth-required
listener" -- it does NOT prove "this client speaks IAM".

All credentials here are synthetic test constants.
"""

from __future__ import annotations

import uuid

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    _scan_topics_for_correlation,
)
from tests.integration.customer_path.redpanda_sasl_harness import RedpandaSasl

pytestmark = [pytest.mark.integration, pytest.mark.kafka, pytest.mark.serial]

# The scan splits this budget across partitions; with one partition the
# backward seek is `end - 250`, which lands below any log start under 250.
MAX_RECORDS = 250


async def _aiokafka_log_start(broker: RedpandaSasl, topic: str) -> int:
    """Read the partition's beginning offset through aiokafka itself."""
    from aiokafka import AIOKafkaConsumer

    from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    config = ModelKafkaEventBusConfig(
        bootstrap_servers=broker.bootstrap
    ).apply_environment_overrides()
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=broker.bootstrap,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        **build_aiokafka_auth_kwargs(config),
    )
    await consumer.start()
    try:
        partitions = sorted(consumer.assignment(), key=lambda tp: tp.partition)
        assert partitions, f"{topic} resolved no partitions"
        begin = await consumer.beginning_offsets(partitions)
        return int(begin[partitions[0]])
    finally:
        await consumer.stop()


@pytest.mark.asyncio
async def test_broker_actually_requires_auth(redpanda_sasl: RedpandaSasl) -> None:
    """Positive control for the harness itself.

    Without this, every 'the client authenticated' assertion below would be
    vacuous on a listener that turned out not to enforce anything.
    """
    unauth = redpanda_sasl.rpk_unauthenticated("cluster", "info")
    assert unauth.returncode != 0, (
        "the harness listener accepted an UNAUTHENTICATED rpk client; "
        "every auth assertion in this module would be vacuous. "
        f"stdout={unauth.stdout!r} stderr={unauth.stderr!r}"
    )
    combined = f"{unauth.stdout}{unauth.stderr}".lower()
    assert "sasl" in combined, f"unexpected refusal reason: {combined!r}"

    authed = redpanda_sasl.rpk("cluster", "info", check=False)
    assert authed.returncode == 0, (
        "the harness's own authenticated client could not reach the broker: "
        f"{authed.stdout}\n{authed.stderr}"
    )


@pytest.mark.asyncio
async def test_terminal_is_found_on_a_truncated_partition(
    kafka_auth_env: RedpandaSasl,
) -> None:
    """The needle is behind the high watermark on a partition trimmed above 0."""
    broker = kafka_auth_env
    topic = f"omn18012.terminal.{uuid.uuid4().hex[:8]}.v1"
    correlation_id = str(uuid.uuid4())

    broker.create_topic(topic, partitions=1)

    # Filler that will be reclaimed, then the needle, then more filler so the
    # needle is not the last record either.
    broker.produce(topic, [f'{{"filler":{i}}}' for i in range(80)])
    broker.produce(topic, [f'{{"correlation_id":"{correlation_id}"}}'])
    broker.produce(topic, [f'{{"filler":{i}}}' for i in range(80, 120)])

    log_start, high = broker.offsets(topic)
    assert log_start == 0, f"expected an untrimmed partition, got {log_start}"
    assert high == 121, f"expected 121 records, got high watermark {high}"

    # Advance the log start ABOVE 0 but BELOW the needle at offset 80.
    broker.trim_prefix(topic, 40)

    # Re-assert through aiokafka: the trim is not assumed, and the client
    # library's own view of the log start is the one the scan will use.
    trimmed_start = await _aiokafka_log_start(broker, topic)
    assert trimmed_start >= 40, (
        f"trim-prefix did not advance the log start as aiokafka sees it: "
        f"{trimmed_start}"
    )
    assert trimmed_start <= 80, (
        f"the trim reclaimed the needle at offset 80 (log start {trimmed_start}); "
        "the test would be proving nothing"
    )

    # `end - MAX_RECORDS` = 121 - 250 = -129, i.e. below the log start. The
    # pre-fix `max(0, ...)` clamp seeks to 0, which is OUT OF RANGE on this
    # partition, and auto_offset_reset="latest" then hides the needle.
    topic_hit, scanned, error = await _scan_topics_for_correlation(
        broker.bootstrap,
        (topic,),
        correlation_id,
        MAX_RECORDS,
        20.0,
        wait_for_arrival=False,
    )

    assert error == "", f"the scan could not be completed: {error}"
    assert topic_hit == topic, (
        "the terminal for this correlation id is on the bus at offset 80 with "
        f"the partition's log start at {trimmed_start}, and the scan reported "
        f"{topic_hit!r} after reading {scanned} records. That is the "
        "OMN-16931 blind readback: a NOT_FOUND indistinguishable from a chain "
        "that never emitted."
    )
    assert scanned > 0


@pytest.mark.asyncio
async def test_absent_correlation_id_still_reports_not_found(
    kafka_auth_env: RedpandaSasl,
) -> None:
    """Negative control: the fix must not turn every scan into a hit."""
    broker = kafka_auth_env
    topic = f"omn18012.terminal.{uuid.uuid4().hex[:8]}.v1"
    broker.create_topic(topic, partitions=1)
    broker.produce(topic, [f'{{"filler":{i}}}' for i in range(60)])
    broker.trim_prefix(topic, 20)

    topic_hit, _scanned, error = await _scan_topics_for_correlation(
        broker.bootstrap,
        (topic,),
        str(uuid.uuid4()),
        MAX_RECORDS,
        10.0,
        wait_for_arrival=False,
    )
    assert error == "", f"the scan could not be completed: {error}"
    assert topic_hit == "", (
        "a correlation id that was never published was reported as found "
        f"on {topic_hit!r}"
    )
