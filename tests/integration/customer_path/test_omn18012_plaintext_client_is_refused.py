# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18012 -- the gate: a PLAINTEXT client must FAIL on an auth-required broker.

The whole value of giving the dev lane an authenticated listener is that the
REJECTION EXISTS. A no-auth broker accepts the plaintext client, which is
exactly how escape 1 of 2026-09-06 passed every local check: omnimarket's
``_build_event_bus`` skipped ``apply_environment_overrides``, produced a
PLAINTEXT client against the IAM-only MSK listener, and every BYOK
registration 503'd in front of a customer.

Structure of the proof, deliberately symmetric:

  * NEGATIVE  -- the config path that reproduces escape 1 (no environment
    overrides applied at all, so ``security_protocol`` keeps its PLAINTEXT
    default) cannot reach the broker.
  * POSITIVE CONTROL -- the same client, same broker, same topic, with
    ``apply_environment_overrides()`` applied, produces and consumes. Without
    this half the negative is not evidence: a broker that is simply down also
    refuses the plaintext client.

RESIDUAL: this proves PLAINTEXT-vs-authenticated, not IAM. Staging and prod
are ``AWS_MSK_IAM`` over ``SASL_SSL``; this is SCRAM-SHA-256 over
``SASL_PLAINTEXT``. The mechanism differs. The failure mode pinned here -- a
client constructed with no credentials at all -- does not.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator

import pytest

from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

from . import redpanda_sasl_harness as harness

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def broker() -> Iterator[harness.RedpandaSasl]:
    if not harness.docker_available():
        pytest.fail(
            "docker is not reachable. This boundary test does not skip: a "
            "silently-absent auth test is how escape 1 stayed invisible."
        )
    running = harness.start_redpanda_sasl()
    try:
        yield running
    finally:
        harness.stop_redpanda(running)


def test_harness_listener_actually_enforces_auth(
    broker: harness.RedpandaSasl,
) -> None:
    """Positive control on the HARNESS itself, before anything else asserts.

    If the listener were not enforcing, every other assertion in this module
    would be vacuously green.
    """
    authenticated = broker.rpk("cluster", "info", check=False)
    assert authenticated.returncode == 0, (
        f"authenticated rpk failed, the harness is broken: "
        f"{authenticated.stdout}\n{authenticated.stderr}"
    )
    anonymous = broker.rpk_unauthenticated("cluster", "info")
    assert anonymous.returncode != 0, (
        "an UNAUTHENTICATED rpk reached the broker -- the listener is not "
        "enforcing auth, so this whole module proves nothing"
    )


def _config_from_env() -> ModelKafkaEventBusConfig:
    """The correct path: construct, then apply the environment overrides."""
    return ModelKafkaEventBusConfig(
        bootstrap_servers=os.environ["KAFKA_BOOTSTRAP_SERVERS"],
    ).apply_environment_overrides()


def _config_without_overrides() -> ModelKafkaEventBusConfig:
    """Escape 1's shape: build the config and never apply the overrides."""
    return ModelKafkaEventBusConfig(
        bootstrap_servers=os.environ["KAFKA_BOOTSTRAP_SERVERS"],
    )


async def _roundtrip(config: ModelKafkaEventBusConfig, topic: str) -> str:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    auth = build_aiokafka_auth_kwargs(config)
    payload = uuid.uuid4().hex.encode()

    producer = AIOKafkaProducer(
        bootstrap_servers=config.bootstrap_servers,
        request_timeout_ms=15000,
        **auth,
    )
    await producer.start()
    try:
        await producer.send_and_wait(topic, payload)
    finally:
        await producer.stop()

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=config.bootstrap_servers,
        group_id=f"omn18012-{uuid.uuid4().hex[:8]}",
        auto_offset_reset="earliest",
        **auth,
    )
    await consumer.start()
    try:
        batch = await consumer.getmany(timeout_ms=15000, max_records=10)
        for records in batch.values():
            for record in records:
                if record.value == payload:
                    return "roundtrip"
    finally:
        await consumer.stop()
    raise AssertionError("authenticated client produced but never consumed back")


def test_plaintext_client_is_refused_and_the_authenticated_one_is_not(
    broker: harness.RedpandaSasl, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in broker.env().items():
        monkeypatch.setenv(key, value)

    topic = f"omn18012.gate.{uuid.uuid4().hex[:8]}"
    broker.create_topic(topic)

    # -- POSITIVE CONTROL first: the correct client works end to end. --------
    assert asyncio.run(_roundtrip(_config_from_env(), topic)) == "roundtrip"

    # -- NEGATIVE: escape 1's shape cannot reach the same broker. ------------
    plaintext = _config_without_overrides()
    assert plaintext.security_protocol == "PLAINTEXT"
    assert build_aiokafka_auth_kwargs(plaintext) == {}, (
        "a PLAINTEXT config must yield NO auth kwargs -- that is precisely "
        "the silent client escape 1 built"
    )
    with pytest.raises(BaseException) as refused:
        asyncio.run(_roundtrip(plaintext, topic))
    assert not isinstance(refused.value, AssertionError), (
        "the plaintext client REACHED the broker and only failed the readback "
        "-- the listener is not refusing it"
    )
