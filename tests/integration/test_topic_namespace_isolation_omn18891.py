# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""End-to-end topic isolation across the real transport seams [OMN-18891].

The unit tests read one seam each. These read the seams TOGETHER, the way a
second runtime on a shared broker exercises them, because the isolation is a
property of the whole set and not of any one function. A namespace applied at
four seams and missed at the fifth reports isolation it does not have, and the
one that was missed is the one that talks to the shared lane.

Three properties are checked against the shipped code paths rather than
against a stand-in:

1. **Round trip.** A topic published through the bus goes out physical and
   comes back canonical, so the name a handler is dispatched under is the one
   its contract declares.
2. **Whole-catalogue pass-through.** Over every platform topic suffix, an
   unset namespace produces byte-identically the list produced today, and a
   set one produces a list that strips back to it byte-identically. This is
   the assertion that lets the change land before any slot exists.
3. **No seam left behind.** The physical name reaching a broker client and the
   canonical name reaching a comparison are never the same string once a
   namespace is configured, which is what makes a missed seam detectable.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.topics import platform_topic_suffixes
from omnibase_infra.topics.topic_namespace import (
    TOPIC_NAMESPACE_ENV_VAR,
    create_topic_resolver,
)

BOOTSTRAP = "localhost:9092"
SLOT = "prepr1"


def _all_platform_suffixes() -> list[str]:
    return sorted(
        {
            value
            for name, value in vars(platform_topic_suffixes).items()
            if name.startswith("SUFFIX_") and isinstance(value, str)
        }
    )


@pytest.mark.integration
def test_every_platform_suffix_is_pass_through_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset, the resolved catalogue is byte-identical to the declared one.

    This is the falsifier for "an existing lane cannot be changed by this
    code", taken over the whole catalogue rather than over one example.
    """
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    suffixes = _all_platform_suffixes()
    assert suffixes, "positive control: the suffix catalogue must not be empty"
    resolver = create_topic_resolver()
    assert [resolver.resolve(s) for s in suffixes] == suffixes


@pytest.mark.integration
def test_every_platform_suffix_round_trips_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set, every name is prefixed and every one strips back exactly."""
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    suffixes = _all_platform_suffixes()
    resolver = create_topic_resolver()
    physical = [resolver.resolve(s) for s in suffixes]

    unprefixed = [p for p in physical if not p.startswith(f"{SLOT}.")]
    assert not unprefixed, f"seam(s) missed the namespace: {unprefixed[:5]}"
    assert [resolver.canonical(p) for p in physical] == suffixes


@pytest.mark.integration
def test_the_physical_catalogue_is_declarable_nowhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: no physical name can be written into a contract.

    The prefix stays a deployment fact because the suffix grammar refuses it,
    so there is no path by which a slot's topic name reaches a ``contract.yaml``.
    """
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    resolver = create_topic_resolver()
    suffixes = _all_platform_suffixes()

    # Positive control first: the canonical names ARE declarable, so a blanket
    # "everything is invalid" bug in the validator cannot pass this test.
    assert all(validate_topic_suffix(s).is_valid for s in suffixes)
    assert not any(
        validate_topic_suffix(resolver.resolve(s)).is_valid for s in suffixes
    )


@pytest.mark.integration
def test_bus_publish_and_subscribe_use_the_physical_name_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both directions move at once, or the isolation is one-sided.

    A missed PUBLISH prefix puts this runtime's events on the shared topic. A
    missed SUBSCRIBE prefix hands it a copy of the shared lane's traffic. Only
    the first is visible from outside, so both are read here.
    """
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    topic = _all_platform_suffixes()[0]
    bus = EventBusKafka(config=ModelKafkaEventBusConfig(bootstrap_servers=BOOTSTRAP))

    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer"
    ) as consumer_cls:
        bus._build_consumer(topic, "group", "instance", "earliest")
    subscribed = consumer_cls.call_args.args[0]

    transport = KafkaTransport(
        config=ModelKafkaEventBusConfig(bootstrap_servers=BOOTSTRAP),
        topics=(topic,),
    )

    assert subscribed == f"{SLOT}.{topic}"
    assert transport._physical_topics == (f"{SLOT}.{topic}",)
    assert transport._topics == (topic,), "the runtime's own view stays canonical"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_record_published_physical_is_handed_back_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The round trip a handler actually sees, over the transport.

    The broker is addressed in physical names and the runtime is answered in
    canonical ones. A handler registry keyed by the contract-declared name is
    exactly what a physical name would miss, so this is the mapping control.
    """
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    topic = _all_platform_suffixes()[0]
    transport = KafkaTransport(
        config=ModelKafkaEventBusConfig(bootstrap_servers=BOOTSTRAP),
        topics=(topic,),
    )

    sent: dict[str, Any] = {}

    class _Metadata:
        topic = f"{SLOT}.{_all_platform_suffixes()[0]}"
        partition = 0
        offset = 7

    async def _send_and_wait(target: str, **kwargs: Any) -> _Metadata:
        sent["topic"] = target
        return _Metadata()

    producer = MagicMock()
    producer.send_and_wait = _send_and_wait
    transport._producer = producer

    coordinate = await transport.send_with_coordinate(topic, None, b"payload", {})

    assert sent["topic"] == f"{SLOT}.{topic}", "published onto the shared topic"
    assert coordinate[0] == topic, "the coordinate must answer in canonical terms"
