# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Extends the ``event_bus_substrate`` fixture with a real Kafka leg (OMN-15789).

The parent directory's ``conftest.py`` (``tests/integration/event_bus/conftest.py``)
pins ``KAFKA_BOOTSTRAP_SERVERS`` to the local Docker Redpanda broker
(``localhost:19092``) via a session-wide ``pytest_configure`` hook, and is
inherited here automatically (pytest conftest hooks apply to the whole
directory tree they're collected under, not just their own directory). The
real opt-in gate -- whether to actually attempt a live broker connection --
is ``KAFKA_INTEGRATION_TESTS=1``, checked here at fixture-execution time
(not module import time), matching the existing precedent in
``tests/integration/event_bus/test_kafka_event_bus_integration.py``.

Defines ``event_bus_substrate`` / ``fidelity_event_bus_substrate`` with the
SAME NAMES as ``omnibase_core.event_bus.testing.fixture_event_bus_substrate``,
adding the ``"real_broker"`` param (``pytest.mark.integration`` +
``pytest.mark.kafka``) backed by the real, already-deployed ``EventBusKafka``
(``omnibase_infra.event_bus.event_bus_kafka``) -- no new bus implementation.

.. versionadded:: OMN-15789
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_core.event_bus.event_bus_semantic_fake import EventBusSemanticFake
from omnibase_core.event_bus.testing.fixture_event_bus_substrate import (
    CORE_EVENT_BUS_SUBSTRATE_PARAMS,
    CORE_FIDELITY_SUBSTRATE_PARAMS,
    DEFAULT_SUBSTRATE_ENVIRONMENT,
    build_core_event_bus_substrate_instance,
)
from omnibase_core.event_bus.testing.topic_constants import (
    FIDELITY_EARLIEST_TOPIC,
    FIDELITY_JOIN_LEAVE_TOPIC,
    FIDELITY_LATEST_TOPIC,
    FIDELITY_REBALANCE_TOPIC,
    FIDELITY_REJOIN_TOPIC,
    SEAM_TEST_TOPIC,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from tests.helpers.util_kafka import KafkaTopicManager

#: Widens omnibase_core's alias with the real-broker-backed concrete type.
InfraEventBusSubstrate = EventBusInmemory | EventBusSemanticFake | EventBusKafka

#: Every topic name the shared contract tests
#: (``omnibase_core.event_bus.testing.contract_event_bus_substrate``) use,
#: literally. Auto-topic-creation is disabled on the local Redpanda broker
#: (see the parent conftest's module docstring), so the real_broker leg must
#: pre-create these -- publish() to a nonexistent topic fails closed rather
#: than silently succeeding. Kept as an explicit, reviewable list rather than
#: derived, so a new shared test topic is a visible one-line addition here.
SHARED_CONTRACT_TEST_TOPICS: tuple[str, ...] = (
    SEAM_TEST_TOPIC,
    FIDELITY_JOIN_LEAVE_TOPIC,
    FIDELITY_EARLIEST_TOPIC,
    FIDELITY_LATEST_TOPIC,
    FIDELITY_REJOIN_TOPIC,
    FIDELITY_REBALANCE_TOPIC,
)

_KAFKA_INTEGRATION_TESTS_ENV_VAR = "KAFKA_INTEGRATION_TESTS"


def _real_broker_available() -> bool:
    """Whether the real_broker leg should attempt a live connection.

    Checked at fixture-execution time (not import time) so a test run can
    set the env var after collection. ``KAFKA_BOOTSTRAP_SERVERS`` is not
    re-checked here: the parent conftest's ``pytest_configure`` has already
    pinned it unconditionally by the time any fixture runs.
    """
    return os.getenv(_KAFKA_INTEGRATION_TESTS_ENV_VAR) == "1"


async def _build_real_broker_instance() -> EventBusKafka:
    bootstrap_servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    config = ModelKafkaEventBusConfig(
        bootstrap_servers=bootstrap_servers,
        environment=DEFAULT_SUBSTRATE_ENVIRONMENT,
        timeout_seconds=30,
        max_retry_attempts=2,
        retry_backoff_base=0.5,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=10.0,
    )
    return EventBusKafka(config=config)


async def _ensure_shared_contract_test_topics(bootstrap_servers: str) -> None:
    """Idempotently create every topic the shared contract tests publish to.

    ``KafkaTopicManager.create_topic`` already treats
    ``TopicAlreadyExistsError`` as harmless (see
    ``tests/helpers/util_kafka.py``), so calling this on every real_broker
    fixture build is safe and cheap once the topics exist.

    Fail-closed post-condition (mergesweep-0809-dualsub-verify F1): a live
    adversarial re-verify proved ``KafkaTopicManager.create_topic()`` can
    report success while creating NOTHING on the broker (rpk through the
    identical tunnel creates the same topic fine -- not environmental). The
    mechanism is that ``create_topic()`` awaits
    ``wait_for_topic_metadata(...)`` but never checks its boolean return
    value, so a metadata-propagation timeout (which means the topic never
    actually appeared) is silently discarded rather than raised. Since
    ``KafkaTopicManager`` is test-only (``tests/helpers/util_kafka.py``,
    no production/runtime caller), the fix here is defense-in-depth at the
    call site rather than a change to the shared helper: verify each topic
    is actually visible via ``describe_topics`` immediately after
    ``create_topic`` returns, and raise with a precise message if it is not.
    """
    async with KafkaTopicManager(bootstrap_servers) as manager:
        for topic in SHARED_CONTRACT_TEST_TOPICS:
            await manager.create_topic(topic, partitions=1, replication_factor=1)
            await _verify_topic_actually_exists(manager, topic)


async def _verify_topic_actually_exists(
    manager: KafkaTopicManager, topic_name: str
) -> None:
    """Fail-closed post-condition readback for ``KafkaTopicManager.create_topic``.

    ``create_topic`` has a known silent-success defect: it can return the
    topic name as if creation succeeded even when the broker created
    nothing (see the caller's docstring above). Confirm the topic is real
    by describing it directly on the admin client before trusting it.

    Raises:
        RuntimeError: if the topic does not actually exist on the broker
            per ``describe_topics``, despite ``create_topic`` reporting
            success.
    """
    admin = manager.admin_client
    if admin is None:
        raise RuntimeError(
            f"Cannot verify topic '{topic_name}': KafkaTopicManager admin "
            f"client was not initialized by create_topic()."
        )

    description = await admin.describe_topics([topic_name])

    topic_info: object | None
    if isinstance(description, dict):
        topic_info = description.get(topic_name)
    elif isinstance(description, list):
        topic_info = next(
            (
                item
                for item in description
                if isinstance(item, dict) and item.get("topic") == topic_name
            ),
            None,
        )
    else:
        topic_info = None

    if topic_info is None:
        raise RuntimeError(
            f"KafkaTopicManager.create_topic() reported success for topic "
            f"'{topic_name}', but describe_topics() shows it does not "
            f"exist on the broker at {manager.bootstrap_servers!r}. This is "
            f"the known silent no-op create defect (mergesweep-0809-"
            f"dualsub-verify F1) -- the real_broker substrate fixture "
            f"cannot proceed without this topic. Verify broker "
            f"reachability and KAFKA_BOOTSTRAP_SERVERS."
        )


def _build_infra_event_bus_substrate_instance(
    param: str,
) -> InfraEventBusSubstrate | None:
    """``None`` return means "real_broker, build separately" (needs await)."""
    if param in CORE_EVENT_BUS_SUBSTRATE_PARAMS:
        return build_core_event_bus_substrate_instance(param)
    if param == "real_broker":
        return None
    raise ValueError(f"Unrecognized event_bus_substrate param: {param!r}")


async def _build_infra_event_bus_substrate(
    param: str,
) -> AsyncIterator[InfraEventBusSubstrate]:
    if param == "real_broker":
        if not _real_broker_available():
            pytest.skip(
                f"real_broker leg requires {_KAFKA_INTEGRATION_TESTS_ENV_VAR}=1 "
                f"(and a reachable broker at KAFKA_BOOTSTRAP_SERVERS, pinned to "
                f"the local Docker Redpanda by the parent conftest's "
                f"pytest_configure hook). Matches the existing gate on "
                f"test_kafka_event_bus_integration.py."
            )
        bootstrap_servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
        await _ensure_shared_contract_test_topics(bootstrap_servers)
        bus: InfraEventBusSubstrate = await _build_real_broker_instance()
    else:
        instance = _build_infra_event_bus_substrate_instance(param)
        assert instance is not None
        bus = instance

    await bus.start()
    try:
        yield bus
    finally:
        try:
            await bus.close()
        except Exception:  # noqa: BLE001 — boundary: test cleanup must not mask the real failure
            pass


#: All three substrates. Kept in the same order as
#: ``CORE_EVENT_BUS_SUBSTRATE_PARAMS`` plus the appended, marked real_broker
#: param -- so a caller that only cares about the shared prefix can still
#: zip against ``CORE_EVENT_BUS_SUBSTRATE_PARAMS`` if needed.
INFRA_EVENT_BUS_SUBSTRATE_PARAMS: tuple[object, ...] = (
    *CORE_EVENT_BUS_SUBSTRATE_PARAMS,
    pytest.param("real_broker", marks=(pytest.mark.integration, pytest.mark.kafka)),
)

#: Fidelity-contract-required substrates: semantic_fake + real_broker (never
#: inmemory -- see CORE_FIDELITY_SUBSTRATE_PARAMS's docstring in core).
INFRA_FIDELITY_SUBSTRATE_PARAMS: tuple[object, ...] = (
    *CORE_FIDELITY_SUBSTRATE_PARAMS,
    pytest.param("real_broker", marks=(pytest.mark.integration, pytest.mark.kafka)),
)


@pytest_asyncio.fixture(params=INFRA_EVENT_BUS_SUBSTRATE_PARAMS)
async def event_bus_substrate(
    request: pytest.FixtureRequest,
) -> AsyncIterator[InfraEventBusSubstrate]:
    """Yield one started substrate per param: inmemory, semantic_fake, real_broker.

    Overrides ``omnibase_core.event_bus.testing.fixture_event_bus_substrate
    .event_bus_substrate`` by name (this ``conftest.py`` is more specific in
    pytest's fixture resolution order) to add the ``real_broker`` leg.
    """
    async for bus in _build_infra_event_bus_substrate(request.param):
        yield bus


@pytest_asyncio.fixture(params=INFRA_FIDELITY_SUBSTRATE_PARAMS)
async def fidelity_event_bus_substrate(
    request: pytest.FixtureRequest,
) -> AsyncIterator[InfraEventBusSubstrate]:
    """Yield one started substrate per fidelity-contract-required param.

    semantic_fake + real_broker. See ``event_bus_substrate`` above and
    ``CORE_FIDELITY_SUBSTRATE_PARAMS`` in omnibase_core for why ``inmemory``
    is excluded.
    """
    async for bus in _build_infra_event_bus_substrate(request.param):
        yield bus


__all__: list[str] = [
    "INFRA_EVENT_BUS_SUBSTRATE_PARAMS",
    "INFRA_FIDELITY_SUBSTRATE_PARAMS",
    "SHARED_CONTRACT_TEST_TOPICS",
    "event_bus_substrate",
    "fidelity_event_bus_substrate",
]
