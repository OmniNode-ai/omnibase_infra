# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23024: the withhold bound fires for a refusal whose message carries the
event's age.

The retry bound (OMN-17379) counts consecutive IDENTICAL refusals of one record,
and "identical" is the failure fingerprint. The fingerprint hashed the rendered
message, and omnimarket's unknown-tenant refusal renders
``the event is <N>s old`` -- N grows by the redelivery interval on every
delivery. Every redelivery therefore produced a new fingerprint, the count reset
to 1 each time, and the bound never fired for the one refusal it was written for
(read on the dev runtime 2026-11-07: one unknown-tenant event redelivered for
hours, 364 records behind it on delegate-skill-completed).

The fingerprint now keeps the stable parts only: the exception type, the cause's
type, and the message with the age figure masked. Each test that asserts the
bound fires has a paired control that asserts a refusal that really changed still
resets the count.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from tests.unit.event_bus.test_omn17379_projection_wedge_bounded import (
    TEST_OFFSET,
    _redeliver,
)


@pytest.fixture
def kafka_config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")


@pytest.fixture
def mock_producer() -> AsyncMock:
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    return producer


#: The refusal as omnimarket renders it for an event older than the catch-up
#: window (``tenant_mirror_catchup_policy.lag_refusal_message``), with the age
#: slot left open. Everything else in it is fixed for one event.
_UNKNOWN_TENANT_TEMPLATE = (
    "(TenantRegistryResolutionError: OMN-22478: tenant_registry_mirror holds no "
    "row for {tenant!r} and the event is {age}s old, past the 30s catch-up "
    "window, so it was not waited for: a mint in flight when it was produced "
    "has landed by now. The mirror's watermark is 2026-11-07T17:40:00+00:00 "
    "while the event was produced at 2026-11-07T17:45:07+00:00; a quiet "
    "registry keeps the watermark behind every event, so this reads as lag but "
    "is an unknown tenant unless node_projection_tenant_registry is not "
    "consuming. No identity will be invented or defaulted for it.)"
)


def _unknown_tenant(*, age: int, tenant: str = "ghost-tenant") -> Exception:
    return ProjectionNotMaterializedError(
        _UNKNOWN_TENANT_TEMPLATE.format(tenant=tenant, age=age),
        projection_type="HandlerProjectionDelegation",
    )


class TestAReplayedRefusalReachesTheBound:
    @pytest.mark.asyncio
    async def test_a_replayed_unknown_tenant_refusal_is_dead_lettered_at_the_bound(
        self,
        kafka_config: ModelKafkaEventBusConfig,
        mock_producer: AsyncMock,
    ) -> None:
        """The failing control: same record, same refusal, a growing age.

        On the unfixed fingerprint every delivery hashes differently, the count
        never passes 1, nothing is dead-lettered and the partition is rewound
        ``bound + 1`` times.
        """
        bound = kafka_config.projection_withhold_max_redeliveries
        errors = [_unknown_tenant(age=4000 + 7 * i) for i in range(bound + 1)]
        consumer, dlq_calls = await _redeliver(kafka_config, mock_producer, errors)
        assert len(dlq_calls) == 1, (
            f"after {bound + 1} refusals of one record that differ only in the "
            "event's age the partition must be released; the age made each one "
            f"look new (seeks={len(consumer.seek_calls)}, dlq={len(dlq_calls)})"
        )
        assert len(consumer.seek_calls) == bound
        assert str(TEST_OFFSET) in str(dlq_calls[0].get("validation_detail"))

    @pytest.mark.asyncio
    async def test_the_bound_is_not_reached_early(
        self,
        kafka_config: ModelKafkaEventBusConfig,
        mock_producer: AsyncMock,
    ) -> None:
        bound = kafka_config.projection_withhold_max_redeliveries
        errors = [_unknown_tenant(age=4000 + 7 * i) for i in range(bound)]
        consumer, dlq_calls = await _redeliver(kafka_config, mock_producer, errors)
        assert dlq_calls == []
        assert len(consumer.seek_calls) == bound


class TestARefusalThatReallyChangedStillResets:
    @pytest.mark.asyncio
    async def test_a_different_tenant_resets_the_count(
        self,
        kafka_config: ModelKafkaEventBusConfig,
        mock_producer: AsyncMock,
    ) -> None:
        """Masking the age must not mask the rest: a refusal about another
        tenant is a different failure and restarts the count."""
        bound = kafka_config.projection_withhold_max_redeliveries
        errors = [
            _unknown_tenant(age=4000 + i, tenant=f"tenant-{i % 2}")
            for i in range(bound * 2 + 2)
        ]
        consumer, dlq_calls = await _redeliver(kafka_config, mock_producer, errors)
        assert dlq_calls == []
        assert len(consumer.seek_calls) == len(errors)


class TestTheFingerprintItself:
    def test_age_only_differences_share_a_fingerprint(self) -> None:
        fingerprint = EventBusKafka._projection_failure_fingerprint
        assert fingerprint(_unknown_tenant(age=4000)) == fingerprint(
            _unknown_tenant(age=4377)
        )

    def test_a_different_tenant_has_a_different_fingerprint(self) -> None:
        fingerprint = EventBusKafka._projection_failure_fingerprint
        assert fingerprint(_unknown_tenant(age=4000, tenant="a")) != fingerprint(
            _unknown_tenant(age=4000, tenant="b")
        )

    def test_a_different_exception_type_has_a_different_fingerprint(self) -> None:
        fingerprint = EventBusKafka._projection_failure_fingerprint
        assert fingerprint(ValueError("x")) != fingerprint(RuntimeError("x"))
