# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for InfraRoutingDecisionsConsumer (OMN-8692).

Tests:
    - Config: defaults, env prefix
    - Consumer: message parsing, batch processing, offset tracking
    - Writer: batch write, circuit breaker state
    - Health check: HEALTHY, DEGRADED, UNHEALTHY
    - mask_dsn_password: password masking utility
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.services.observability.infra_routing_decisions.config import (
    ConfigInfraRoutingDecisionsConsumer,
)
from omnibase_infra.services.observability.infra_routing_decisions.consumer import (
    EnumHealthStatus,
    InfraRoutingDecisionsConsumer,
    mask_dsn_password,
)
from omnibase_infra.services.observability.infra_routing_decisions.writer_postgres import (
    WriterInfraRoutingDecisionsPostgres,
)

# =============================================================================
# Helpers
# =============================================================================


def make_mock_consumer_record(
    topic: str,
    partition: int,
    offset: int,
    value: dict[str, object],
) -> MagicMock:
    record = MagicMock()
    record.topic = topic
    record.partition = partition
    record.offset = offset
    record.value = json.dumps(value).encode("utf-8")
    return record


def make_routing_decided_event() -> dict[str, object]:
    return {
        "correlation_id": str(uuid4()),
        "selected_provider": "anthropic",
        "selected_tier": "claude",
        "selected_model": "claude-opus-4-6",
        "selection_mode": "round_robin",
        "fallback_indicator": False,
        "is_fallback": False,
        "reason": "primary selection",
        "candidates_evaluated": 3,
        "candidate_providers": ["anthropic", "openai", "local"],
        "task_type": "code",
        "session_id": str(uuid4()),
        "latency_ms": 42.5,
    }


_TOPIC = "onex.evt.omnibase-infra.routing-decided.v1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> ConfigInfraRoutingDecisionsConsumer:
    return ConfigInfraRoutingDecisionsConsumer(
        kafka_bootstrap_servers="localhost:19092",
        postgres_dsn="postgresql://test:test@localhost:5432/test",
        batch_size=10,
        batch_timeout_ms=500,
        health_check_port=18097,
    )


@pytest.fixture
def consumer(
    mock_config: ConfigInfraRoutingDecisionsConsumer,
) -> InfraRoutingDecisionsConsumer:
    return InfraRoutingDecisionsConsumer(mock_config)


# =============================================================================
# Config Tests
# =============================================================================


@pytest.mark.unit
class TestConfigInfraRoutingDecisionsConsumer:
    def test_default_topic(self) -> None:
        config = ConfigInfraRoutingDecisionsConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        assert _TOPIC in config.topics
        assert len(config.topics) == 1

    def test_default_health_check_port(self) -> None:
        config = ConfigInfraRoutingDecisionsConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        assert config.health_check_port == 8097

    def test_default_group_id(self) -> None:
        config = ConfigInfraRoutingDecisionsConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        assert config.kafka_group_id == "infra-routing-decisions-postgres"


# =============================================================================
# mask_dsn_password Tests
# =============================================================================


@pytest.mark.unit
class TestMaskDsnPassword:
    def test_masks_password(self) -> None:
        dsn = "postgresql://user:secret@localhost:5432/db"
        result = mask_dsn_password(dsn)
        assert "secret" not in result
        assert "***" in result

    def test_no_password_unchanged(self) -> None:
        dsn = "postgresql://localhost:5432/db"
        assert mask_dsn_password(dsn) == dsn

    def test_invalid_dsn_returns_as_is(self) -> None:
        assert mask_dsn_password("not-a-url") == "not-a-url"


# =============================================================================
# Consumer Message Parsing Tests
# =============================================================================


@pytest.mark.unit
class TestInfraRoutingDecisionsConsumerParsing:
    def test_parse_valid_dict_message(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        payload = make_routing_decided_event()
        record = make_mock_consumer_record(_TOPIC, 0, 0, payload)
        result = consumer._parse_message(record)
        assert result is not None
        assert result["selected_provider"] == payload["selected_provider"]

    def test_parse_invalid_json_returns_none(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        record = MagicMock()
        record.topic = _TOPIC
        record.partition = 0
        record.offset = 0
        record.value = b"not-json"
        assert consumer._parse_message(record) is None

    def test_parse_array_wrapped_legacy(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        payload = make_routing_decided_event()
        record = MagicMock()
        record.topic = _TOPIC
        record.partition = 0
        record.offset = 0
        record.value = json.dumps([payload]).encode("utf-8")
        result = consumer._parse_message(record)
        assert result is not None
        assert result["selected_provider"] == payload["selected_provider"]

    def test_parse_multi_item_list_returns_none(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        record = MagicMock()
        record.topic = _TOPIC
        record.partition = 0
        record.offset = 0
        record.value = json.dumps(
            [make_routing_decided_event(), make_routing_decided_event()]
        ).encode("utf-8")
        assert consumer._parse_message(record) is None


# =============================================================================
# Writer Tests
# =============================================================================


@pytest.mark.unit
class TestWriterInfraRoutingDecisionsPostgres:
    def test_empty_batch_returns_zero(self) -> None:
        pool = MagicMock()
        writer = WriterInfraRoutingDecisionsPostgres(pool)

        import asyncio

        result = asyncio.run(writer.write_routing_decisions([]))
        assert result == 0

    def test_get_circuit_breaker_state_returns_dict(self) -> None:
        pool = MagicMock()
        writer = WriterInfraRoutingDecisionsPostgres(pool)
        state = writer.get_circuit_breaker_state()
        assert isinstance(state, dict)


# =============================================================================
# Batch Processing Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestInfraRoutingDecisionsConsumerBatchProcessing:
    async def test_process_batch_records_offsets(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        records = [
            make_mock_consumer_record(_TOPIC, 0, 0, make_routing_decided_event()),
            make_mock_consumer_record(_TOPIC, 0, 1, make_routing_decided_event()),
            make_mock_consumer_record(_TOPIC, 1, 5, make_routing_decided_event()),
        ]

        mock_writer = AsyncMock()
        mock_writer.write_routing_decisions = AsyncMock(return_value=3)
        mock_writer.get_circuit_breaker_state = MagicMock(return_value={})
        consumer._writer = mock_writer

        from aiokafka import TopicPartition

        committed = await consumer._process_batch(records)

        assert committed[TopicPartition(_TOPIC, 0)] == 1
        assert committed[TopicPartition(_TOPIC, 1)] == 5

    async def test_process_batch_excludes_failed_partitions(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        records = [
            make_mock_consumer_record(_TOPIC, 0, 0, make_routing_decided_event()),
        ]

        mock_writer = AsyncMock()
        mock_writer.write_routing_decisions = AsyncMock(
            side_effect=RuntimeError("DB error")
        )
        mock_writer.get_circuit_breaker_state = MagicMock(return_value={})
        consumer._writer = mock_writer

        committed = await consumer._process_batch(records)
        assert len(committed) == 0

    async def test_process_batch_skips_unparseable_messages(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        bad_record = MagicMock()
        bad_record.topic = _TOPIC
        bad_record.partition = 0
        bad_record.offset = 0
        bad_record.value = b"invalid-json"

        mock_writer = AsyncMock()
        mock_writer.write_routing_decisions = AsyncMock(return_value=0)
        mock_writer.get_circuit_breaker_state = MagicMock(return_value={})
        consumer._writer = mock_writer

        consumer.config = ConfigInfraRoutingDecisionsConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
            dlq_enabled=False,
        )

        committed = await consumer._process_batch([bad_record])
        assert len(committed) == 0


# =============================================================================
# Health Check Tests
# =============================================================================


@pytest.mark.unit
class TestInfraRoutingDecisionsConsumerHealthCheck:
    def test_unhealthy_when_not_running(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        consumer._running = False
        response, http_code = consumer._build_health_response()
        assert response["status"] == str(EnumHealthStatus.UNHEALTHY)
        assert http_code == 503

    def test_healthy_when_running_and_no_messages(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        consumer._running = True
        consumer.metrics.last_poll_at = datetime.now(UTC)
        consumer.metrics.messages_received = 0

        response, http_code = consumer._build_health_response()
        assert response["status"] == str(EnumHealthStatus.HEALTHY)
        assert http_code == 200
        assert response["idle"] is True

    def test_degraded_when_no_polls(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        consumer._running = True
        consumer.metrics.last_poll_at = None

        response, http_code = consumer._build_health_response()
        assert response["status"] == str(EnumHealthStatus.DEGRADED)
        assert http_code == 503
