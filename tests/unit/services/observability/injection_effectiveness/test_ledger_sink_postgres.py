# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for LedgerSinkInjectionEffectivenessPostgres.

Tests ledger append operations with mocked asyncpg pool.

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.services.observability.injection_effectiveness.ledger_sink_postgres import (
    LEDGER_SOURCE,
    LedgerSinkInjectionEffectivenessPostgres,
)

pytestmark = pytest.mark.unit


class TestAppendSessionEntry:
    """Tests for append_session_entry()."""

    @pytest.mark.asyncio
    async def test_returns_ledger_id_on_insert(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Returns ledger_entry_id when insert succeeds."""
        ledger_id = uuid4()
        mock_pool._test_conn.fetchval = AsyncMock(return_value=ledger_id)

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        result = await sink.append_session_entry(
            session_id=sample_session_id,
            correlation_id=sample_correlation_id,
            event_type="context_utilization",
            event_payload=b'{"score": 0.85}',
            kafka_topic="onex.evt.omniclaude.context-utilization.v1",
            kafka_partition=0,
            kafka_offset=42,
        )

        assert result == ledger_id

    @pytest.mark.asyncio
    async def test_returns_none_on_duplicate(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Returns None when ON CONFLICT triggers (duplicate)."""
        mock_pool._test_conn.fetchval = AsyncMock(return_value=None)

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        result = await sink.append_session_entry(
            session_id=sample_session_id,
            correlation_id=sample_correlation_id,
            event_type="context_utilization",
            event_payload=b'{"score": 0.85}',
            kafka_topic="onex.evt.omniclaude.context-utilization.v1",
            kafka_partition=0,
            kafka_offset=42,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_passes_correct_parameters(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Verifies correct SQL parameters are passed."""
        mock_pool._test_conn.fetchval = AsyncMock(return_value=uuid4())

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        await sink.append_session_entry(
            session_id=sample_session_id,
            correlation_id=sample_correlation_id,
            event_type="agent_match",
            event_payload=b'{"match_score": 1.0}',
            kafka_topic="onex.evt.omniclaude.agent-match.v1",
            kafka_partition=1,
            kafka_offset=99,
        )

        call_args = mock_pool._test_conn.fetchval.call_args
        sql = call_args.args[0]
        params = call_args.args[1:]

        assert "INSERT INTO event_ledger" in sql
        assert "ON CONFLICT" in sql
        assert params[0] == "onex.evt.omniclaude.agent-match.v1"  # topic
        assert params[1] == 1  # partition
        assert params[2] == 99  # kafka_offset
        assert params[3] == str(sample_session_id).encode()  # event_key
        assert params[4] == b'{"match_score": 1.0}'  # event_value
        # onex_headers is JSON string
        headers = json.loads(params[5])
        assert headers["source"] == LEDGER_SOURCE
        assert headers["event_type"] == "agent_match"
        assert params[6] == sample_correlation_id  # correlation_id
        assert params[7] == "agent_match"  # event_type
        assert params[8] == LEDGER_SOURCE  # source

    @pytest.mark.asyncio
    async def test_rejects_bool_as_kafka_partition(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Raises TypeError when kafka_partition is bool (bool is subclass of int)."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="kafka_partition must be int"):
            await sink.append_session_entry(
                session_id=sample_session_id,
                correlation_id=sample_correlation_id,
                event_type="context_utilization",
                event_payload=b"{}",
                kafka_topic="test.topic",
                kafka_partition=True,
                kafka_offset=0,
            )

    @pytest.mark.asyncio
    async def test_rejects_negative_kafka_partition(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Raises ValueError when kafka_partition is negative."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="kafka_partition must be >= 0"):
            await sink.append_session_entry(
                session_id=sample_session_id,
                correlation_id=sample_correlation_id,
                event_type="context_utilization",
                event_payload=b"{}",
                kafka_topic="test.topic",
                kafka_partition=-1,
                kafka_offset=0,
            )

    @pytest.mark.asyncio
    async def test_rejects_negative_kafka_offset(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Raises ValueError when kafka_offset is negative."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="kafka_offset must be >= 0"):
            await sink.append_session_entry(
                session_id=sample_session_id,
                correlation_id=sample_correlation_id,
                event_type="context_utilization",
                event_payload=b"{}",
                kafka_topic="test.topic",
                kafka_partition=0,
                kafka_offset=-5,
            )

    @pytest.mark.asyncio
    async def test_rejects_bool_as_kafka_offset(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Raises TypeError when kafka_offset is bool."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="kafka_offset must be int"):
            await sink.append_session_entry(
                session_id=sample_session_id,
                correlation_id=sample_correlation_id,
                event_type="context_utilization",
                event_payload=b"{}",
                kafka_topic="test.topic",
                kafka_partition=0,
                kafka_offset=False,
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_kafka_topic(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Raises TypeError when kafka_topic is empty string."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="kafka_topic must be a non-empty str"):
            await sink.append_session_entry(
                session_id=sample_session_id,
                correlation_id=sample_correlation_id,
                event_type="context_utilization",
                event_payload=b"{}",
                kafka_topic="",
                kafka_partition=0,
                kafka_offset=0,
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_event_type(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Raises TypeError when event_type is empty string."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="event_type must be a non-empty str"):
            await sink.append_session_entry(
                session_id=sample_session_id,
                correlation_id=sample_correlation_id,
                event_type="",
                event_payload=b"{}",
                kafka_topic="test.topic",
                kafka_partition=0,
                kafka_offset=0,
            )

    @pytest.mark.asyncio
    async def test_rejects_non_str_kafka_topic(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Raises TypeError when kafka_topic is not a string."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="kafka_topic must be a non-empty str"):
            await sink.append_session_entry(
                session_id=sample_session_id,
                correlation_id=sample_correlation_id,
                event_type="context_utilization",
                event_payload=b"{}",
                kafka_topic=123,  # type: ignore[arg-type]
                kafka_partition=0,
                kafka_offset=0,
            )

    @pytest.mark.asyncio
    async def test_rejects_non_str_event_type(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Raises TypeError when event_type is not a string."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="event_type must be a non-empty str"):
            await sink.append_session_entry(
                session_id=sample_session_id,
                correlation_id=sample_correlation_id,
                event_type=42,  # type: ignore[arg-type]
                event_payload=b"{}",
                kafka_topic="test.topic",
                kafka_partition=0,
                kafka_offset=0,
            )

    @pytest.mark.asyncio
    async def test_sets_statement_timeout(
        self, mock_pool: MagicMock, sample_session_id, sample_correlation_id
    ) -> None:
        """Verifies statement_timeout is set on connection."""
        mock_pool._test_conn.fetchval = AsyncMock(return_value=uuid4())

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool, query_timeout=10.0)
        await sink.append_session_entry(
            session_id=sample_session_id,
            correlation_id=sample_correlation_id,
            event_type="context_utilization",
            event_payload=b"{}",
            kafka_topic="test.topic",
            kafka_partition=0,
            kafka_offset=0,
        )

        mock_pool._test_conn.execute.assert_any_call(
            "SET LOCAL statement_timeout = $1", "10000"
        )


class TestAppendBatch:
    """Tests for append_batch()."""

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_batch(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Returns 0 when entries list is empty."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        result = await sink.append_batch([], sample_correlation_id)
        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_count_for_batch(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Returns count of entries in batch."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "context_utilization",
                "event_payload": b'{"score": 0.85}',
                "kafka_topic": "onex.evt.omniclaude.context-utilization.v1",
                "kafka_partition": 0,
                "kafka_offset": i,
            }
            for i in range(3)
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        result = await sink.append_batch(entries, sample_correlation_id)
        assert result == 3

    @pytest.mark.asyncio
    async def test_calls_executemany(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Verifies executemany is called for batch writes."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "latency_breakdown",
                "event_payload": b'{"latency_ms": 350}',
                "kafka_topic": "onex.evt.omniclaude.latency-breakdown.v1",
                "kafka_partition": 0,
                "kafka_offset": 0,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        await sink.append_batch(entries, sample_correlation_id)

        mock_pool._test_conn.executemany.assert_called_once()
        call_args = mock_pool._test_conn.executemany.call_args
        sql = call_args.args[0]
        assert "INSERT INTO event_ledger" in sql
        assert "ON CONFLICT" in sql

    @pytest.mark.asyncio
    async def test_rejects_entries_with_missing_keys(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises ValueError when entry is missing required keys."""
        entries = [{"session_id": uuid4(), "event_type": "test"}]  # missing keys

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="Entry 0 missing required keys"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_entries_with_wrong_partition_type(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises TypeError when kafka_partition is not int."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "context_utilization",
                "event_payload": b"{}",
                "kafka_topic": "test.topic",
                "kafka_partition": "0",  # string instead of int
                "kafka_offset": 0,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="kafka_partition must be int"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_entries_with_wrong_offset_type(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises TypeError when kafka_offset is not int."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "context_utilization",
                "event_payload": b"{}",
                "kafka_topic": "test.topic",
                "kafka_partition": 0,
                "kafka_offset": "42",  # string instead of int
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="kafka_offset must be int"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_entries_with_wrong_session_id_type(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises TypeError when session_id is not UUID."""
        entries = [
            {
                "session_id": "not-a-uuid",
                "event_type": "context_utilization",
                "event_payload": b"{}",
                "kafka_topic": "test.topic",
                "kafka_partition": 0,
                "kafka_offset": 0,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="session_id must be UUID"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_entries_with_wrong_event_type_type(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises TypeError when event_type is not str."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": 123,
                "event_payload": b"{}",
                "kafka_topic": "test.topic",
                "kafka_partition": 0,
                "kafka_offset": 0,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="event_type must be a non-empty str"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_entries_with_wrong_event_payload_type(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises TypeError when event_payload is not bytes."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "context_utilization",
                "event_payload": '{"score": 0.85}',  # string instead of bytes
                "kafka_topic": "test.topic",
                "kafka_partition": 0,
                "kafka_offset": 0,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="event_payload must be bytes"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_entries_with_wrong_kafka_topic_type(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises TypeError when kafka_topic is not str."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "context_utilization",
                "event_payload": b"{}",
                "kafka_topic": 123,  # int instead of str
                "kafka_partition": 0,
                "kafka_offset": 0,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="kafka_topic must be a non-empty str"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_empty_kafka_topic_in_batch(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises TypeError when kafka_topic is empty string in batch."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "context_utilization",
                "event_payload": b"{}",
                "kafka_topic": "",
                "kafka_partition": 0,
                "kafka_offset": 0,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="kafka_topic must be a non-empty str"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_empty_event_type_in_batch(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises TypeError when event_type is empty string in batch."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "",
                "event_payload": b"{}",
                "kafka_topic": "test.topic",
                "kafka_partition": 0,
                "kafka_offset": 0,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="event_type must be a non-empty str"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_bool_as_kafka_partition(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises TypeError when kafka_partition is bool (bool is subclass of int)."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "context_utilization",
                "event_payload": b"{}",
                "kafka_topic": "test.topic",
                "kafka_partition": True,  # bool, not int
                "kafka_offset": 0,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(TypeError, match="kafka_partition must be int"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_negative_kafka_partition(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises ValueError when kafka_partition is negative."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "context_utilization",
                "event_payload": b"{}",
                "kafka_topic": "test.topic",
                "kafka_partition": -1,
                "kafka_offset": 0,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="kafka_partition must be >= 0"):
            await sink.append_batch(entries, sample_correlation_id)

    @pytest.mark.asyncio
    async def test_rejects_negative_kafka_offset(
        self, mock_pool: MagicMock, sample_correlation_id
    ) -> None:
        """Raises ValueError when kafka_offset is negative."""
        entries = [
            {
                "session_id": uuid4(),
                "event_type": "context_utilization",
                "event_payload": b"{}",
                "kafka_topic": "test.topic",
                "kafka_partition": 0,
                "kafka_offset": -5,
            }
        ]

        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        with pytest.raises(ValueError, match="kafka_offset must be >= 0"):
            await sink.append_batch(entries, sample_correlation_id)


class TestLedgerSinkInitialization:
    """Tests for initialization and configuration."""

    def test_default_timeout(self, mock_pool: MagicMock) -> None:
        """Uses default timeout when not specified."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool)
        assert sink._query_timeout == 30.0

    def test_custom_timeout(self, mock_pool: MagicMock) -> None:
        """Uses custom timeout when specified."""
        sink = LedgerSinkInjectionEffectivenessPostgres(mock_pool, query_timeout=5.0)
        assert sink._query_timeout == 5.0
