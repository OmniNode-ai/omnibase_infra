# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16931 — the backward seek must clamp to the LOG-START offset, not to 0.

The defect these tests pin
--------------------------
Measured on the .201 dev lane 2026-09-06T18:xxZ. The chain canary reported
``terminal_missing`` on every 2h run of the day (latest: run 34049551352,
17:44:40Z, ``verdict=terminal_missing``, 1/5 links) while the terminal for that
run's OWN correlation id ``773dde5d-14bf-4a55-858c-950979b7a7ba`` sat on
``onex.evt.omnimarket.delegate-skill-completed.v1`` at offset 214, published
17:45:19.93Z — 21 s after the dispatch and well inside the 110 s readback
window. Reproduced locally: probe ``f278723e-482b-47ec-a2b5-20cbe165cd60``
landed at offset 217 and the same run's receipt said ``not_found`` with
``terminal_readback_records_scanned: 35``.

35 is the whole story. At that moment the two scanned topics stood at:

    delegate-skill-completed.v1   LOG-START 67   HIGH-WATERMARK 218
    delegate-skill-failed.v1      LOG-START  0   HIGH-WATERMARK  35

``_scan_topics_for_correlation`` computed ``start = max(0, end - per_partition)``
= ``max(0, 218 - 250)`` = **0** for the completed topic and seeked there. Offset
0 is below that partition's log start (24 h local retention had already reclaimed
it), so the broker answered OFFSET_OUT_OF_RANGE, and the consumer's
``auto_offset_reset="latest"`` silently repositioned it to the high watermark —
past every record already written, including the one it was looking for. The
scan then read the failed topic end to end (0 → 35, in range) and nothing at all
from the completed topic: exactly 35 records, and a NOT_FOUND that reads
identically to a chain that never emitted.

``max(0, ...)`` is only correct for a partition that has never been truncated.
The clamp has to be the partition's own beginning offset.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.handlers import (
    handler_chain_canary as canary_module,
)

_COMPLETED = "onex.evt.omnimarket.delegate-skill-completed.v1"
_FAILED = "onex.evt.omnimarket.delegate-skill-failed.v1"


class _TopicPartition:
    """Minimal stand-in for aiokafka's TopicPartition (hashable, ordered)."""

    def __init__(self, topic: str, partition: int) -> None:
        self.topic = topic
        self.partition = partition

    def __hash__(self) -> int:
        return hash((self.topic, self.partition))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TopicPartition) and (self.topic, self.partition) == (
            other.topic,
            other.partition,
        )

    def __repr__(self) -> str:
        return f"TP({self.topic},{self.partition})"


class _Record:
    def __init__(self, value: bytes) -> None:
        self.value = value


class _BrokerBackedConsumer:
    """Consumer that reproduces the broker's out-of-range reset semantics.

    A seek below a partition's log start is not an error the caller sees: the
    broker rejects the fetch and ``auto_offset_reset="latest"`` moves the
    position to the high watermark. That silent jump is the whole defect, so
    the fake models it rather than raising.
    """

    def __init__(self, *topics: str, **kwargs: Any) -> None:
        self._topics = topics
        self._auto_offset_reset = kwargs.get("auto_offset_reset", "latest")
        self._log: dict[_TopicPartition, tuple[int, list[_Record]]] = {}
        self._position: dict[_TopicPartition, int] = {}
        self.started = False

    def load(self, tp: _TopicPartition, log_start: int, records: list[bytes]) -> None:
        self._log[tp] = (log_start, [_Record(v) for v in records])

    async def start(self) -> None:
        self.started = True
        for tp, (log_start, _records) in self._log.items():
            self._position[tp] = log_start

    async def stop(self) -> None:
        self.started = False

    def assignment(self) -> set[_TopicPartition]:
        return set(self._log)

    async def end_offsets(
        self, partitions: list[_TopicPartition]
    ) -> dict[_TopicPartition, int]:
        return {tp: self._log[tp][0] + len(self._log[tp][1]) for tp in partitions}

    async def beginning_offsets(
        self, partitions: list[_TopicPartition]
    ) -> dict[_TopicPartition, int]:
        return {tp: self._log[tp][0] for tp in partitions}

    def seek(self, tp: _TopicPartition, offset: int) -> None:
        log_start, records = self._log[tp]
        end = log_start + len(records)
        if offset < log_start:
            # OFFSET_OUT_OF_RANGE -> auto_offset_reset. This is the silent
            # jump the live lane took.
            self._position[tp] = (
                end if self._auto_offset_reset == "latest" else log_start
            )
            return
        self._position[tp] = min(offset, end)

    async def position(self, tp: _TopicPartition) -> int:
        return self._position[tp]

    async def getmany(
        self, timeout_ms: int = 0, max_records: int = 500
    ) -> dict[_TopicPartition, list[_Record]]:
        await asyncio.sleep(0)
        batches: dict[_TopicPartition, list[_Record]] = {}
        for tp, (log_start, records) in self._log.items():
            pos = self._position[tp]
            end = log_start + len(records)
            if pos >= end:
                continue
            take = records[pos - log_start : pos - log_start + max_records]
            self._position[tp] = pos + len(take)
            batches[tp] = take
        return batches


def _install(monkeypatch: pytest.MonkeyPatch, consumer: _BrokerBackedConsumer) -> None:
    """Serve the fake wherever the scan imports ``aiokafka.AIOKafkaConsumer``."""
    import aiokafka

    monkeypatch.setattr(
        aiokafka, "AIOKafkaConsumer", lambda *a, **k: consumer, raising=True
    )
    monkeypatch.setattr(
        canary_module,
        "build_aiokafka_auth_kwargs_from_env",
        dict,
        raising=False,
    )
    from omnibase_infra.event_bus import kafka_auth

    monkeypatch.setattr(
        kafka_auth, "build_aiokafka_auth_kwargs_from_env", dict, raising=True
    )


def _truncated_lane_consumer(needle: str) -> _BrokerBackedConsumer:
    """The live shape: completed truncated to log-start 67, failed intact at 0."""
    consumer = _BrokerBackedConsumer(_COMPLETED, _FAILED)
    completed_tp = _TopicPartition(_COMPLETED, 0)
    failed_tp = _TopicPartition(_FAILED, 0)
    # 67..217 inclusive == 151 retained records; the needle sits at 217, the
    # last one, exactly as the live reproduction did.
    completed = [f'{{"correlation_id":"filler-{i}"}}'.encode() for i in range(67, 217)]
    completed.append(f'{{"correlation_id":"{needle}"}}'.encode())
    consumer.load(completed_tp, 67, completed)
    consumer.load(
        failed_tp, 0, [f'{{"correlation_id":"other-{i}"}}'.encode() for i in range(35)]
    )
    return consumer


@pytest.mark.unit
@pytest.mark.asyncio
async def test_seek_clamps_to_log_start_so_a_truncated_topic_is_still_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The needle is retained and inside the lookback — the scan must find it.

    ``end - per_partition`` is 218 - 250 = -32 here. Clamped to 0 it is out of
    range and the partition is skipped entirely; clamped to the partition's own
    log start (67) every retained record is read.
    """
    needle = "773dde5d-14bf-4a55-858c-950979b7a7ba"
    consumer = _truncated_lane_consumer(needle)
    _install(monkeypatch, consumer)

    topic, scanned, error = await canary_module._scan_topics_for_correlation(
        "broker:19092",
        (_COMPLETED, _FAILED),
        needle,
        500,
        5.0,
        wait_for_arrival=False,
    )

    assert error == ""
    assert topic == _COMPLETED, (
        "a retained terminal inside the lookback window read as missing — the "
        f"backward seek was clamped to 0 below log-start 67 (scanned={scanned})"
    )
    # 151 retained completed records + 35 failed records. The live failure
    # scanned exactly 35: the completed partition contributed nothing.
    assert scanned > 35


@pytest.mark.unit
@pytest.mark.asyncio
async def test_absent_correlation_id_on_a_truncated_topic_is_still_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clamp must not turn every scan into a hit — negative control."""
    consumer = _truncated_lane_consumer("773dde5d-14bf-4a55-858c-950979b7a7ba")
    _install(monkeypatch, consumer)

    topic, scanned, error = await canary_module._scan_topics_for_correlation(
        "broker:19092",
        (_COMPLETED, _FAILED),
        "00000000-0000-4000-8000-000000000000",
        500,
        5.0,
        wait_for_arrival=False,
    )

    assert error == ""
    assert topic == ""
    assert scanned > 35


@pytest.mark.unit
@pytest.mark.asyncio
async def test_untruncated_topic_still_honours_the_record_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partition that starts at 0 keeps the existing bounded lookback.

    The clamp is ``max(log_start, end - per_partition)``; when the log has not
    been truncated that is still ``end - per_partition``, so a record older than
    the budget stays out of the window and the scan stays bounded.
    """
    consumer = _BrokerBackedConsumer(_COMPLETED, _FAILED)
    completed_tp = _TopicPartition(_COMPLETED, 0)
    failed_tp = _TopicPartition(_FAILED, 0)
    old = "11111111-1111-4111-8111-111111111111"
    records = [f'{{"correlation_id":"{old}"}}'.encode()]
    records += [f'{{"correlation_id":"filler-{i}"}}'.encode() for i in range(1, 1000)]
    consumer.load(completed_tp, 0, records)
    consumer.load(failed_tp, 0, [])
    _install(monkeypatch, consumer)

    topic, scanned, error = await canary_module._scan_topics_for_correlation(
        "broker:19092",
        (_COMPLETED, _FAILED),
        old,
        100,
        5.0,
        wait_for_arrival=False,
    )

    assert error == ""
    assert topic == "", "a record older than the lookback budget must stay out of it"
    assert scanned <= 100
