# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Infra-side golden chain: steel topic -> dispatch -> event_ledger row (OMN-15169).

Scope (hostile finding #1, plan `2026-07-26-steel-node-dispatch-integration-
plan.md` §2 step 8): this test proves ONLY what `omnibase_infra` can own --
topic -> dispatch -> a real `event_ledger` row. It does NOT drive
`steel_onslaught`'s own code (a private, separate repo) and does NOT claim
that half of the proof. The steel-side driver test
(`steel_onslaught/tests/live/test_omn15170_live_driver.py`, OMN-15170) is the
other, independently-owned half; it already passed live 2026-07-26 (real
match, real Kafka publish, correlation_id `ad230e9e-b336-4599-b870-
f6746033be47` consumed back at offset 5).

Two checkpoints, per the plan's terminal-checkpoint requirement (never
"event published" alone -- OMN-15002/OMN-15006 already proved that class of
false-positive):

1. NEGATIVE (allowlist-gate proof, OMN-15002 precedent). `node_ledger_
   projection_compute`'s `subscribe_topics` allowlist is the only thing that
   makes this topic's events durable in `event_ledger` -- widening
   `subscribe_topics` without dispatch is the documented NO_DISPATCHER class
   (OMN-14594). Before the OMN-15168 paired diff was deployed to the
   stability-test lane, `event_ledger` held ZERO rows for
   `onex.evt.steel-onslaught.match-terminal.v1` even though the topic already
   carried 14 real events (offsets 0-13, produced by steel's own OMN-15170
   live-driver runs -- a genuine, unplanned natural experiment). This is
   captured as durable, structured evidence in
   `tests/fixtures/golden_chains/steel_dispatch_ledger_negative_evidence.json`
   (recorded 2026-07-26, verified via `docker exec ... cat contract.yaml` on
   the deployed `omninode-stability-test-runtime-effects` container showing
   `contract_version: 1.1.0` with no steel topic, plus a direct `psql` count
   against `100.109.203.94:15436`). `test_negative_case_...` below asserts
   this recorded evidence is internally consistent; it does not, and cannot,
   re-derive the historical fact live (the deployed contract has since been
   refreshed to include the topic -- see below -- so the negative window is
   permanently closed and is recorded, not reproduced).

2. POSITIVE (topic -> dispatch -> ledger row). `test_golden_chain_positive_
   topic_to_ledger_row` publishes ONE synthetic event with a freshly minted
   `correlation_id` to the real stability-test Kafka topic, then polls
   `event_ledger` directly via SQL (never `ledger.query` RPC -- direct SQL is
   explicitly sanctioned by the governing plan §2 step 8) for a row carrying
   that `correlation_id`.

KNOWN LIVE BLOCKER as of 2026-07-26 (OMN-15215, filed this session, blocks
this ticket): a stability-lane warm refresh
(`omnibase_infra/scripts/runtime_build/refresh_stability_lane.sh --ref
origin/dev --execute`) was run and its own health-gate PASSED; the deployed
`node_ledger_projection_compute` contract on
`omninode-stability-test-runtime-effects` was re-verified afterward to
include the steel topic (`subscribe_topics` + paired `handler_routing`
entry, confirmed via `docker exec ... cat contract.yaml`). Despite this, two
independent synthetic publishes (one before, one after an additional
targeted restart of just `omninode-stability-test-runtime`) produced zero
`event_ledger` rows within a 60s poll window each, and `rpk group list`
shows zero live Kafka consumer groups for this contract on ANY of the 19
topics added since OMN-15006 (not just the OMN-15168 steel topic) -- ruling
out an OMN-15168-specific cause. `test_golden_chain_positive_topic_to_
ledger_row` is written to the correct, intended behavior and WILL currently
fail (not skip) if run live against the stability-test lane today with a
message citing OMN-15215; it is expected to pass once OMN-15215 clears. It
is skipped automatically (like every test in this module) when the lane is
unreachable, e.g. in CI, which cannot reach the private `.201`/Tailscale
network.

Run: uv run pytest tests/integration/runtime/test_steel_dispatch_golden_chain_live_runtime.py -v -s -m integration
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.enums.generated.enum_steel_onslaught_topic import (
    EnumSteelOnslaughtTopic,
)

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "golden_chains"

# The stability-test lane's advertised Kafka listener (matches the literal
# steel_onslaught's own OMN-15170 live-driver test dials -- verified live,
# not a hostname substitute; see that test's module docstring for why a
# hostname would break on the post-bootstrap metadata-driven reconnect).
# Overridable via STABILITY_TEST_KAFKA_BOOTSTRAP_SERVERS for any other lane.
_DEFAULT_KAFKA_BOOTSTRAP = "100.109.203.94:39092"  # sanitize-ok
_ENV_KAFKA_BOOTSTRAP = "STABILITY_TEST_KAFKA_BOOTSTRAP_SERVERS"
KAFKA_BOOTSTRAP = os.environ.get(_ENV_KAFKA_BOOTSTRAP, _DEFAULT_KAFKA_BOOTSTRAP)

# The stability-test lane's Postgres. No hardcoded default DSN (rule 8:
# fail-fast on missing env, never a silent fallback that could point at the
# wrong lane) -- a full DSN is required via env var, exactly the shape
# `tests/helpers/util_postgres.py::PostgresConfig.from_env` already
# establishes for `OMNIBASE_INFRA_DB_URL`, but under a lane-scoped name so
# this test never silently reads a DEV-lane DSN some other test configured.
_ENV_POSTGRES_DSN = "STABILITY_TEST_POSTGRES_DSN"

TOPIC: str = EnumSteelOnslaughtTopic.EVT_MATCH_TERMINAL_V1.value

_CONSUMER_TIMEOUT_SECONDS = 60.0
_PRODUCER_FLUSH_TIMEOUT_SECONDS = 30.0


def _can_reach(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _kafka_bootstrap_reachable() -> bool:
    parts = KAFKA_BOOTSTRAP.rsplit(":", 1)
    if len(parts) != 2:
        return False
    host, port_str = parts
    try:
        port = int(port_str)
    except ValueError:
        return False
    return _can_reach(host, port)


def _postgres_dsn() -> str | None:
    dsn = os.environ.get(_ENV_POSTGRES_DSN)
    return dsn.strip() if dsn and dsn.strip() else None


# Module-wide: categorize as integration. The live-infra reachability skips
# below are scoped to TestSteelDispatchGoldenChainPositiveCase ONLY -- the
# negative-case test reads a checked-in fixture and needs no live infra, so
# it must not be swept up by a module-wide skip (it would otherwise never
# run at all in an environment without stability-lane reachability, e.g. CI).
pytestmark = [pytest.mark.integration, pytest.mark.kafka]

_live_infra_skips = [
    pytest.mark.skipif(
        not _kafka_bootstrap_reachable(),
        reason=(
            f"stability-test Kafka on {KAFKA_BOOTSTRAP} not reachable "
            f"(set {_ENV_KAFKA_BOOTSTRAP} to override)"
        ),
    ),
    pytest.mark.skipif(
        _postgres_dsn() is None,
        reason=(
            f"{_ENV_POSTGRES_DSN} not set -- required full DSN for the "
            "stability-test lane's Postgres (postgresql://user:pass@host:port/db)"
        ),
    ),
]


def _load_json_fixture(name: str) -> dict[str, object] | list[object]:
    result: dict[str, object] | list[object] = json.loads(
        (FIXTURES_DIR / f"{name}.json").read_text()
    )
    return result


class TestSteelDispatchGoldenChainNegativeCase:
    """OMN-15002-precedent negative case: recorded, not re-derived live."""

    def test_negative_case_evidence_recorded_and_self_consistent(self) -> None:
        evidence = _load_json_fixture("steel_dispatch_ledger_negative_evidence")
        assert isinstance(evidence, dict)

        assert evidence["ticket"] == "OMN-15169"
        assert evidence["precedent"] == "OMN-15002"
        assert evidence["topic"] == TOPIC
        assert evidence["lane"] == "stability-test"

        # The load-bearing claim: zero ledger rows while the deployed
        # contract predated the paired allowlist diff.
        assert (
            evidence["steel_topic_present_in_deployed_contract_before_refresh"] is False
        )
        assert evidence["event_ledger_row_count_before_refresh"] == 0

        # The natural-experiment offsets must be non-empty and contiguous
        # from 0 -- otherwise this isn't proof the allowlist (not some other
        # gate) was the cause, since an empty/negative range proves nothing.
        offsets = evidence["offsets_present_on_topic_with_zero_ledger_rows"]
        assert offsets == sorted(offsets)
        assert offsets[0] == 0
        assert len(offsets) == evidence["topic_high_watermark_at_check_time"]


@_live_infra_skips[0]
@_live_infra_skips[1]
class TestSteelDispatchGoldenChainPositiveCase:
    """Live positive case: topic -> dispatch -> event_ledger row."""

    def test_golden_chain_positive_topic_to_ledger_row(self) -> None:
        """Publish one synthetic event; assert a real `event_ledger` row.

        Terminal checkpoint is a direct SQL readback by `correlation_id` --
        never "producer.flush() returned 0 pending" alone. A successful
        Kafka delivery only proves the broker accepted the message; it does
        not prove `node_ledger_projection_compute` dispatched and
        `node_ledger_write_effect` persisted it (exactly the gap OMN-15002
        and now OMN-15215 both document).
        """
        confluent_kafka = pytest.importorskip(
            "confluent_kafka",
            reason="requires confluent-kafka (uv sync --extra live, or add as a dep)",
        )
        asyncpg = pytest.importorskip("asyncpg")

        correlation_id = str(uuid4())
        match_id = f"match.omn15169.golden-chain.{uuid4().hex[:12]}"

        golden = _load_json_fixture("steel_dispatch_ledger_success")
        assert isinstance(golden, list)
        expected_event_types = {entry["event_type"] for entry in golden}
        assert "topic_event_published" in expected_event_types
        assert "ledger_row_persisted" in expected_event_types

        # --- Checkpoint 0: publish to the real stability-test topic ---
        producer = confluent_kafka.Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
        value = {
            "event_type": "match_started",
            "match_id": match_id,
            "envelope": {
                "correlation_id": correlation_id,
                "causation_id": None,
                "entity_id": match_id,
                "message_id": str(uuid4()),
            },
            "payload": {"probe": "OMN-15169 infra golden-chain positive case"},
        }
        headers = [
            ("correlation_id", correlation_id.encode("utf-8")),
            ("event_type", b"match_started"),
            ("source", b"omn15169-golden-chain-test"),
        ]
        delivery_errors: list[str] = []

        def _on_delivery(err: object, _msg: object) -> None:
            if err is not None:
                delivery_errors.append(str(err))

        producer.produce(
            topic=TOPIC,
            key=match_id.encode("utf-8"),
            value=json.dumps(value, default=str).encode("utf-8"),
            headers=headers,
            callback=_on_delivery,
        )
        pending = producer.flush(_PRODUCER_FLUSH_TIMEOUT_SECONDS)
        assert pending == 0, (
            f"{pending} message(s) still undelivered after flush timeout"
        )
        assert not delivery_errors, (
            f"Kafka producer reported delivery failures (topic missing / "
            f"partition-cap class): {delivery_errors}"
        )

        # --- Terminal checkpoint: a real event_ledger row, by correlation_id ---
        import asyncio

        async def _poll() -> dict[str, object] | None:
            conn = await asyncpg.connect(_postgres_dsn())
            try:
                deadline = time.monotonic() + _CONSUMER_TIMEOUT_SECONDS
                while time.monotonic() < deadline:
                    row = await conn.fetchrow(
                        "SELECT ledger_entry_id, topic, partition, kafka_offset, "
                        "correlation_id, event_type, source, ledger_written_at "
                        "FROM event_ledger WHERE correlation_id = $1",
                        correlation_id,
                    )
                    if row is not None:
                        return dict(row)
                    await asyncio.sleep(2.0)
                return None
            finally:
                await conn.close()

        row = asyncio.run(_poll())

        assert row is not None, (
            f"event_ledger has no row for correlation_id={correlation_id} "
            f"(topic={TOPIC}, match_id={match_id}) within "
            f"{_CONSUMER_TIMEOUT_SECONDS}s of a confirmed Kafka delivery. "
            "The topic->dispatch->ledger chain did not complete. Known live "
            "blocker as of 2026-07-26: OMN-15215 (node_ledger_projection_"
            "compute never attaches a live consumer group for this topic, "
            "or any of the 18 other topics added since OMN-15006, on the "
            "stability-test lane, even after a verified contract refresh)."
        )
        assert row["topic"] == TOPIC
        assert str(row["correlation_id"]) == correlation_id

        print(
            "\nOMN-15169 infra golden chain POSITIVE case: PASS\n"
            f"  correlation_id={correlation_id}\n"
            f"  match_id={match_id}\n"
            f"  ledger_entry_id={row['ledger_entry_id']}\n"
            f"  partition={row['partition']} kafka_offset={row['kafka_offset']}\n"
        )
