# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""B23: real production upsert outcomes through the runtime callback.

The producer's SQL and schema are frozen with provenance, rather than replaced
by a database double or inferred from a writer's name in an old log. This repo
cannot import omnimarket (the dependency runs the other way). The fixture uses
its exact upsert with psycopg2 parameter binding against disposable Postgres.
The owning writer's real-Postgres suite separately proves its result accounting.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from psycopg2.extras import RealDictCursor

from omnibase_infra.runtime.auto_wiring import handler_wiring
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)
from tests.integration.migrations.conftest import (
    EphemeralPostgres,
)

pytestmark = [pytest.mark.integration, pytest.mark.ephemeral_pg]
_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "omn18992"
_TOPIC = "onex.evt.platform.node-heartbeat.v1"
_LOGGER = handler_wiring.__name__
_START = datetime(2026, 9, 21, 10, 18, tzinfo=UTC)


def _upsert_sql() -> str:
    provenance = json.loads((_FIXTURES / "consumer-flow-provenance.json").read_text())
    for name, digest in provenance["assets"].items():
        assert hashlib.sha256((_FIXTURES / name).read_bytes()).hexdigest() == digest
    # Driver binding changes from asyncpg's $n to psycopg2's %s; SQL is unchanged.
    return re.sub(
        r"\$\d+", "%s", (_FIXTURES / "consumer-flow-upsert.sql.captured").read_text()
    )


class _SqlConsumerFlowWriter:
    """A fixture executing the producer's real conflict predicate and RETURNING."""

    def __init__(self, pg: EphemeralPostgres, *, disable_guard: bool = False) -> None:
        self.pg = pg
        self.node_id = str(uuid4())
        self.results: list[dict[str, Any]] = []
        self.sql = _upsert_sql()
        if disable_guard:
            self.sql = re.sub(
                r"WHERE .*?RETURNING", "RETURNING", self.sql, flags=re.DOTALL
            )

    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        window = data.get("flow_window")
        written: list[dict[str, Any]] = []
        if window is not None:
            with (
                self.pg.connect() as conn,
                conn.cursor(cursor_factory=RealDictCursor) as cur,
            ):
                cur.execute(
                    self.sql,
                    (
                        "omn18992-group",
                        _TOPIC,
                        _START,
                        _START + timedelta(minutes=1),
                        self.node_id,
                        window["ingest_sequence"],
                        10,
                        10,
                        0,
                        0,
                        None,
                        "NONE",
                        "FLOWING",
                        _START + timedelta(minutes=1),
                    ),
                )
                written = [dict(row) for row in cur.fetchall()]
        result = {
            "rows_upserted": len(written),
            "flow_rows": written,
            "rows_refused_by_ordering_guard": int(window is not None and not written),
        }
        self.results.append(result)
        return result


@pytest.fixture
def writer(ephemeral_postgres: EphemeralPostgres) -> _SqlConsumerFlowWriter:
    with ephemeral_postgres.connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE SCHEMA omninode_internal")
        cur.execute((_FIXTURES / "consumer-flow-schema.sql.captured").read_text())
    return _SqlConsumerFlowWriter(ephemeral_postgres)


def _callback(writer: _SqlConsumerFlowWriter, monkeypatch: pytest.MonkeyPatch):
    configure_projection_dsns(
        monkeypatch, url="postgresql://user:pass@host:5432/omnidash_analytics"
    )
    # This SQL fixture owns its connection; the runtime-injected adapter is unused.
    monkeypatch.setattr(handler_wiring, "_build_projection_db_adapter", lambda *_: None)
    return handler_wiring._make_projection_dispatch_callback(
        writer,
        projection_database_target("consumer_flow_windows", schema="omninode_internal"),
        (_TOPIC,),
        contract_name="projection_consumer_flow",
    )


def _heartbeat(sequence: int | None) -> dict[str, Any]:
    payload = {} if sequence is None else {"flow_window": {"ingest_sequence": sequence}}
    return {"topic": _TOPIC, "payload": payload}


def _assert_guard_fired(writer: _SqlConsumerFlowWriter) -> None:
    result = writer.results[-1]
    assert result["rows_upserted"] == 0, "proof refused: ordering guard did not fire"
    assert result["rows_refused_by_ordering_guard"] == 1
    with writer.pg.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ingest_sequence FROM omninode_internal.consumer_flow_windows"
        )
        assert cur.fetchone() == (11,), "refused sequence changed the stored row"


def test_real_guard_and_windowless_heartbeat_are_distinct(
    writer: _SqlConsumerFlowWriter,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    callback = _callback(writer, monkeypatch)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        for sequence in (10, 11):
            asyncio.run(callback(_heartbeat(sequence)))
            assert writer.results[-1]["rows_upserted"] == 1
            assert writer.results[-1]["rows_refused_by_ordering_guard"] == 0
        asyncio.run(callback(_heartbeat(10)))
        _assert_guard_fired(writer)
        guard = [
            r for r in caplog.records if "ORDERING_GUARD_REFUSED" in r.getMessage()
        ]
        assert len(guard) == 1 and guard[0].levelno == logging.INFO
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)

        caplog.clear()
        asyncio.run(callback(_heartbeat(None)))
        assert writer.results[-1] == {
            "rows_upserted": 0,
            "flow_rows": [],
            "rows_refused_by_ordering_guard": 0,
        }
        heartbeat = [
            r for r in caplog.records if "WINDOW_LESS_HEARTBEAT" in r.getMessage()
        ]
        assert len(heartbeat) == 1 and heartbeat[0].levelno == logging.INFO
        assert not any(
            "ORDERING_GUARD_REFUSED" in r.getMessage() for r in caplog.records
        )
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)

        caplog.clear()
        # The production predicate accepts equality. It must not be called a refusal.
        asyncio.run(callback(_heartbeat(11)))
        assert writer.results[-1]["rows_upserted"] == 1
        assert writer.results[-1]["rows_refused_by_ordering_guard"] == 0
        assert not any("outcome=" in r.getMessage() for r in caplog.records)


def test_proof_refuses_an_upsert_whose_guard_never_fires(
    writer: _SqlConsumerFlowWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unguarded = _SqlConsumerFlowWriter(writer.pg, disable_guard=True)
    callback = _callback(unguarded, monkeypatch)
    asyncio.run(callback(_heartbeat(11)))
    asyncio.run(callback(_heartbeat(10)))
    with pytest.raises(AssertionError, match="ordering guard did not fire"):
        _assert_guard_fired(unguarded)
