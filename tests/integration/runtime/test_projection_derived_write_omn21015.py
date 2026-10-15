# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real storage controls for the generic transactional derived-write capability.

Failure modes: derive from the incoming sparse dict instead of RETURNING;
commit the upsert before its derived update; leave the upsert committed when
derivation fails; or return an intermediate row. These tests use the shipped
live-events migration/grants and the real topology adapter, on an owned cluster.
The derived summary is a text identity, not a second savings computation.
Money computation and publication are covered by the omnimarket caller tests.
"""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_projection_db_adapter,
)
from tests.helpers.application_db_topology import (
    projection_database_target,
    projection_database_urls,
)
from tests.integration.migrations.conftest import EphemeralPostgres
from tests.integration.runtime.test_live_events_projection_write_path_omn15359 import (
    ANALYTICS_DB,
    _internal_role_dsn,
    live_events_pg,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]
__all__ = ["live_events_pg"]


def _adapter(pg: EphemeralPostgres):
    target = projection_database_target("live_events", schema="omninode_internal")
    return _build_projection_db_adapter(
        projection_database_urls(target, _internal_role_dsn(pg)), target, None, None
    )


def _derive(row: Mapping[str, object]) -> dict[str, object]:
    return {"summary": f"derived:{row['payload']}"}


def _write(adapter, row: dict[str, object], derive=_derive):
    return adapter.upsert_returning(
        "live_events",
        "event_id",
        row,
        derive=derive,
        input_columns=("payload",),
        derived_columns=("summary",),
        returning=("event_id", "payload", "summary"),
    )


def _stored(pg: EphemeralPostgres, event_id: str) -> tuple[str, str] | None:
    with pg.connect(dbname=ANALYTICS_DB) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT payload, summary FROM omninode_internal.live_events "
            "WHERE event_id=%s",
            (event_id,),
        )
        return cur.fetchone()


@pytest.mark.parametrize("held_row", ["sparse", "terminal"])
def test_two_connections_derive_from_the_actual_merged_row(
    live_events_pg: EphemeralPostgres, held_row: str
) -> None:
    first_db, second_db = _adapter(live_events_pg), _adapter(live_events_pg)
    try:
        for _ in range(3):
            event_id = str(uuid4())
            sparse = {"event_id": event_id, "type": "ACTION"}
            terminal = {**sparse, "payload": "terminal"}
            first_row, second_row = (
                (sparse, terminal) if held_row == "sparse" else (terminal, sparse)
            )
            entered, release = Event(), Event()

            def held_write(_entered=entered, _release=release, _row=first_row):
                _entered.set()
                assert _release.wait(15), "owned schedule was not released"
                return _write(first_db, _row)

            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(held_write)
                assert entered.wait(15)
                try:
                    _write(second_db, second_row)
                finally:
                    release.set()
                result = first.result(timeout=15)
            assert _stored(live_events_pg, event_id) == ("terminal", "derived:terminal")
            assert result == [
                {
                    "event_id": event_id,
                    "payload": "terminal",
                    "summary": "derived:terminal",
                }
            ]
    finally:
        first_db.close()
        second_db.close()


def test_no_intermediate_upsert_is_committed_while_derivation_is_running(
    live_events_pg: EphemeralPostgres,
) -> None:
    adapter = _adapter(live_events_pg)
    event_id = str(uuid4())
    entered, release = Event(), Event()

    def held_derive(row: Mapping[str, object]) -> dict[str, object]:
        entered.set()
        assert release.wait(15), "owned derivation was not released"
        return _derive(row)

    try:
        # Positive control: the observer sees a real committed row on this table.
        adapter.upsert_returning(
            "live_events",
            "event_id",
            {"event_id": event_id, "payload": "initial", "summary": "derived:initial"},
            returning=("event_id",),
        )
        assert _stored(live_events_pg, event_id) == ("initial", "derived:initial")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                _write,
                adapter,
                {"event_id": event_id, "payload": "terminal"},
                held_derive,
            )
            try:
                assert entered.wait(15), "derive was not called after the upsert"
                assert _stored(live_events_pg, event_id) == (
                    "initial",
                    "derived:initial",
                )
            finally:
                release.set()
            final = pending.result(timeout=15)
        assert final[0]["summary"] == "derived:terminal"
        assert _stored(live_events_pg, event_id) == ("terminal", "derived:terminal")
    finally:
        release.set()
        adapter.close()


def test_a_derivation_error_rolls_back_the_input_write(
    live_events_pg: EphemeralPostgres,
) -> None:
    adapter = _adapter(live_events_pg)
    event_id = str(uuid4())

    def broken(_: Mapping[str, object]) -> dict[str, object]:
        raise RuntimeError("planted derivation failure")

    try:
        adapter.upsert_returning(
            "live_events",
            "event_id",
            {"event_id": event_id, "payload": "initial", "summary": "derived:initial"},
            returning=("event_id",),
        )
        with pytest.raises(RuntimeError, match="planted derivation failure"):
            _write(adapter, {"event_id": event_id, "payload": "terminal"}, broken)
        assert _stored(live_events_pg, event_id) == ("initial", "derived:initial")
        # The same adapter must remain usable after the failed transaction.
        assert (
            _write(adapter, {"event_id": event_id, "payload": "retry"})[0]["summary"]
            == "derived:retry"
        )
    finally:
        adapter.close()


def test_unscoped_derived_write_commits_without_a_read_only_seam(
    live_events_pg: EphemeralPostgres,
) -> None:
    adapter = _adapter(live_events_pg)
    event_id = str(uuid4())

    try:
        result = _write(adapter, {"event_id": event_id, "payload": "committed"})
        assert result[0]["summary"] == "derived:committed"
        assert _stored(live_events_pg, event_id) == ("committed", "derived:committed")
        assert all(
            conn.autocommit
            for conn in adapter._binding_connections.connections.values()
        )
    finally:
        adapter.close()
