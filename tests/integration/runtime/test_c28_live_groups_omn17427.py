# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise both C28 collectors against recorded and real PostgreSQL replies."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_constants import (
    LIVE_WINDOW_SQL,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
    ConsumerFlowLane,
)
from scripts.ci import c28_consumer_flow_probe as script
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_MIGRATION = (
    _ROOT
    / "docker/migrations/forward/nodes/node_projection_consumer_flow"
    / "0000_create_consumer_flow_windows.sql"
)
_RECORDING = (
    _ROOT / "tests/ci/fixtures/ci_live_contact/c28_live_groups_dev201_omn17427.json"
)
_SCAN_SQL = (
    "select distinct consumer_group from omninode_internal.consumer_flow_windows "
    "where window_start > now() - interval '10 minutes'"
)


def test_recorded_live_groups_reaches_both_collectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SQL sent to psql must match the reply's recorded index-backed query."""
    recording = json.loads(_RECORDING.read_text())["response"]
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert argv[:5] == ["recorded-docker", "exec", script.PG_CONTAINER, "sh", "-c"]
        assert f'-c "{recording["index_backed"]["sql"]}"' in argv[5]
        assert kwargs["timeout"] == 120
        return subprocess.CompletedProcess(
            argv, 0, "\n".join(recording["index_backed"]["stdout_lines"]), ""
        )

    monkeypatch.setattr(script.subprocess, "run", runner)
    node_lane = ConsumerFlowLane(
        docker="recorded-docker", base_url="http://projection.test", runner=runner
    )
    script_lane = script.Lane(docker="recorded-docker")
    expected = sorted(set(recording["whole_table"]["stdout_lines"]))

    assert len(expected) == 533
    assert node_lane.live_groups() == expected
    assert script_lane.live_groups() == expected
    assert len(calls) == 2


@pytest.mark.postgres
def test_live_groups_postgres_results_and_index_plan(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """Execute the shipped SQL on the migration's schema, including cutoff edges."""
    connection = ephemeral_postgres.connect()
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA omninode_internal")
            cursor.execute(_MIGRATION.read_text())
            for query in (LIVE_WINDOW_SQL, script.LIVE_WINDOW_SQL):
                cursor.execute(query)
                assert cursor.fetchall() == []

            # now() is fixed for this transaction, so equality at the ten-minute
            # cutoff cannot drift between INSERT and either collector's read.
            cursor.execute(
                """
                INSERT INTO omninode_internal.consumer_flow_windows
                    (consumer_group, topic, window_start, window_end, node_id,
                     ingest_sequence, upstream_evidence, flow_state, evaluated_at)
                SELECT grp, 'proof-topic', now() - start_age, now() - end_age,
                       '00000000-0000-0000-0000-000000000001'::uuid,
                       1, 'proof', 'FLOWING', now()
                FROM (VALUES
                    ('active-a', interval '1 minute', interval '0 minutes'),
                    ('active-a', interval '2 minutes', interval '1 minute'),
                    ('active-z', interval '9 minutes', interval '8 minutes'),
                    ('cutoff', interval '10 minutes', interval '9 minutes'),
                    ('long-window', interval '11 minutes', interval '1 minute'),
                    ('stale', interval '1 day', interval '23 hours')
                ) AS windows(grp, start_age, end_age)
                """
            )
            cursor.execute(_SCAN_SQL)
            expected = sorted(row[0] for row in cursor.fetchall())
            assert expected == ["active-a", "active-z"]
            for query in (LIVE_WINDOW_SQL, script.LIVE_WINDOW_SQL):
                cursor.execute(query)
                assert sorted(row[0] for row in cursor.fetchall()) == expected

            # Many historical windows per group model the workload that made
            # DISTINCT scan the whole table. Leave the planner settings alone.
            cursor.execute(
                """
                INSERT INTO omninode_internal.consumer_flow_windows
                    (consumer_group, topic, window_start, window_end, node_id,
                     ingest_sequence, upstream_evidence, flow_state, evaluated_at)
                SELECT 'history-' || g, 'proof-topic',
                       now() - interval '1 day' - n * interval '1 minute',
                       now() - interval '1 day' - (n - 1) * interval '1 minute',
                       '00000000-0000-0000-0000-000000000001'::uuid,
                       n, 'proof', 'IDLE', now()
                FROM generate_series(1, 20) g, generate_series(1, 1000) n
                """
            )
            cursor.execute("ANALYZE omninode_internal.consumer_flow_windows")
            for query in (LIVE_WINDOW_SQL, script.LIVE_WINDOW_SQL):
                cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + query)
                plan = json.dumps(cursor.fetchone()[0])
                assert '"Seq Scan"' not in plan, plan
                assert "idx_consumer_flow_windows_group_time" in plan, plan
                cursor.execute(query)
                assert sorted(row[0] for row in cursor.fetchall()) == expected
    finally:
        connection.close()
