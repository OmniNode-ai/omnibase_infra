# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""C28's live-groups read stays on an index, never a scan of the whole table.

``omninode_internal.consumer_flow_windows`` keeps every window since the lane
first projected (48,263,601 rows, 32 GB on dev-201 at 2026-10-16T21:50Z) and has
no index leading with ``window_start``. ``select distinct consumer_group ...
where window_start > ...`` planned a parallel seq scan of the whole table: about
15 s with a warm cache, and past the probe's 120 s ``docker exec`` timeout under
host load, so C28 run 38968109234 graded INDETERMINATE (``docker ['exec',
'omnibase-infra-postgres'] could not run: TimeoutExpired``) and the board read red.

The read now walks the distinct groups through ``idx_consumer_flow_windows_group_time``
(consumer_group, window_end DESC) and checks each group's recent windows on the
same index: 55 ms on the same table, same 533 groups.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers import (
    _consumer_flow_constants as node_constants,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
    ConsumerFlowLane,
)
from scripts.ci import c28_consumer_flow_probe as script

pytestmark = pytest.mark.unit


def _normalised(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().lower()


def test_live_window_sql_never_scans_the_whole_table() -> None:
    sql = _normalised(node_constants.LIVE_WINDOW_SQL)
    # The scan form: a top-level distinct filtered only on the unindexed column.
    assert not sql.startswith("select distinct"), sql
    # Distinct groups come from a loose index scan over consumer_group ...
    assert sql.startswith("with recursive"), sql
    assert "order by c.consumer_group limit 1" in sql
    assert "c.consumer_group > g.consumer_group" in sql
    # ... and each group's recency is bounded on the indexed (group, window_end) pair.
    assert "w.consumer_group = g.consumer_group" in sql
    assert "w.window_end > now() - interval '10 minutes'" in sql
    # The criterion's own predicate is unchanged: a window that started in the window.
    assert "w.window_start > now() - interval '10 minutes'" in sql


def test_live_window_sql_survives_the_double_quoted_shell_argument() -> None:
    # live_groups embeds the SQL in sh -c "psql ... -c \"<sql>\"".
    assert '"' not in node_constants.LIVE_WINDOW_SQL
    assert "$" not in node_constants.LIVE_WINDOW_SQL
    assert "`" not in node_constants.LIVE_WINDOW_SQL


def test_script_and_node_read_live_groups_with_the_same_sql() -> None:
    assert script.LIVE_WINDOW_SQL == node_constants.LIVE_WINDOW_SQL


def test_live_groups_runs_the_index_backed_sql_in_the_postgres_container() -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "group-b\ngroup-a\n\ngroup-b\n", "")

    lane = ConsumerFlowLane(
        docker="fake-docker", base_url="http://projection.test", runner=runner
    )

    assert lane.live_groups() == ["group-a", "group-b"]
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:5] == ["fake-docker", "exec", "omnibase-infra-postgres", "sh", "-c"]
    assert f'-c "{node_constants.LIVE_WINDOW_SQL}"' in argv[5]
    assert "-d omnidash_analytics" in argv[5]
