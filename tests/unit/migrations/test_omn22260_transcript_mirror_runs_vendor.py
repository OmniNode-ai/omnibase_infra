# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22260: pin the transcript_mirror_runs migration bytes, ledger binding, class and shape.

Infra half only, of plan task T0a. The reducer ``node_projection_transcript_mirror_run``
(omnimarket) is the sole writer of ``omninode_internal.transcript_mirror_runs``, one row per
(run id, attempt id) holding the attempt's counts and its terminal state; the orchestrator
gates a new run on ``mirror_interval_s`` against the newest ``started_at`` in the table.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_transcript_mirror_run"
_VENDOR = _FORWARD / "nodes" / _NODE
_CREATE = "0000_create_transcript_mirror_runs.sql"
_GRANT = "0001_grant_omninode_runtime_transcript_mirror_runs.sql"
_SHA256 = {
    _CREATE: "5e6fa1c1087122706543b053e93385e7c3054e8db25bf46b9979f6426de8b357",
    _GRANT: "a2252fd70e10dbe920392dafa8c16ef2e95e94e9a575e84fa980331404339414",
}
_COUNT_COLUMNS = (
    "files_seen",
    "stored",
    "would_store",
    "unchanged",
    "destination_ahead",
    "refused_stale",
    "refused_conflict",
    "source_unreadable",
    "bytes_put",
)
_COLUMNS = (
    "run_id",
    "attempt_id",
    "state",
    "started_at",
    "finished_at",
    *_COUNT_COLUMNS,
)


def _statements(filename: str) -> str:
    text = (_VENDOR / filename).read_text(encoding="utf-8")
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("--")
    )


def test_only_the_create_and_grant_files_are_vendored() -> None:
    assert sorted(path.name for path in _VENDOR.glob("*")) == [_CREATE, _GRANT]


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_vendor_bytes_and_manifest_binding_are_exact(filename: str) -> None:
    artifact = f"nodes/{_NODE}/{filename}"
    digest = hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest()
    assert digest == _SHA256[filename]
    rows = [
        row.split("\t")
        for row in (_FORWARD / "_ledger" / "application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if row.split("\t", 1)[0] == artifact
    ]
    assert rows == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{filename}",
            digest,
        ]
    ]


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_both_migrations_are_declared_expand_only(filename: str) -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{filename}"] == "expand-only"


def test_creates_exactly_one_table_keyed_by_run_and_attempt() -> None:
    sql = _statements(_CREATE)
    assert re.findall(r"CREATE TABLE IF NOT EXISTS (\S+)", sql) == [
        "omninode_internal.transcript_mirror_runs"
    ]
    assert len(re.findall(r"\bCREATE TABLE\b", sql)) == 1
    assert re.search(r"PRIMARY KEY \(run_id, attempt_id\)", sql)


def test_every_row_model_column_is_declared_and_reconciled() -> None:
    sql = _statements(_CREATE)
    block = re.search(
        r"CREATE TABLE IF NOT EXISTS omninode_internal\.transcript_mirror_runs \((.*?)\n\);",
        sql,
        re.S,
    )
    assert block is not None
    declared = re.findall(
        r"^\s{4}(\w+)\s+(?:TEXT|TIMESTAMPTZ|BIGINT)\b", block.group(1), re.M
    )
    assert tuple(declared) == _COLUMNS
    added = re.findall(
        r"ALTER TABLE omninode_internal\.transcript_mirror_runs\s+ADD COLUMN IF NOT EXISTS (\w+)",
        sql,
    )
    assert tuple(added) == _COLUMNS


def test_the_terminal_states_are_the_reducers_and_a_terminal_row_has_a_finish_time() -> (
    None
):
    sql = _statements(_CREATE)
    assert "state IN ('started', 'completed', 'dry-run-completed', 'failed')" in sql
    assert "(state = 'started') = (finished_at IS NULL)" in sql


def test_the_last_attempt_gate_reads_an_index() -> None:
    assert re.search(
        r"CREATE INDEX IF NOT EXISTS idx_transcript_mirror_runs_started_at\s+"
        r"ON omninode_internal\.transcript_mirror_runs \(started_at DESC\)",
        _statements(_CREATE),
    )


def test_create_never_removes_anything_and_grants_nothing() -> None:
    sql = _statements(_CREATE)
    assert not re.search(r"\b(DROP|DELETE|TRUNCATE|GRANT|REVOKE|RENAME)\b", sql, re.I)
    assert set(re.findall(r"ALTER TABLE (\S+)", sql)) == {
        "omninode_internal.transcript_mirror_runs"
    }


def test_the_grant_is_the_runtime_writers_and_withholds_delete() -> None:
    grant = _statements(_GRANT)
    assert "GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;" in grant
    assert re.search(
        r"GRANT SELECT, INSERT, UPDATE\s+ON omninode_internal\.transcript_mirror_runs"
        r"\s+TO omninode_runtime;",
        grant,
    )
    grants = re.findall(r"GRANT\s+([^;]*?)\s+TO\s+(\S+?);", grant, re.I | re.S)
    assert {to for _, to in grants} == {"omninode_runtime"}
    assert not re.search(r"\b(DELETE|TRUNCATE|REFERENCES|TRIGGER|ALL)\b", grant, re.I)
    assert not re.search(r"\bPUBLIC\b|\bWITH GRANT OPTION\b", grant, re.I)


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_migration_passes_the_application_database_sql_gate(filename: str) -> None:
    from omnibase_infra.topology.application_database import (
        TOPOLOGY_PROFILE_INSTANCE_MAP,
        load_topology_profile,
    )
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / filename).read_text(encoding="utf-8")
    for profile in sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())):
        violations = lint_application_database_sql(sql, load_topology_profile(profile))
        assert violations == (), f"{profile}: {violations}"


def test_the_linter_is_live_positive_control() -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / _CREATE).read_text(encoding="utf-8")
    broken = sql.replace(
        "omninode_internal.transcript_mirror_runs", "tenant.transcript_mirror_runs"
    )
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()
