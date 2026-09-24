# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19344 (plan row R7, rule RB): migration class declaration + RB-2 barrier.

Every forward migration declares a class -- ``expand-only``, ``forward-only``
or ``contract`` -- in ``config/migration_classes.yaml``. A DDL checker refuses a
destructive statement declared ``expand-only`` and refuses an undeclared file.
A down-migration lifts rule RB-2's rollback barrier only when a recorded lab
execution, bound to the migration id AND to the exact bytes of both scripts,
exists in ``config/migration_down_executions.yaml``.

The known-bad inputs are REAL checked-in bytes where the ticket names them:
``087_drop_stale_delegation_events_decoy.sql`` (a ``DROP TABLE ... CASCADE``)
and ``node_canary_score_reducer/0003`` (a column type change inside a DO block).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "validation" / "check_migration_class.py"
_MIGRATIONS = _REPO_ROOT / "docker" / "migrations"

_KEY_087 = "forward/087_drop_stale_delegation_events_decoy.sql"
_KEY_CANARY_0003 = "forward/nodes/node_canary_score_reducer/0003_capability_scores_tenant_id_to_uuid.sql"
# A real, purely additive migration: one CREATE TABLE plus its indexes.
_KEY_ADDITIVE = "forward/020_create_agent_actions_table.sql"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("check_migration_class", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> object:
    return _load()


def _text(key: str) -> str:
    return (_MIGRATIONS / key).read_text(encoding="utf-8")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# AC1 -- 087 declared expand-only is refused; declared forward-only passes.
# --------------------------------------------------------------------------


def test_087_declared_expand_only_is_refused(gate: object) -> None:
    violations = gate.violations_for(_KEY_087, _text(_KEY_087), "expand-only")  # type: ignore[attr-defined]
    assert violations, "a DROP TABLE ... CASCADE declared expand-only must be refused"
    assert any("DROP" in v for v in violations), violations
    assert all(_KEY_087 in v for v in violations), violations


def test_087_declared_forward_only_passes(gate: object) -> None:
    assert gate.violations_for(_KEY_087, _text(_KEY_087), "forward-only") == []  # type: ignore[attr-defined]


def test_087_declared_contract_passes(gate: object) -> None:
    assert gate.violations_for(_KEY_087, _text(_KEY_087), "contract") == []  # type: ignore[attr-defined]


def test_canary_0003_type_change_inside_do_block_declared_expand_only_is_refused(
    gate: object,
) -> None:
    violations = gate.violations_for(  # type: ignore[attr-defined]
        _KEY_CANARY_0003, _text(_KEY_CANARY_0003), "expand-only"
    )
    assert violations
    assert any("ALTER COLUMN" in v for v in violations), violations


def test_real_additive_migration_declared_expand_only_passes(gate: object) -> None:
    assert (
        gate.violations_for(_KEY_ADDITIVE, _text(_KEY_ADDITIVE), "expand-only")  # type: ignore[attr-defined]
        == []
    )


def test_unknown_class_value_is_refused(gate: object) -> None:
    violations = gate.violations_for(_KEY_ADDITIVE, _text(_KEY_ADDITIVE), "expand")  # type: ignore[attr-defined]
    assert violations and "not a declared class" in violations[0]


# --------------------------------------------------------------------------
# AC2 -- a new migration with no declared class is refused.
# --------------------------------------------------------------------------


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "migrations"
    for key, text in files.items():
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_new_migration_with_no_declared_class_is_refused(
    gate: object, tmp_path: Path
) -> None:
    root = _tree(
        tmp_path,
        {
            "forward/001_a.sql": "CREATE TABLE a (id int);\n",
            "forward/002_new.sql": "CREATE TABLE b (id int);\n",
        },
    )
    violations = gate.check_tree(root, {"forward/001_a.sql": "expand-only"}, [])  # type: ignore[attr-defined]
    assert len(violations) == 1, violations
    assert "forward/002_new.sql" in violations[0]
    assert "no declared class" in violations[0]


def test_node_stream_and_intelligence_files_are_enumerated(
    gate: object, tmp_path: Path
) -> None:
    root = _tree(
        tmp_path,
        {
            "forward/001_a.sql": "CREATE TABLE a (id int);\n",
            "forward/nodes/node_x/0001_x.sql": "CREATE TABLE x (id int);\n",
            "intelligence/001_i.sql": "CREATE TABLE i (id int);\n",
            "forward/_ledger/bootstrap.sql": "DROP TABLE whatever;\n",
            "rollback/rollback_001_a.sql": "DROP TABLE a;\n",
        },
    )
    assert gate.enumerate_forward_migrations(root) == (  # type: ignore[attr-defined]
        "forward/001_a.sql",
        "forward/nodes/node_x/0001_x.sql",
        "intelligence/001_i.sql",
    )
    violations = gate.check_tree(root, {}, [])  # type: ignore[attr-defined]
    assert len(violations) == 3, violations


def test_manifest_entry_for_a_missing_file_is_refused(
    gate: object, tmp_path: Path
) -> None:
    root = _tree(tmp_path, {"forward/001_a.sql": "CREATE TABLE a (id int);\n"})
    violations = gate.check_tree(  # type: ignore[attr-defined]
        root,
        {"forward/001_a.sql": "expand-only", "forward/009_gone.sql": "expand-only"},
        [],
    )
    assert len(violations) == 1 and "forward/009_gone.sql" in violations[0]


def test_committed_tree_passes_positive_control(gate: object) -> None:
    manifest = gate.load_manifest(gate.DEFAULT_MANIFEST)  # type: ignore[attr-defined]
    executions = gate.load_executions(gate.DEFAULT_EXECUTIONS)  # type: ignore[attr-defined]
    assert gate.check_tree(_MIGRATIONS, manifest, executions) == []  # type: ignore[attr-defined]
    # The manifest covers every forward migration the runner applies.
    assert set(manifest) == set(gate.enumerate_forward_migrations(_MIGRATIONS))  # type: ignore[attr-defined]
    # And it declares the two ticket-named destructive files as barriers.
    assert manifest[_KEY_087] in ("forward-only", "contract")
    assert manifest[_KEY_CANARY_0003] in ("forward-only", "contract")


def test_committed_tree_with_087_flipped_to_expand_only_is_refused(
    gate: object,
) -> None:
    manifest = dict(gate.load_manifest(gate.DEFAULT_MANIFEST))  # type: ignore[attr-defined]
    manifest[_KEY_087] = "expand-only"
    violations = gate.check_tree(_MIGRATIONS, manifest, [])  # type: ignore[attr-defined]
    assert len(violations) == 1 and _KEY_087 in violations[0], violations


# --------------------------------------------------------------------------
# The DDL analyzer: destructive shapes found, additive shapes and noise not.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "needle"),
    [
        ("ALTER TABLE t ADD COLUMN c int NOT NULL;", "NOT NULL"),
        ("ALTER TABLE t RENAME COLUMN a TO b;", "RENAME"),
        ("ALTER TABLE t ALTER COLUMN a TYPE bigint;", "ALTER COLUMN"),
        ("ALTER TABLE t ADD CONSTRAINT ck CHECK (a > 0);", "ADD CONSTRAINT"),
        ("ALTER TABLE t ENABLE ROW LEVEL SECURITY;", "ROW LEVEL SECURITY"),
        ("UPDATE t SET a = 1;", "UPDATE"),
        ("DELETE FROM t WHERE a = 1;", "DELETE"),
        ("TRUNCATE t;", "TRUNCATE"),
        ("REVOKE SELECT ON t FROM r;", "REVOKE"),
        ("CREATE OR REPLACE VIEW v AS SELECT 1;", "CREATE OR REPLACE"),
        ("CREATE UNIQUE INDEX u ON t (a);", "UNIQUE INDEX"),
        ("DO $$ BEGIN EXECUTE format('drop table %I', 'x'); END $$;", "EXECUTE"),
        ("DO $$ BEGIN IF true THEN DROP TABLE x; END IF; END $$;", "DROP"),
        ("ALTER TYPE e RENAME VALUE 'a' TO 'b';", "RENAME"),
        ("ALTER VIEW v SET (security_invoker = true);", "ALTER VIEW"),
    ],
)
def test_destructive_shapes_are_found(gate: object, sql: str, needle: str) -> None:
    findings = gate.destructive_findings(sql)  # type: ignore[attr-defined]
    assert findings, sql
    assert any(needle in f for f in findings), (sql, findings)


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE IF NOT EXISTS t (id uuid PRIMARY KEY, a text NOT NULL);",
        "CREATE INDEX IF NOT EXISTS i ON t (a);",
        "ALTER TABLE t ADD COLUMN IF NOT EXISTS c text;",
        "ALTER TABLE t ADD COLUMN c int NOT NULL DEFAULT 0, ADD COLUMN d numeric(10,2);",
        "CREATE VIEW v AS SELECT 1;",
        "GRANT SELECT, INSERT, UPDATE ON t TO r;",
        "GRANT EXECUTE ON FUNCTION f() TO r;",
        "COMMENT ON TABLE t IS 'we will drop and rename nothing here';",
        "-- DROP TABLE t;\nCREATE TABLE u (id int);",
        "/* DELETE FROM t */ CREATE TABLE u (id int);",
        "INSERT INTO t (a) VALUES ('x') ON CONFLICT DO NOTHING;",
        "ALTER TYPE e ADD VALUE IF NOT EXISTS 'z';",
        "CREATE TABLE p (id int REFERENCES q (id) ON UPDATE CASCADE ON DELETE SET NULL);",
        # A new table may be tightened freely in the file that creates it:
        # no older composition has ever read or written it.
        "CREATE TABLE n (id int, a int);\n"
        "ALTER TABLE n ENABLE ROW LEVEL SECURITY;\n"
        "ALTER TABLE n ADD CONSTRAINT ck CHECK (a > 0);\n"
        "CREATE UNIQUE INDEX nu ON n (a);\n"
        "CREATE TRIGGER nt BEFORE UPDATE ON n FOR EACH ROW EXECUTE FUNCTION f();",
        "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_attribute WHERE attname = 'c') "
        "THEN ALTER TABLE t ADD COLUMN c text; END IF; END $$;",
    ],
)
def test_additive_shapes_are_not_findings(gate: object, sql: str) -> None:
    assert gate.destructive_findings(sql) == [], sql  # type: ignore[attr-defined]


def test_trigger_on_existing_table_is_a_finding(gate: object) -> None:
    findings = gate.destructive_findings(  # type: ignore[attr-defined]
        "CREATE TRIGGER t1 BEFORE INSERT ON old_table FOR EACH ROW EXECUTE FUNCTION f();"
    )
    assert any("TRIGGER" in f for f in findings), findings


# --------------------------------------------------------------------------
# AC3 -- RB-2: a down-migration lifts the barrier only with a recorded lab
# execution bound to the migration id and both scripts' bytes.
# --------------------------------------------------------------------------

_FWD = "forward/005_drop_thing.sql"
_DOWN = "rollback/rollback_005_drop_thing.sql"
_FWD_SQL = "DROP TABLE thing;\n"
_DOWN_SQL = "CREATE TABLE thing (id int);\n"


def _record(gate: object, **overrides: object) -> object:
    fields: dict[str, object] = {
        "migration": _FWD,
        "down_script": _DOWN,
        "forward_sha256": _sha(_FWD_SQL),
        "down_sha256": _sha(_DOWN_SQL),
        "surface": "mac-scratch:wr-recovery",
        "database": "omnibase_infra",
        "executed_at": "2026-09-24T02:00:00Z",
        "outcome": "PASS",
        "evidence": "forward exit 0; down exit 0; forward re-apply exit 0",
    }
    fields.update(overrides)
    return gate.ModelDownExecution(**fields)  # type: ignore[attr-defined]


@pytest.fixture
def rb_root(tmp_path: Path) -> Path:
    return _tree(tmp_path, {_FWD: _FWD_SQL, _DOWN: _DOWN_SQL})


def test_rb2_forward_only_with_no_execution_record_keeps_the_barrier(
    gate: object, rb_root: Path
) -> None:
    # The down-script EXISTS on disk; a file alone does not lift the barrier.
    assert (rb_root / _DOWN).is_file()
    barrier = gate.rb2_barrier(_FWD, "forward-only", [], rb_root)  # type: ignore[attr-defined]
    assert barrier is not None
    assert _FWD in barrier and "no recorded lab execution" in barrier


def test_rb2_forward_only_with_recorded_lab_execution_lifts_the_barrier(
    gate: object, rb_root: Path
) -> None:
    assert gate.rb2_barrier(_FWD, "forward-only", [_record(gate)], rb_root) is None  # type: ignore[attr-defined]


def test_rb2_record_for_edited_down_script_does_not_lift(
    gate: object, rb_root: Path
) -> None:
    (rb_root / _DOWN).write_text(_DOWN_SQL + "-- edited after the run\n", "utf-8")
    barrier = gate.rb2_barrier(_FWD, "forward-only", [_record(gate)], rb_root)  # type: ignore[attr-defined]
    assert barrier is not None and "down_sha256" in barrier


def test_rb2_failed_or_non_lab_record_does_not_lift(
    gate: object, rb_root: Path
) -> None:
    failed = _record(gate, outcome="FAIL")
    assert gate.rb2_barrier(_FWD, "forward-only", [failed], rb_root) is not None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="surface"):
        _record(gate, surface="stability-test")


def test_rb2_undeclared_class_is_a_barrier(gate: object, rb_root: Path) -> None:
    barrier = gate.rb2_barrier(_FWD, None, [], rb_root)  # type: ignore[attr-defined]
    assert barrier is not None and "no declared class" in barrier


def test_rb2_expand_only_is_never_a_barrier(gate: object, rb_root: Path) -> None:
    assert gate.rb2_barrier(_FWD, "expand-only", [], rb_root) is None  # type: ignore[attr-defined]


def test_stale_execution_record_fails_the_tree_check(
    gate: object, rb_root: Path
) -> None:
    edited = _record(gate, down_sha256="0" * 64)
    violations = gate.check_tree(rb_root, {_FWD: "forward-only"}, [edited])  # type: ignore[attr-defined]
    assert len(violations) == 1 and "down_sha256" in violations[0], violations


def test_committed_execution_records_each_lift_their_barrier(gate: object) -> None:
    manifest = gate.load_manifest(gate.DEFAULT_MANIFEST)  # type: ignore[attr-defined]
    executions = gate.load_executions(gate.DEFAULT_EXECUTIONS)  # type: ignore[attr-defined]
    for record in executions:
        assert (
            gate.rb2_barrier(  # type: ignore[attr-defined]
                record.migration, manifest[record.migration], executions, _MIGRATIONS
            )
            is None
        ), record.migration


# --------------------------------------------------------------------------
# The recorder's verdict: PASS only for an exact round trip.
# --------------------------------------------------------------------------

_RECORDER = _REPO_ROOT / "scripts" / "migrations" / "record_down_migration_execution.py"


@pytest.fixture(scope="module")
def recorder() -> object:
    spec = importlib.util.spec_from_file_location(
        "record_down_migration_execution", _RECORDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trip(recorder: object, **overrides: object) -> object:
    before = frozenset({"rel:public.db_metadata:r"})
    after = before | {"rel:public.thing:r", "col:public.thing.id:integer"}
    fields: dict[str, object] = {
        "forward_exit": 0,
        "down_exit": 0,
        "reforward_exit": 0,
        "before": before,
        "after_forward": after,
        "after_down": before,
        "after_reforward": after,
    }
    fields.update(overrides)
    return recorder.ModelRoundTrip(**fields)  # type: ignore[attr-defined]


def test_recorder_exact_round_trip_is_pass(recorder: object) -> None:
    outcome, evidence = recorder.verdict(_trip(recorder))  # type: ignore[attr-defined]
    assert outcome == "PASS", evidence


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"down_exit": 3}, "down_exit=3"),
        ({"reforward_exit": 1}, "reforward_exit=1"),
        (
            {
                "after_down": frozenset(
                    {"rel:public.db_metadata:r", "rel:public.thing:r"}
                )
            },
            "did not restore",
        ),
        ({"after_down": frozenset()}, "did not restore"),
        ({"after_forward": frozenset({"rel:public.db_metadata:r"})}, "vacuous"),
        ({"after_reforward": frozenset({"rel:public.db_metadata:r"})}, "re-applied"),
    ],
)
def test_recorder_any_deviation_is_fail(
    recorder: object, overrides: dict[str, object], needle: str
) -> None:
    outcome, evidence = recorder.verdict(_trip(recorder, **overrides))  # type: ignore[attr-defined]
    assert outcome == "FAIL" and needle in evidence, evidence


def test_real_tree_rb2_031_lifted_by_its_record_and_087_stays_a_barrier(
    gate: object,
) -> None:
    """The executed pair on the committed tree, not a synthetic one."""
    manifest = gate.load_manifest(gate.DEFAULT_MANIFEST)  # type: ignore[attr-defined]
    executions = gate.load_executions(gate.DEFAULT_EXECUTIONS)  # type: ignore[attr-defined]
    key_031 = "forward/031_create_llm_call_metrics_and_cost_aggregates.sql"
    assert manifest[key_031] == "forward-only"
    assert gate.rb2_barrier(key_031, manifest[key_031], executions, _MIGRATIONS) is None  # type: ignore[attr-defined]
    # Without its record the same migration is a barrier (the record is load-bearing).
    assert gate.rb2_barrier(key_031, manifest[key_031], [], _MIGRATIONS) is not None  # type: ignore[attr-defined]
    barrier = gate.rb2_barrier(_KEY_087, manifest[_KEY_087], executions, _MIGRATIONS)  # type: ignore[attr-defined]
    assert barrier is not None and _KEY_087 in barrier


def test_duplicated_manifest_key_is_refused(gate: object, tmp_path: Path) -> None:
    path = tmp_path / "classes.yaml"
    path.write_text(
        "schema_version: 1\nmigrations:\n"
        f"  {_KEY_087}: forward-only\n"
        f"  {_KEY_087}: expand-only\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicated migration key"):
        gate.load_manifest(path)  # type: ignore[attr-defined]
