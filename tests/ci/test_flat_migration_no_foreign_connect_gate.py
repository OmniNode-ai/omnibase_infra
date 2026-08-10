# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15819 — static gate: no NEW flat migration with an un-ledgered foreign `\\connect`.

Companion to the `omninode_infra` runner-honesty fix
(`k8s/migrations/omnibase-infra-migrate.yaml`): that Job's flat loop has no
execution path for a migration whose `\\connect` names a database other
than its own (``omnibase_infra``) -- 098/099 sat with a false "applied"
ledger row while their real target, ``omnidash_analytics``, never saw
either file. This gate is the pre-merge half, enforced in THIS repo (where
the flat SQL corpus actually lives): a cross-DB flat file must be listed,
with a citation, in
``docker/migrations/forward/cross-database-flat-migrations.yaml`` --
otherwise the gate fails closed.

``test_gate_rejects_a_synthetic_new_cross_db_flat_file`` is the RED-first
proof this deliverable calls for: it builds an isolated fixture tree with a
migration the manifest does NOT know about and asserts
``check_flat_migration_foreign_connect.check()`` returns a violation naming
it. Every other synthetic-fixture test in this file follows the same
isolated-tmp-tree shape so it never depends on, or mutates, the live
``docker/migrations/forward/`` corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.ci import check_flat_migration_foreign_connect as gate

pytestmark = pytest.mark.unit


def _write_manifest(path: Path, entries: list[dict[str, str]]) -> None:
    path.write_text(yaml.safe_dump({"entries": entries}), encoding="utf-8")


# ---------------------------------------------------------------------------
# The live repo: this gate must be clean against real HEAD (AC2, both
# directions) -- if this fails, either a new cross-DB flat file landed
# un-ledgered, or the manifest drifted from live reality.
# ---------------------------------------------------------------------------


def test_gate_passes_against_the_live_repo() -> None:
    # frozen_seed=MANIFEST_FROZEN_SEED matches what main() (the real CI
    # entrypoint) actually passes -- a bare gate.check() would not exercise
    # the closed-ledger enforcement against the live manifest at all.
    violations = gate.check(frozen_seed=gate.MANIFEST_FROZEN_SEED)
    assert not violations, "\n".join(v.describe() for v in violations)


def test_manifest_frozen_seed_matches_the_live_manifest_exactly() -> None:
    """MANIFEST_FROZEN_SEED is a hand-maintained pin, not derived from the
    manifest at runtime (by design -- it must not move just because someone
    edits the manifest). Catch drift between the two explicitly instead of
    letting it show up only as a confusing rejection/acceptance elsewhere."""
    manifest = gate.load_manifest()
    assert set(manifest) == gate.MANIFEST_FROZEN_SEED


def test_live_manifest_has_the_two_omn15819_undeliverable_entries() -> None:
    manifest = gate.load_manifest()
    undeliverable = {
        name for name, entry in manifest.items() if entry.disposition == "undeliverable"
    }
    assert undeliverable == {
        "098_create_omninode_internal_schema.sql",
        "099_create_omninode_internal_live_events.sql",
    }
    for name in undeliverable:
        assert "OMN-15819" in manifest[name].citation, name


def test_live_manifest_entries_all_target_omnidash_analytics() -> None:
    # Not a structural requirement of the gate (a future entry could target a
    # different foreign DB) -- but every entry today does, and a silent
    # change here is worth a loud diff rather than passing quietly.
    manifest = gate.load_manifest()
    assert manifest, "expected at least one grandfathered/undeliverable entry"
    for name, entry in manifest.items():
        assert entry.connect_target == "omnidash_analytics", name


# ---------------------------------------------------------------------------
# Pure detector: mirrors the k8s Job's own `awk '$1 == "\\connect"'` field
# semantics (first line whose first token IS `\connect`, not a `--` comment
# that merely mentions it).
# ---------------------------------------------------------------------------


def test_connect_target_ignores_a_prose_comment_mentioning_connect(
    tmp_path: Path,
) -> None:
    sql = tmp_path / "001_example.sql"
    sql.write_text(
        "-- this file switches with `\\connect other_db` partway through\n"
        "CREATE TABLE t (id int);\n"
    )
    assert gate.flat_migration_connect_target(sql) is None


def test_connect_target_finds_a_real_directive_after_comments(tmp_path: Path) -> None:
    sql = tmp_path / "001_example.sql"
    sql.write_text(
        "-- some header prose\n"
        "-- more prose\n"
        "\\connect omnidash_analytics\n"
        "CREATE TABLE t (id int);\n"
    )
    assert gate.flat_migration_connect_target(sql) == "omnidash_analytics"


def test_connect_target_takes_the_first_directive_only(tmp_path: Path) -> None:
    sql = tmp_path / "001_example.sql"
    sql.write_text("\\connect first_db\nSELECT 1;\n\\connect second_db\n")
    assert gate.flat_migration_connect_target(sql) == "first_db"


def test_connect_target_matches_a_directive_with_leading_whitespace(
    tmp_path: Path,
) -> None:
    """The k8s Job's own `awk '$1 == "\\connect"'` predicate strips leading
    whitespace via default field splitting -- an indented `\\connect` is a
    real, live directive to the runner. This gate must see it too, or a
    migration with `  \\connect other_db` reads as "no \\connect" here while
    remaining foreign to the runner (OMN-15819 CodeRabbit thread
    r3749990754)."""
    sql = tmp_path / "001_example.sql"
    sql.write_text("  \\connect omnidash_analytics\nCREATE TABLE t (id int);\n")
    assert gate.flat_migration_connect_target(sql) == "omnidash_analytics"


def test_connect_target_matches_a_directive_with_a_leading_tab(tmp_path: Path) -> None:
    sql = tmp_path / "001_example.sql"
    sql.write_text("\t\\connect omnidash_analytics\nCREATE TABLE t (id int);\n")
    assert gate.flat_migration_connect_target(sql) == "omnidash_analytics"


def test_flat_migration_files_excludes_the_nodes_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "001_flat.sql").write_text("SELECT 1;\n")
    node_dir = tmp_path / "nodes" / "some_node"
    node_dir.mkdir(parents=True)
    (node_dir / "0000_create.sql").write_text("\\connect omnidash_analytics\n")

    files = gate.flat_migration_files(tmp_path)
    assert [p.name for p in files] == ["001_flat.sql"]


# ---------------------------------------------------------------------------
# RED-first: the gate must FAIL (return a violation) against a synthetic new
# cross-DB flat file the manifest has never heard of.
# ---------------------------------------------------------------------------


def test_gate_rejects_a_synthetic_new_cross_db_flat_file(tmp_path: Path) -> None:
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    (forward_dir / "001_ordinary.sql").write_text("CREATE TABLE ordinary (id int);\n")
    # The defect class: a brand-new flat migration nobody has ever ledgered,
    # targeting a database the runner does not own.
    (forward_dir / "200_new_cross_db_migration.sql").write_text(
        "\\connect some_other_database\nCREATE TABLE t (id int);\n"
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, entries=[])

    violations = gate.check(forward_dir=forward_dir, manifest_path=manifest_path)

    assert len(violations) == 1
    assert violations[0].file == "200_new_cross_db_migration.sql"
    assert "OMN-15819" in violations[0].reason
    assert "no execution path" in violations[0].reason


def test_gate_without_frozen_seed_passes_a_new_file_with_a_matching_entry(
    tmp_path: Path,
) -> None:
    """`frozen_seed=None` (the default) does not enforce the closed-ledger
    property -- unrelated tests in this file rely on that. This is the
    permissive baseline the next two tests contrast against."""
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    (forward_dir / "200_new_cross_db_migration.sql").write_text(
        "\\connect some_other_database\nCREATE TABLE t (id int);\n"
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_path,
        entries=[
            {
                "file": "200_new_cross_db_migration.sql",
                "connect_target": "some_other_database",
                "disposition": "undeliverable",
                "citation": "OMN-99999 -- test fixture",
            }
        ],
    )

    assert gate.check(forward_dir=forward_dir, manifest_path=manifest_path) == []


# ---------------------------------------------------------------------------
# The closed-ledger property (OMN-15819 CodeRabbit thread r3749990788): a
# manifest entry, on its own, must never be able to authorize a brand-new
# cross-DB flat migration -- only a filename already present at
# gate-authorship time (``MANIFEST_FROZEN_SEED``) may pass, regardless of
# disposition or citation. This is what ``main()`` actually enforces.
# ---------------------------------------------------------------------------


def test_gate_with_frozen_seed_rejects_a_new_file_even_with_a_matching_entry(
    tmp_path: Path,
) -> None:
    """RED-first: the exact scenario the permissive test above shows passing
    must now fail once ``frozen_seed`` is supplied -- a new file + a new,
    perfectly well-formed manifest entry in the same PR is still a reject."""
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    (forward_dir / "200_new_cross_db_migration.sql").write_text(
        "\\connect some_other_database\nCREATE TABLE t (id int);\n"
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_path,
        entries=[
            {
                "file": "200_new_cross_db_migration.sql",
                "connect_target": "some_other_database",
                "disposition": "undeliverable",
                "citation": "OMN-99999 -- test fixture",
            }
        ],
    )

    violations = gate.check(
        forward_dir=forward_dir,
        manifest_path=manifest_path,
        frozen_seed=gate.MANIFEST_FROZEN_SEED,
    )

    assert len(violations) == 1
    assert violations[0].file == "200_new_cross_db_migration.sql"
    assert "not part of the frozen OMN-15819 seed set" in violations[0].reason
    assert "hard reject" in violations[0].reason


def test_gate_with_frozen_seed_passes_a_file_that_is_in_the_seed(
    tmp_path: Path,
) -> None:
    """Positive control: a filename genuinely in ``MANIFEST_FROZEN_SEED``
    (one of the two OMN-15819 undeliverable entries) with a matching,
    correct manifest entry still passes -- the closed-ledger check only
    rejects filenames OUTSIDE the frozen set."""
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    (forward_dir / "098_create_omninode_internal_schema.sql").write_text(
        "\\connect omnidash_analytics\nCREATE SCHEMA IF NOT EXISTS x;\n"
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_path,
        entries=[
            {
                "file": "098_create_omninode_internal_schema.sql",
                "connect_target": "omnidash_analytics",
                "disposition": "undeliverable",
                "citation": "OMN-15819 -- test fixture",
            }
        ],
    )

    violations = gate.check(
        forward_dir=forward_dir,
        manifest_path=manifest_path,
        frozen_seed=gate.MANIFEST_FROZEN_SEED,
    )

    assert violations == []


def test_gate_ignores_a_same_database_flat_file(tmp_path: Path) -> None:
    """A \\connect back to the runner's own DB is not foreign -- not a violation."""
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    (forward_dir / "001_same_db.sql").write_text(
        "\\connect omnibase_infra\nCREATE TABLE t (id int);\n"
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, entries=[])

    assert gate.check(forward_dir=forward_dir, manifest_path=manifest_path) == []


def test_gate_ignores_a_flat_file_with_no_connect_directive(tmp_path: Path) -> None:
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    (forward_dir / "001_no_connect.sql").write_text("CREATE TABLE t (id int);\n")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, entries=[])

    assert gate.check(forward_dir=forward_dir, manifest_path=manifest_path) == []


def test_gate_rejects_a_stale_manifest_entry_for_a_removed_file(tmp_path: Path) -> None:
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    (forward_dir / "001_unrelated.sql").write_text("SELECT 1;\n")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_path,
        entries=[
            {
                "file": "999_long_gone.sql",
                "connect_target": "some_other_database",
                "disposition": "grandfathered",
                "citation": "OMN-1 -- test fixture",
            }
        ],
    )

    violations = gate.check(forward_dir=forward_dir, manifest_path=manifest_path)
    assert len(violations) == 1
    assert violations[0].file == "999_long_gone.sql"
    assert "no live counterpart" in violations[0].reason


def test_gate_rejects_a_manifest_entry_whose_target_drifted(tmp_path: Path) -> None:
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    (forward_dir / "001_drifted.sql").write_text("\\connect new_target\nSELECT 1;\n")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_path,
        entries=[
            {
                "file": "001_drifted.sql",
                "connect_target": "old_target",
                "disposition": "grandfathered",
                "citation": "OMN-1 -- test fixture",
            }
        ],
    )

    violations = gate.check(forward_dir=forward_dir, manifest_path=manifest_path)
    assert len(violations) == 1
    assert "old_target" in violations[0].reason
    assert "new_target" in violations[0].reason


@pytest.mark.parametrize(
    ("bad_field", "bad_value"),
    [("disposition", "vibes-based"), ("citation", "")],
)
def test_malformed_manifest_entry_fails_to_load(
    tmp_path: Path, bad_field: str, bad_value: str
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    entry = {
        "file": "001_x.sql",
        "connect_target": "other_db",
        "disposition": "grandfathered",
        "citation": "OMN-1 -- test fixture",
    }
    entry[bad_field] = bad_value
    _write_manifest(manifest_path, entries=[entry])

    with pytest.raises(AssertionError):
        gate.load_manifest(manifest_path)


def test_duplicate_manifest_entry_fails_to_load(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    entry = {
        "file": "001_x.sql",
        "connect_target": "other_db",
        "disposition": "grandfathered",
        "citation": "OMN-1 -- test fixture",
    }
    _write_manifest(manifest_path, entries=[entry, dict(entry)])

    with pytest.raises(AssertionError, match="duplicate"):
        gate.load_manifest(manifest_path)


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------


def test_main_returns_nonzero_and_prints_on_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    (forward_dir / "200_new.sql").write_text("\\connect other_db\nSELECT 1;\n")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, entries=[])

    rc = gate.main(
        ["--forward-dir", str(forward_dir), "--manifest", str(manifest_path)]
    )
    out = capsys.readouterr()

    assert rc != 0
    assert "OMN-15819" in out.err
    assert "200_new.sql" in out.err


def test_main_returns_zero_when_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    forward_dir = tmp_path / "forward"
    forward_dir.mkdir()
    (forward_dir / "001_clean.sql").write_text("SELECT 1;\n")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, entries=[])

    rc = gate.main(
        ["--forward-dir", str(forward_dir), "--manifest", str(manifest_path)]
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "OK" in out.out
