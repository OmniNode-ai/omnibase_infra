# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The migration gate must inspect actual changed SQL, not fixture strings."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.ci.check_application_database_sql import (
    validate_changed_sql,
    violation_key,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).parents[2]
_OMN15547_FIXTURE = (
    _ROOT
    / "tests"
    / "fixtures"
    / "omn15547"
    / "node-service-registry-tenant-rls-unqualified.sql.captured"
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=OMN-15361 proof",
        "-c",
        "user.email=omn-15361@example.invalid",
        "commit",
        "-m",
        message,
    )
    return _git(repository, "rev-parse", "HEAD")


def _ownership_manifest(
    repository: Path,
    table_name: str,
    *,
    filename: str = "ownership.yaml",
    service: str = "sql_gate_test",
) -> Path:
    path = repository / filename
    path.write_text(
        f"""schema_version: \"1.0\"
service: {service}
target_database_ref: application
db_io:
  db_tables:
    - name: {table_name}
      database_ref: application
      schema: tenant
      migration: qualified.sql
      access: read_write
      role: test_relation
""",
        encoding="utf-8",
    )
    return path


def _function_ownership_manifest(repository: Path) -> Path:
    path = repository / "function-ownership.yaml"
    path.write_text(
        """schema_version: "1.0"
service: sql_gate_function_test
target_database_ref: application
db_io:
  db_tables: []
database_objects:
  - name: safe_report
    kind: function
    database_ref: application
    schema: tenant
    domain: TENANT
    owner_declaration: service:sql_gate_function_test
    function_signature: "()"
""",
        encoding="utf-8",
    )
    return path


def test_gate_checks_only_real_changed_sql_files(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")

    (repository / "unchanged_legacy.sql").write_text(
        "CREATE TABLE legacy_unqualified (id uuid);\n",
        encoding="utf-8",
    )
    base_revision = _commit(repository, "baseline")

    (repository / "qualified.sql").write_text(
        "CREATE TABLE tenant.changed_table (id uuid);\n",
        encoding="utf-8",
    )
    manifest = _ownership_manifest(repository, "changed_table")
    green_head = _commit(repository, "qualified")
    assert not validate_changed_sql(
        repository,
        base_revision,
        green_head,
        ownership_manifest_paths=(manifest,),
    ).violations

    (repository / "unqualified.sql").write_text(
        "SELECT * FROM changed_table;\n",
        encoding="utf-8",
    )
    red_head = _commit(repository, "unqualified")
    violations = validate_changed_sql(
        repository,
        base_revision,
        red_head,
        ownership_manifest_paths=(manifest,),
    ).violations
    assert any("unqualified.sql" in violation for violation in violations)
    assert any("schema-qualified" in violation for violation in violations)
    assert all("unchanged_legacy.sql" not in violation for violation in violations)


def test_gate_exempts_only_the_omn15503_legacy_node_migration_path(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    base_revision = _commit(repository, "baseline")

    exempt = (
        repository
        / "docker"
        / "migrations"
        / "forward"
        / "nodes"
        / "node_projection_delegation"
        / "0029_delegation_terminal_failure_cause.sql"
    )
    exempt.parent.mkdir(parents=True)
    exempt.write_text(
        "ALTER TABLE delegation_events ADD COLUMN terminal_ok boolean;\n",
        encoding="utf-8",
    )
    exempt_head = _commit(repository, "exempt legacy migration")
    assert not validate_changed_sql(
        repository,
        base_revision,
        exempt_head,
        ownership_manifest_paths=(),
    ).violations

    adjacent = exempt.with_name("0030_adjacent_unqualified.sql")
    adjacent.write_text(
        "ALTER TABLE delegation_events ADD COLUMN still_blocked text;\n",
        encoding="utf-8",
    )
    adjacent_head = _commit(repository, "adjacent legacy migration")
    violations = validate_changed_sql(
        repository,
        exempt_head,
        adjacent_head,
        ownership_manifest_paths=(),
    ).violations
    assert any("0030_adjacent_unqualified.sql" in item for item in violations)
    assert any("schema-qualified" in item for item in violations)


def test_gate_exempts_omn15655_legacy_root_shape_repair_paths(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    base_revision = _commit(repository, "baseline")

    migration_dir = repository / "docker" / "migrations" / "forward"
    migration_dir.mkdir(parents=True)
    for filename in (
        "031_create_llm_call_metrics_and_cost_aggregates.sql",
        "050_create_baselines_tables.sql",
    ):
        (migration_dir / filename).write_text(
            "ALTER TABLE public.legacy_shape ADD COLUMN tenant_id uuid;\n",
            encoding="utf-8",
        )

    exempt_head = _commit(repository, "exempt legacy root shape repairs")
    assert not validate_changed_sql(
        repository,
        base_revision,
        exempt_head,
        ownership_manifest_paths=(),
    ).violations

    adjacent = migration_dir / "051_adjacent_unqualified.sql"
    adjacent.write_text(
        "ALTER TABLE legacy_shape ADD COLUMN still_blocked text;\n",
        encoding="utf-8",
    )
    adjacent_head = _commit(repository, "adjacent root migration")
    violations = validate_changed_sql(
        repository,
        exempt_head,
        adjacent_head,
        ownership_manifest_paths=(),
    ).violations
    assert any("051_adjacent_unqualified.sql" in item for item in violations)
    assert any("schema-qualified" in item for item in violations)


def test_qualified_create_requires_an_authoritative_ownership_declaration(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    base_revision = _commit(repository, "baseline")
    (repository / "unclassified.sql").write_text(
        "CREATE TABLE tenant.unclassified (id uuid);\n",
        encoding="utf-8",
    )
    head_revision = _commit(repository, "unclassified")

    violations = validate_changed_sql(
        repository,
        base_revision,
        head_revision,
        ownership_manifest_paths=(),
    ).violations
    assert any("ownership declaration" in violation for violation in violations)


def test_qualified_non_create_target_requires_authoritative_ownership(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    base_revision = _commit(repository, "baseline")
    (repository / "undeclared_alter.sql").write_text(
        "ALTER TABLE tenant.undeclared ADD COLUMN payload text;\n",
        encoding="utf-8",
    )
    head_revision = _commit(repository, "qualified alter")

    violations = validate_changed_sql(
        repository,
        base_revision,
        head_revision,
        ownership_manifest_paths=(),
    ).violations
    assert any("exactly one ownership declaration" in item for item in violations)


def test_duplicate_or_conflicting_owner_declarations_fail_closed(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    base_revision = _commit(repository, "baseline")
    (repository / "qualified.sql").write_text(
        "CREATE TABLE tenant.changed_table (id uuid);\n",
        encoding="utf-8",
    )
    first = _ownership_manifest(
        repository,
        "changed_table",
        filename="owner-a.yaml",
        service="owner_a",
    )
    second = _ownership_manifest(
        repository,
        "changed_table",
        filename="owner-b.yaml",
        service="owner_b",
    )
    head_revision = _commit(repository, "duplicate owners")

    violations = validate_changed_sql(
        repository,
        base_revision,
        head_revision,
        ownership_manifest_paths=(first, second),
    ).violations
    assert any("exactly one ownership declaration" in item for item in violations)


def test_red_control_wrong_routine_overload(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    base_revision = _commit(repository, "baseline")
    (repository / "wrong-overload.sql").write_text(
        "ALTER FUNCTION tenant.safe_report(uuid) OWNER TO owner_onex_tenant;\n",
        encoding="utf-8",
    )
    manifest = _function_ownership_manifest(repository)
    head_revision = _commit(repository, "wrong overload")

    violations = validate_changed_sql(
        repository,
        base_revision,
        head_revision,
        ownership_manifest_paths=(manifest,),
    ).violations
    assert any("exact routine ownership declaration" in item for item in violations)


def test_red_control_wrong_object_kind(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    base_revision = _commit(repository, "baseline")
    (repository / "wrong-kind.sql").write_text(
        "ALTER TABLE tenant.safe_report ADD COLUMN payload text;\n",
        encoding="utf-8",
    )
    manifest = _function_ownership_manifest(repository)
    head_revision = _commit(repository, "wrong kind")

    violations = validate_changed_sql(
        repository,
        base_revision,
        head_revision,
        ownership_manifest_paths=(manifest,),
    ).violations
    assert any("exact object-kind ownership declaration" in item for item in violations)


def test_omn15361_replay_rejects_real_unqualified_application_migration(
    tmp_path: Path,
) -> None:
    """Replay the real node_service_registry migration shape that motivated the gate."""
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    base_revision = _commit(repository, "baseline")
    migration = (
        repository
        / "docker"
        / "migrations"
        / "forward"
        / "nodes"
        / "node_projection_registration"
        / "0002_node_service_registry_tenant_rls.sql"
    )
    migration.parent.mkdir(parents=True)
    migration.write_bytes(_OMN15547_FIXTURE.read_bytes())
    head_revision = _commit(repository, "captured unqualified migration")

    violations = validate_changed_sql(
        repository,
        base_revision,
        head_revision,
        ownership_manifest_paths=(),
    ).violations

    rendered = "\n".join(violations)
    assert "0002_node_service_registry_tenant_rls.sql" in rendered
    assert "schema-qualified" in rendered


# ---------------------------------------------------------------------------
# OMN-15361 frozen-baseline ratchet. Mirrors the OMN-14443 deploy-gate
# grandfather pattern: pre-existing violations soft-pass, everything else is
# held to the full bar, and the list may only shrink.
# ---------------------------------------------------------------------------


def _baseline_repository(tmp_path: Path) -> tuple[Path, str, str, Path]:
    """A repo whose changed SQL carries exactly one unqualified relation."""
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "seed.sql").write_text(
        "CREATE TABLE tenant.seeded (id uuid);\n", encoding="utf-8"
    )
    base_revision = _commit(repository, "baseline")

    (repository / "offender.sql").write_text(
        "SELECT * FROM unqualified_relation;\n", encoding="utf-8"
    )
    manifest = _ownership_manifest(repository, "seeded")
    head = _commit(repository, "offender")
    return repository, base_revision, head, manifest


def _write_baseline(path: Path, entries: list[tuple[str, str]]) -> Path:
    lines = ['generated_at: "2026-08-15"', f"count: {len(entries)}", "violations:"]
    for key, recorded_path in entries:
        lines.append(f'  - key: "{key}"')
        lines.append(f'    path: "{recorded_path}"')
        lines.append('    violation: "recorded"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_unbaselined_violation_is_held_to_the_full_bar(tmp_path: Path) -> None:
    repository, base, head, manifest = _baseline_repository(tmp_path)
    empty = _write_baseline(tmp_path / "baseline.yaml", [])

    outcome = validate_changed_sql(
        repository,
        base,
        head,
        ownership_manifest_paths=(manifest,),
        baseline_path=empty,
    )

    assert any("unqualified_relation" in v for v in outcome.violations)
    assert outcome.grandfathered == 0


def test_baselined_violation_is_grandfathered_and_counted(tmp_path: Path) -> None:
    repository, base, head, manifest = _baseline_repository(tmp_path)
    raw = validate_changed_sql(
        repository, base, head, ownership_manifest_paths=(manifest,)
    )
    offender = next(v for v in raw.violations if "unqualified_relation" in v)
    baseline = _write_baseline(
        tmp_path / "baseline.yaml", [(violation_key(offender), "offender.sql")]
    )

    outcome = validate_changed_sql(
        repository,
        base,
        head,
        ownership_manifest_paths=(manifest,),
        baseline_path=baseline,
    )

    assert not any("unqualified_relation" in v for v in outcome.violations)
    assert outcome.grandfathered == 1


def test_a_corrupt_baseline_grandfathers_nothing(tmp_path: Path) -> None:
    repository, base, head, manifest = _baseline_repository(tmp_path)
    corrupt = tmp_path / "baseline.yaml"
    corrupt.write_text("{ this is: not: valid yaml ][", encoding="utf-8")

    outcome = validate_changed_sql(
        repository,
        base,
        head,
        ownership_manifest_paths=(manifest,),
        baseline_path=corrupt,
    )

    assert any("unqualified_relation" in v for v in outcome.violations)
    assert outcome.grandfathered == 0


def test_a_missing_baseline_grandfathers_nothing(tmp_path: Path) -> None:
    repository, base, head, manifest = _baseline_repository(tmp_path)

    outcome = validate_changed_sql(
        repository,
        base,
        head,
        ownership_manifest_paths=(manifest,),
        baseline_path=tmp_path / "does-not-exist.yaml",
    )

    assert any("unqualified_relation" in v for v in outcome.violations)
    assert outcome.grandfathered == 0


def test_a_baseline_entry_for_a_deleted_file_fails_stale(tmp_path: Path) -> None:
    repository, base, head, manifest = _baseline_repository(tmp_path)
    baseline = _write_baseline(
        tmp_path / "baseline.yaml", [("deadbeefdeadbeef", "removed_migration.sql")]
    )

    outcome = validate_changed_sql(
        repository,
        base,
        head,
        ownership_manifest_paths=(manifest,),
        baseline_path=baseline,
    )

    assert any(
        "stale baseline entry" in v and "no longer exists" in v
        for v in outcome.violations
    )


def test_a_no_longer_firing_baseline_entry_fails_stale(tmp_path: Path) -> None:
    repository, base, head, manifest = _baseline_repository(tmp_path)
    # offender.sql IS linted this run, but this key never fires against it.
    baseline = _write_baseline(
        tmp_path / "baseline.yaml", [("cafebabecafebabe", "offender.sql")]
    )

    outcome = validate_changed_sql(
        repository,
        base,
        head,
        ownership_manifest_paths=(manifest,),
        baseline_path=baseline,
    )

    assert any(
        "stale baseline entry" in v and "no longer fires" in v
        for v in outcome.violations
    )


def test_an_entry_for_an_unscanned_file_is_not_reported_stale(tmp_path: Path) -> None:
    """A dev PR touching two files must not mass-fail on the other 160 entries."""
    repository, base, head, manifest = _baseline_repository(tmp_path)
    # seed.sql exists but is NOT in this run's changed set, so its entry is
    # unobservable — not stale.
    baseline = _write_baseline(
        tmp_path / "baseline.yaml", [("0123456789abcdef", "seed.sql")]
    )

    outcome = validate_changed_sql(
        repository,
        base,
        head,
        ownership_manifest_paths=(manifest,),
        baseline_path=baseline,
    )

    assert not any("stale baseline entry" in v for v in outcome.violations)


def test_violation_key_is_content_addressed_not_positional() -> None:
    same = "a.sql: application relation target 'x' must be schema-qualified"
    assert violation_key(same) == violation_key(same)
    assert violation_key(same) != violation_key(same.replace("a.sql", "b.sql"))
    assert violation_key(same) != violation_key(same.replace("'x'", "'y'"))
