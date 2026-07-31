# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The migration gate must inspect actual changed SQL, not fixture strings."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.ci.check_application_database_sql import validate_changed_sql

pytestmark = pytest.mark.unit


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
    )

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
    )
    assert any("unqualified.sql" in violation for violation in violations)
    assert any("schema-qualified" in violation for violation in violations)
    assert all("unchanged_legacy.sql" not in violation for violation in violations)


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
    )
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
    )
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
    )
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
    )
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
    )
    assert any("exact object-kind ownership declaration" in item for item in violations)
