# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage: the SQL gate accepts a grant on a sequence the manifest declares bare.

Drives the production gate entry point (``validate_changed_sql``) over a real git
history: a changed SQL file grants on ``public.tenant_event_outbox_id_seq`` while the
ownership manifest lists that sequence as a bare name under its domain, the shape of
the cloud ownership manifest's entry, with no ``database_ref`` or ``schema``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from scripts.ci.check_application_database_sql import validate_changed_sql

_GRANT = 'GRANT USAGE ON SEQUENCE public.tenant_event_outbox_id_seq TO "tenant_identity_writer";\n'

_MANIFEST = """\
schema_version: "1.0"
service: onex_api
target_database_ref: application
db_io:
  db_tables:
    - name: tenant_event_outbox
      database_ref: application
      schema: omninode_internal
      migration: 20260816_tenant_event_outbox.sql
      access: read_write
      role: tenant_lifecycle_outbox
database_objects:
  - name: tenant_event_outbox_id_seq
    kind: sequence
    domain: OMNINODE_INTERNAL
    owner_declaration: service:onex_api
    dependencies: [{dependency}]
"""


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        env=scrub_git_location_env(os.environ),
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
        "user.name=OMN-26471 proof",
        "-c",
        "user.email=omn-26471@example.invalid",
        "commit",
        "-m",
        message,
    )
    return _git(repository, "rev-parse", "HEAD")


def _violations(tmp_path: Path, dependency: str) -> tuple[str, ...]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "README.md").write_text("baseline\n", encoding="utf-8")
    base = _commit(repository, "baseline")
    (repository / "seam.sql").write_text(_GRANT, encoding="utf-8")
    manifest = repository / "ownership.yaml"
    manifest.write_text(_MANIFEST.format(dependency=dependency), encoding="utf-8")
    head = _commit(repository, "grant on the outbox sequence")
    return validate_changed_sql(
        repository, base, head, ownership_manifest_paths=(manifest,)
    ).violations


@pytest.mark.integration
def test_grant_on_a_bare_declared_sequence_passes_the_gate(tmp_path: Path) -> None:
    assert _violations(tmp_path, "tenant_event_outbox") == ()


@pytest.mark.integration
def test_grant_on_a_sequence_with_no_owning_table_is_still_refused(
    tmp_path: Path,
) -> None:
    """Positive control: the gate still refuses an undeclared sequence."""
    violations = _violations(tmp_path, "absent_table")

    assert any(
        "public.tenant_event_outbox_id_seq requires exactly one ownership declaration"
        in violation
        for violation in violations
    )
