# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Authority-universe isolation controls for the OMN-15361 SQL gate."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.validation.application_database_domain_enforcement import (
    application_database_created_catalog_identities,
    application_database_sql_target_requirements,
    lint_application_database_sql,
    load_application_database_ownership_identities,
)
from scripts.ci.check_application_database_sql import changed_sql_paths

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).parents[2]
_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_PROOF_MANIFEST = _ROOT / "config" / "application_database_domain_proof_ownership.yaml"
_PROOF_SEED = _ROOT / "docker" / "application-domain-enforcement" / "seed.sql"


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


def test_ephemeral_proof_seed_is_not_a_deployable_changed_sql_path(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    base_revision = _commit(repository, "baseline")

    proof_directory = repository / "docker" / "application-domain-enforcement"
    proof_directory.mkdir(parents=True)
    (proof_directory / "seed.sql").write_text(
        "CREATE TABLE tenant.proof_only (id uuid);\n",
        encoding="utf-8",
    )
    migration_directory = repository / "migrations"
    migration_directory.mkdir()
    deployed = migration_directory / "deployed.sql"
    deployed.write_text(
        "CREATE TABLE tenant.deployed (id uuid);\n",
        encoding="utf-8",
    )
    head_revision = _commit(repository, "changed SQL")

    assert changed_sql_paths(repository, base_revision, head_revision) == (deployed,)


def test_fixture_and_control_bootstrap_sql_are_not_deployable_changed_sql(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    base_revision = _commit(repository, "baseline")

    excluded_paths = (
        repository / "docker/legacy-rds-fixture/legacy-seed.sql",
        repository
        / "docker/legacy-rds-fixture/ledger-control/forward/000_db_metadata.sql",
        repository / "docker/migrations/forward/_ledger/bootstrap.sql",
        repository / "src/omnibase_infra/migration/cutover/sql/bootstrap.sql",
    )
    for path in excluded_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "CREATE TABLE public.fixture_only (id uuid);\n", encoding="utf-8"
        )

    deployed = repository / "docker/migrations/forward/nodes/node_real/0001.sql"
    deployed.parent.mkdir(parents=True)
    deployed.write_text("CREATE TABLE tenant.deployed (id uuid);\n", encoding="utf-8")
    head_revision = _commit(repository, "changed SQL")

    assert changed_sql_paths(repository, base_revision, head_revision) == (deployed,)


def test_production_sql_workflow_never_composes_ephemeral_proof_authority() -> None:
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    trusted_step = workflow.split(
        "- name: Enforce schema qualification in changed SQL (trusted)",
        maxsplit=1,
    )[1].split("- name:", maxsplit=1)[0]
    fork_step = workflow.split(
        "- name: Enforce schema qualification in changed SQL (public fork, fail closed)",
        maxsplit=1,
    )[1].split("- name:", maxsplit=1)[0]

    assert "config/application_database_domain_proof_ownership.yaml" not in trusted_step
    assert "config/application_database_domain_proof_ownership.yaml" not in fork_step
    assert ".proof-dependencies/omninode-infra" in trusted_step
    assert ".proof-dependencies/omninode-infra" not in fork_step


def test_mandatory_source_lane_executes_sql_regression_and_isolation_controls() -> None:
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    source_step = workflow.split(
        "- name: Execute mandatory source assertions and seeded RED controls",
        maxsplit=1,
    )[1].split("- name:", maxsplit=1)[0]

    assert (
        "tests/unit/validation/"
        "test_application_database_sql_enforcement_regressions.py" in source_step
    )
    assert (
        "tests/ci/test_application_database_sql_authority_isolation.py" in source_step
    )


def test_ephemeral_proof_seed_has_an_isolated_exact_authority_universe() -> None:
    topology = load_topology_profile("local")
    sql = _PROOF_SEED.read_text(encoding="utf-8")
    authoritative = load_application_database_ownership_identities((_PROOF_MANIFEST,))
    authoritative_identities = {identity.identity for identity in authoritative}

    assert not lint_application_database_sql(sql, topology)
    assert all(
        any(
            identity.schema == requirement.schema
            and identity.name == requirement.name
            and identity.kind in requirement.allowed_kinds
            and (
                requirement.function_signature is None
                or identity.function_signature == requirement.function_signature
            )
            for identity in authoritative
        )
        for requirement in application_database_sql_target_requirements(sql, topology)
    )
    assert {
        identity.identity
        for identity in application_database_created_catalog_identities(sql)
    } == authoritative_identities
