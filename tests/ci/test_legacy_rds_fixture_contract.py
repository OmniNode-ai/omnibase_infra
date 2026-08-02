# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Static anti-vacuity contract for the OMN-15422 PostgreSQL fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "docker" / "legacy-rds-fixture"
COMPOSE = FIXTURE_ROOT / "compose.yml"
DOCKERFILE = FIXTURE_ROOT / "Dockerfile"
MANIFEST = FIXTURE_ROOT / "fixture-manifest.json"
LEGACY_SEED = FIXTURE_ROOT / "legacy-seed.sql"
PROOF = FIXTURE_ROOT / "prove.sh"
PROOF_REPLAY_CAPTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn15547"
    / "legacy-rds-fixture-prove.sh.captured"
)
CUTOVER_PROOF = FIXTURE_ROOT / "cutover-proof" / "prove.sh"
CUTOVER_BOOTSTRAP = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "migration"
    / "cutover"
    / "sql"
    / "bootstrap.sql"
)
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "legacy-rds-fixture-proof.yml"

REQUIRED_CASES = {
    "mapping_ambiguity",
    "checksum_conflict",
    "owner_drift",
    "unsafe_rls_policy",
    "unsafe_view",
    "unsafe_function",
    "transformation_collision",
    "flat_node_shape_collision",
    "legacy_shape_collision",
    "application_migration_ledger",
    "cutover_receipts_and_rollback_boundary",
}


def test_fixture_manifest_is_synthetic_complete_and_discriminating() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["ticket"] == "OMN-15422"
    assert manifest["postgres_major"] == 16
    assert manifest["provenance"]["live_database_read"] is False
    assert manifest["sanitization"]["customer_data"] is False
    assert manifest["sanitization"]["credentials"] is False
    assert set(manifest["database_names"]) >= {
        "omnibase_infra",
        "omnidash_analytics",
        "omninode_cloud",
    }
    assert set(manifest["ledger_shapes"]) == {
        "migration_id_checksum_source_set",
        "version_nullable_checksum",
        "filename_applied_at",
    }

    cases = {case["id"]: case for case in manifest["cases"]}
    assert set(cases) == REQUIRED_CASES
    for case_id, case in cases.items():
        assert case["positive_fixture"], case_id
        assert case["red_fixture"], case_id
        assert case["detector"], case_id
        assert case["expected_red_signature"], case_id


def test_fixture_builds_all_paths_from_postgresql_16_without_credentials() -> None:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"fresh-postgres", "legacy-postgres", "proof"}

    assert not any("ports" in service for service in services.values())
    for name in ("fresh-postgres", "legacy-postgres"):
        service = services[name]
        assert service["build"]["context"] == "../.."
        assert service["build"]["dockerfile"] == (
            "docker/legacy-rds-fixture/Dockerfile"
        )
        assert service["build"]["target"] == name.removesuffix("-postgres")
        assert service["environment"] == {"POSTGRES_HOST_AUTH_METHOD": "trust"}

    proof = services["proof"]
    assert proof["build"]["target"] == "proof"
    assert proof["depends_on"]["fresh-postgres"]["condition"] == "service_healthy"
    assert proof["depends_on"]["legacy-postgres"]["condition"] == "service_healthy"

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert dockerfile.count("FROM postgres:16-alpine") == 3
    assert "PASSWORD" not in dockerfile.upper()

    compose_text = COMPOSE.read_text(encoding="utf-8").upper()
    assert "POSTGRES_PASSWORD" not in compose_text
    assert "PASSWORD:" not in compose_text


def test_legacy_seed_reproduces_the_named_catalog_collision_classes() -> None:
    sql = LEGACY_SEED.read_text(encoding="utf-8")
    for database in ("omnibase_infra", "omnidash_analytics", "omninode_cloud"):
        assert database in sql
    for role in ("omninodeadmin", "role_omnidash", "app_dashboard", "onex_api"):
        assert role in sql
    for relation in (
        "schema_migrations",
        "llm_cost_aggregates",
        "baselines_comparisons",
        "tenant_usage_legacy",
    ):
        assert relation in sql
    for dependency in (
        "CREATE VIEW",
        "CREATE FUNCTION",
        "CREATE INDEX",
        "REFERENCES",
        "ENABLE ROW LEVEL SECURITY",
        "ALTER DEFAULT PRIVILEGES",
    ):
        assert dependency in sql
    for tenant_value in (
        "legacy-acme",
        "omninode",
        "00000000-0000-0000-0000-000000000000",
    ):
        assert tenant_value in sql


def test_proof_runs_real_migrations_twice_and_pins_the_blocked_upgrade() -> None:
    proof = PROOF.read_text(encoding="utf-8")
    replay = PROOF_REPLAY_CAPTURE.read_text(encoding="utf-8")
    assert proof.count("run-forward-migrations.sh") >= 1
    assert "for pass in 1 2" in proof
    assert "fresh-postgres" in proof
    assert "legacy-postgres" in proof
    assert "OMN-15423" in proof
    assert "fixture_case=legacy_upgrade status=PASS" in proof
    assert "Sentinel set. Migration gate will report HEALTHY." in proof
    assert "second pass was not idempotent" in proof
    assert "platform_catalog.schema_migrations" in proof
    assert "fixture_case=application_ledger_fresh" in proof
    assert "fixture_case=application_ledger_legacy" in proof
    assert "selected_oid_preserved=true" in proof
    assert "fixture_case=legacy_upgrade status=BLOCKED" in replay
    assert "fixture_status=PASS_WITH_EXPECTED_BLOCKER blocker=OMN-15423" in replay

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "ledger-control/" in dockerfile
    assert "ledger-control/forward/_ledger/bootstrap.sql" in dockerfile


def test_required_ci_executes_rebuilt_fixture_and_always_cleans_it() -> None:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["fixture-proof"]
    assert "needs" not in job
    assert "OMNI_RUNNER_SELECTOR_V1" in CI_WORKFLOW.read_text(encoding="utf-8")
    assert "OMNI_DOCKER_CI_RUNS_ON_JSON" in job["runs-on"]
    assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" in job["runs-on"]
    steps = job["steps"]
    proof = next(
        step
        for step in steps
        if step.get("name") == "Sanitized legacy-RDS Docker proof"
    )
    assert "docker compose" in proof["run"]
    assert "--build" in proof["run"]
    assert "--exit-code-from proof" in proof["run"]

    cleanup = next(
        step
        for step in steps
        if step.get("name") == "Clean sanitized legacy-RDS Docker proof"
    )
    assert cleanup["if"] == "always()"
    assert "down --volumes --remove-orphans" in cleanup["run"]


def test_cutover_extension_is_durable_and_red_proven_in_the_rebuilt_image() -> None:
    proof = CUTOVER_PROOF.read_text(encoding="utf-8")
    bootstrap = CUTOVER_BOOTSTRAP.read_text(encoding="utf-8")
    outer_proof = PROOF.read_text(encoding="utf-8")
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "cutover-proof/prove.sh" in outer_proof
    assert "cutover-proof/prove.sh" in dockerfile
    assert "migration/cutover/sql/bootstrap.sql" in dockerfile
    for table in (
        "cutover_family_contracts",
        "transformation_receipts",
        "cutover_journal",
        "reverse_delta_proofs",
        "reverse_delta_entries",
    ):
        assert table in bootstrap
    for signature in (
        "cutover_family_mismatch_isolation",
        "cutover_pre_checkpoint_dsn_rollback",
        "cutover_blind_dual_write",
        "cutover_post_checkpoint_direct_rollback",
        "cutover_reverse_delta_coverage",
        "cutover_reverse_delta_complete",
        "cutover_forward_fix_only",
        "cutover_durable_journal",
    ):
        assert signature in proof
