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
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

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
    assert proof.count("run-forward-migrations.sh") >= 1
    assert "for pass in 1 2" in proof
    assert "fresh-postgres" in proof
    assert "legacy-postgres" in proof
    assert (
        'column "migration_id" of relation "schema_migrations" does not exist' in proof
    )
    assert "OMN-15413" in proof
    assert "fixture_case=legacy_upgrade status=BLOCKED" in proof


def test_required_ci_executes_rebuilt_fixture_and_always_cleans_it() -> None:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["migration-integration"]["steps"]
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
