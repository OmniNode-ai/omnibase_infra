# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Ratchet tests for migration-gate healthcheck start_period (OMN-12973).

These tests are the enforcement half of the P2.8 fix. They run in the standard
``uv run pytest tests/`` suite, so they gate every PR (CI) and pre-commit.

The classification (P2.8): the prod migration-gate was reported UNHEALTHY then
self-resolved to HEALTHY because its healthcheck ``start_period`` (10s) was far
shorter than the real migration window (~2 min on a cold prod volume). The gate
is a correctly-modeled long-running sentinel, NOT a mis-modeled one-shot — the
fix is to widen ``start_period`` so a still-applying gate stays in
``health: starting`` instead of flipping to UNHEALTHY.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.enum_depends_on_condition import (
    EnumDependsOnCondition,
)
from omnibase_infra.docker.catalog.manifest_schema import (
    CatalogManifest,
    DependsOnEntry,
    HealthCheck,
)
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.docker.catalog.validator_healthcheck_start_period import (
    MIGRATION_GATE_START_PERIOD_FLOOR_S,
    is_migration_completion_gate,
    validate_migration_gate_start_period,
)

CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")


def _gate(start_period_s: int) -> CatalogManifest:
    """A minimal migration-completion-gate-shaped manifest for unit testing."""
    return CatalogManifest(
        name="migration-gate",
        description="test gate",
        image="postgres:16-alpine",
        layer="infrastructure",
        required_env=[],
        hardcoded_env={},
        operational_defaults={},
        ports=None,
        healthcheck=HealthCheck(
            test="sh /check_migrations_complete.sh",
            interval_s=10,
            timeout_s=5,
            retries=30,
            start_period_s=start_period_s,
        ),
        volumes=[],
        depends_on=[
            DependsOnEntry(
                service="forward-migration",
                condition=EnumDependsOnCondition.SERVICE_COMPLETED_SUCCESSFULLY,
            ),
        ],
    )


@pytest.mark.unit
def test_real_catalog_migration_gate_meets_start_period_floor() -> None:
    """The shipped migration-gate manifest must satisfy the start_period floor.

    This is the regression lock for the prod fix: if anyone lowers
    migration-gate.start_period_s below the floor, this fails.
    """
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    result = validate_migration_gate_start_period(resolved.manifests)
    assert result.ok, result.report()


@pytest.mark.unit
def test_migration_gate_is_classified_as_completion_gate() -> None:
    """migration-gate must be recognized as a migration-completion gate.

    Guards against the classifier silently no-op'ing (which would let a
    too-short start_period slip through unvalidated).
    """
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    assert "migration-gate" in resolved.manifests
    assert is_migration_completion_gate(resolved.manifests["migration-gate"])


@pytest.mark.unit
def test_validator_flags_too_short_start_period() -> None:
    """A gate below the floor (the pre-fix 10s value) must be flagged."""
    result = validate_migration_gate_start_period({"migration-gate": _gate(10)})
    assert not result.ok
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.service == "migration-gate"
    assert violation.start_period_s == 10
    assert violation.floor_s == MIGRATION_GATE_START_PERIOD_FLOOR_S
    assert "UNHEALTHY" in violation.message()


@pytest.mark.unit
def test_validator_passes_at_floor_boundary() -> None:
    """A gate exactly at the floor passes (>= semantics)."""
    result = validate_migration_gate_start_period(
        {"migration-gate": _gate(MIGRATION_GATE_START_PERIOD_FLOOR_S)}
    )
    assert result.ok


@pytest.mark.unit
def test_non_migration_gate_is_not_flagged() -> None:
    """Services without a service_completed_successfully dep are ignored.

    A plain healthcheck service (e.g. postgres) with a short start_period must
    not be swept into the migration-gate floor.
    """
    postgres = CatalogManifest(
        name="postgres",
        description="db",
        image="postgres:16-alpine",
        layer="infrastructure",
        required_env=[],
        hardcoded_env={},
        operational_defaults={},
        ports=None,
        healthcheck=HealthCheck(
            test="pg_isready",
            interval_s=30,
            timeout_s=10,
            retries=3,
            start_period_s=10,
        ),
        volumes=[],
        depends_on=[],
    )
    assert not is_migration_completion_gate(postgres)
    result = validate_migration_gate_start_period({"postgres": postgres})
    assert result.ok


@pytest.mark.unit
def test_app_service_depending_on_migration_runner_is_not_flagged() -> None:
    """An app service whose healthcheck is a liveness probe is NOT a gate.

    Regression for the over-broad classifier (OMN-12973): ``intelligence-api``
    depends on ``intelligence-migration`` via ``service_completed_successfully``
    but its healthcheck is an HTTP liveness probe, not a migration-completion
    poll. Its start_period (40s) is legitimately short and must not be swept
    into the migration-gate floor. Only the sentinel whose healthcheck literally
    polls migration completion is subject to the floor.
    """
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    assert "intelligence-api" in resolved.manifests
    intelligence_api = resolved.manifests["intelligence-api"]
    # It does depend on a migration runner completing...
    assert any(
        dep.condition is EnumDependsOnCondition.SERVICE_COMPLETED_SUCCESSFULLY
        for dep in intelligence_api.depends_on
    )
    # ...but its healthcheck is a liveness probe, so it is NOT a completion gate.
    assert not is_migration_completion_gate(intelligence_api)
    result = validate_migration_gate_start_period(resolved.manifests)
    assert result.ok, result.report()


@pytest.mark.unit
def test_completion_gate_requires_migration_probe_healthcheck() -> None:
    """A migration-runner dependant with a non-migration healthcheck is ignored.

    Unit-level guard for the discriminating signal: depending on a
    ``service_completed_successfully`` sibling is necessary but not sufficient —
    the healthcheck command must also poll migration completion.
    """
    app_service = CatalogManifest(
        name="some-api",
        description="app",
        image="runtime:latest",
        layer="runtime",
        required_env=[],
        hardcoded_env={},
        operational_defaults={},
        ports=None,
        healthcheck=HealthCheck(
            test="curl -sf http://localhost:8053/health",
            interval_s=30,
            timeout_s=10,
            retries=3,
            start_period_s=40,
        ),
        volumes=[],
        depends_on=[
            DependsOnEntry(
                service="intelligence-migration",
                condition=EnumDependsOnCondition.SERVICE_COMPLETED_SUCCESSFULLY,
            ),
        ],
    )
    assert not is_migration_completion_gate(app_service)
    result = validate_migration_gate_start_period({"some-api": app_service})
    assert result.ok


@pytest.mark.unit
def test_completion_gate_with_list_form_healthcheck_is_classified() -> None:
    """The migration probe is detected in CMD (argv-list) healthcheck form too."""
    gate = CatalogManifest(
        name="migration-gate",
        description="gate",
        image="postgres:16-alpine",
        layer="infrastructure",
        required_env=[],
        hardcoded_env={},
        operational_defaults={},
        ports=None,
        healthcheck=HealthCheck(
            test=["CMD-SHELL", "sh /check_migrations_complete.sh"],
            interval_s=10,
            timeout_s=5,
            retries=30,
            start_period_s=180,
        ),
        volumes=[],
        depends_on=[
            DependsOnEntry(
                service="forward-migration",
                condition=EnumDependsOnCondition.SERVICE_COMPLETED_SUCCESSFULLY,
            ),
        ],
    )
    assert is_migration_completion_gate(gate)
