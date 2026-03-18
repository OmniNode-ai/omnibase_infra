# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the service manifest and bundle schema."""

from __future__ import annotations

import pytest

from omnibase_infra.docker.catalog.manifest_schema import (
    Bundle,
    DependsOnCondition,
    DependsOnEntry,
    HealthCheck,
    PortMapping,
    ServiceLayer,
    ServiceManifest,
)


@pytest.mark.unit
def test_service_manifest_validates_required_fields() -> None:
    manifest = ServiceManifest(
        name="postgres",
        description="Primary PostgreSQL data store",
        image="postgres:16-alpine",
        layer=ServiceLayer.INFRASTRUCTURE,
        required_env=["POSTGRES_PASSWORD"],
        hardcoded_env={"POSTGRES_DB": "omnibase_infra"},
        operational_defaults={"POSTGRES_USER": "postgres"},
        ports=PortMapping(external=5436, internal=5432),
        healthcheck=HealthCheck(
            test="pg_isready -U postgres -d omnibase_infra",
            interval_s=30,
            timeout_s=10,
            retries=3,
            start_period_s=10,
        ),
        volumes=["postgres_data:/var/lib/postgresql/data"],
        depends_on=[],
    )
    assert manifest.name == "postgres"
    assert "POSTGRES_PASSWORD" in manifest.required_env
    assert manifest.layer == ServiceLayer.INFRASTRUCTURE
    assert manifest.ports is not None
    assert manifest.ports.external == 5436


@pytest.mark.unit
def test_service_manifest_rejects_invalid_layer() -> None:
    with pytest.raises(ValueError):
        ServiceManifest(
            name="bad",
            description="",
            image="x",
            layer="bogus",  # type: ignore[arg-type]
            required_env=[],
            hardcoded_env={},
            operational_defaults={},
            ports=PortMapping(external=1, internal=1),
            healthcheck=HealthCheck(test="true"),
            volumes=[],
            depends_on=[],
        )


@pytest.mark.unit
def test_depends_on_with_conditions() -> None:
    dep = DependsOnEntry(
        service="postgres", condition=DependsOnCondition.SERVICE_HEALTHY
    )
    assert dep.condition == DependsOnCondition.SERVICE_HEALTHY

    dep_completed = DependsOnEntry(
        service="forward-migration",
        condition=DependsOnCondition.SERVICE_COMPLETED_SUCCESSFULLY,
    )
    assert dep_completed.condition == DependsOnCondition.SERVICE_COMPLETED_SUCCESSFULLY


@pytest.mark.unit
def test_one_shot_service() -> None:
    """Migration runners use restart='no' and are one-shot."""
    manifest = ServiceManifest(
        name="forward-migration",
        description="Apply pending forward migrations",
        image="postgres:16-alpine",
        layer=ServiceLayer.INFRASTRUCTURE,
        required_env=["POSTGRES_PASSWORD"],
        hardcoded_env={},
        operational_defaults={},
        ports=None,
        healthcheck=None,
        volumes=["../scripts/run-forward-migrations.sh:/run-forward-migrations.sh:ro"],
        depends_on=[DependsOnEntry("postgres", DependsOnCondition.SERVICE_HEALTHY)],
        restart="no",
        command=["sh", "/run-forward-migrations.sh"],
    )
    assert manifest.restart == "no"
    assert manifest.healthcheck is None  # one-shot services have no healthcheck


@pytest.mark.unit
def test_bundle_resolves_transitive_dependencies() -> None:
    pg = ServiceManifest(
        name="postgres",
        description="",
        image="postgres:16-alpine",
        layer=ServiceLayer.INFRASTRUCTURE,
        required_env=["POSTGRES_PASSWORD"],
        hardcoded_env={},
        operational_defaults={},
        ports=PortMapping(external=5436, internal=5432),
        healthcheck=HealthCheck(test="pg_isready"),
        volumes=[],
        depends_on=[],
    )
    valkey = ServiceManifest(
        name="valkey",
        description="",
        image="valkey/valkey:8.0-alpine",
        layer=ServiceLayer.INFRASTRUCTURE,
        required_env=["VALKEY_PASSWORD"],
        hardcoded_env={},
        operational_defaults={},
        ports=PortMapping(external=16379, internal=6379),
        healthcheck=HealthCheck(test="valkey-cli ping"),
        volumes=[],
        depends_on=[],
    )
    bundle = Bundle(
        name="runtime",
        description="ONEX runtime",
        services=["omninode-runtime"],
        includes=["core"],
    )
    # The runtime service depends on postgres and redpanda
    runtime = ServiceManifest(
        name="omninode-runtime",
        description="",
        image="runtime:latest",
        layer=ServiceLayer.RUNTIME,
        required_env=["POSTGRES_PASSWORD", "VALKEY_PASSWORD"],
        hardcoded_env={"KAFKA_BOOTSTRAP_SERVERS": "redpanda:9092"},
        operational_defaults={},
        ports=PortMapping(external=8085, internal=8085),
        healthcheck=HealthCheck(test="curl -sf http://localhost:8085/health"),
        volumes=[],
        depends_on=[
            DependsOnEntry("postgres", DependsOnCondition.SERVICE_HEALTHY),
            DependsOnEntry("redpanda", DependsOnCondition.SERVICE_HEALTHY),
        ],
    )
    catalog = {"postgres": pg, "valkey": valkey, "omninode-runtime": runtime}
    all_required = bundle.all_required_env(catalog=catalog)
    assert "POSTGRES_PASSWORD" in all_required
    assert "VALKEY_PASSWORD" in all_required


@pytest.mark.unit
def test_bundle_rejects_circular_dependency() -> None:
    """Bundles that include each other must raise."""
    with pytest.raises(ValueError, match="circular"):
        Bundle(name="a", description="", services=[], includes=["b"]).resolve_includes(
            bundles={
                "a": Bundle(name="a", description="", services=[], includes=["b"]),
                "b": Bundle(name="b", description="", services=[], includes=["a"]),
            }
        )
