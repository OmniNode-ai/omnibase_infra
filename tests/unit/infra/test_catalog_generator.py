# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the compose generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.generator import (
    _validate_no_localhost_db_urls,
    generate_compose,
)
from omnibase_infra.docker.catalog.manifest_schema import CatalogManifest
from omnibase_infra.docker.catalog.resolver import CatalogResolver

CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")


@pytest.mark.unit
def test_generated_compose_preserves_container_names() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["core"])
    compose = generate_compose(resolved)
    services = compose["services"]
    assert services["postgres"]["container_name"] == "omnibase-infra-postgres"
    assert services["redpanda"]["container_name"] == "omnibase-infra-redpanda"


@pytest.mark.unit
def test_generated_compose_preserves_healthcheck_timing() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["core"])
    compose = generate_compose(resolved)
    pg_hc = compose["services"]["postgres"]["healthcheck"]
    assert pg_hc["interval"] == "30s"
    assert pg_hc["timeout"] == "10s"
    assert pg_hc["retries"] == 3
    assert pg_hc["start_period"] == "10s"


@pytest.mark.unit
def test_generated_compose_preserves_depends_on_conditions() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    rt_deps = compose["services"]["omninode-runtime"]["depends_on"]
    assert rt_deps["migration-gate"]["condition"] == "service_healthy"


@pytest.mark.unit
def test_generated_compose_preserves_one_shot_semantics() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    fm = compose["services"]["forward-migration"]
    assert fm["restart"] == "no"
    assert "healthcheck" not in fm  # one-shot entries have no healthcheck


@pytest.mark.unit
def test_generated_compose_injects_bundle_env() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime", "memgraph"])
    compose = generate_compose(resolved)
    rt_env = compose["services"]["omninode-runtime"]["environment"]
    assert rt_env["OMNIMEMORY_ENABLED"] == "true"


@pytest.mark.unit
def test_generated_compose_omits_bundle_env_when_unselected() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    rt_env = compose["services"]["omninode-runtime"]["environment"]
    assert "OMNIMEMORY_ENABLED" not in rt_env


@pytest.mark.unit
def test_generated_compose_includes_network_and_volumes() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["core"])
    compose = generate_compose(resolved)
    assert "omnibase-infra-network" in compose["networks"]
    assert "postgres_data" in compose.get("volumes", {})


# --- Regression guard: DB URLs must use Docker-internal addresses [OMN-5700] ---


def _make_manifest(
    name: str = "test-svc",
    required_env: list[str] | None = None,
    hardcoded_env: dict[str, str] | None = None,
    operational_defaults: dict[str, str] | None = None,
) -> CatalogManifest:
    """Create a minimal CatalogManifest for validation tests."""
    return CatalogManifest(
        name=name,
        description="test",
        image="test:latest",
        layer=EnumInfraLayer.RUNTIME,
        required_env=required_env or [],
        hardcoded_env=hardcoded_env or {},
        operational_defaults=operational_defaults or {},
        ports=None,
        healthcheck=None,
        volumes=[],
        depends_on=[],
    )


@pytest.mark.unit
def test_rejects_db_url_in_required_env() -> None:
    """DB URL vars in required_env would pass through localhost from host .env."""
    manifest = _make_manifest(required_env=["OMNIBASE_INFRA_DB_URL"])
    with pytest.raises(ValueError, match="required_env"):
        _validate_no_localhost_db_urls(manifest)


@pytest.mark.unit
def test_rejects_dsn_in_required_env() -> None:
    manifest = _make_manifest(
        required_env=["OMNIBASE_INFRA_AGENT_ACTIONS_POSTGRES_DSN"]
    )
    with pytest.raises(ValueError, match="required_env"):
        _validate_no_localhost_db_urls(manifest)


@pytest.mark.unit
def test_rejects_localhost_in_hardcoded_db_url() -> None:
    manifest = _make_manifest(
        hardcoded_env={
            "OMNIINTELLIGENCE_DB_URL": "postgresql://postgres:pw@localhost:5436/omniintelligence"
        }
    )
    with pytest.raises(ValueError, match="localhost"):
        _validate_no_localhost_db_urls(manifest)


@pytest.mark.unit
def test_accepts_docker_internal_db_url() -> None:
    """DB URLs using Docker-internal hostnames should pass validation."""
    manifest = _make_manifest(
        hardcoded_env={
            "OMNIBASE_INFRA_DB_URL": "postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/omnibase_infra",
            "OMNIINTELLIGENCE_DB_URL": "postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/omniintelligence",
        }
    )
    # Should not raise
    _validate_no_localhost_db_urls(manifest)


@pytest.mark.unit
def test_accepts_non_db_localhost_env_vars() -> None:
    """Non-DB env vars using localhost (e.g. healthcheck URLs) are fine."""
    manifest = _make_manifest(
        operational_defaults={
            "KEYCLOAK_ISSUER": "http://localhost:28080/realms/omninode",
            "CORS_ORIGINS": "http://localhost:3000",
        }
    )
    # Should not raise
    _validate_no_localhost_db_urls(manifest)


@pytest.mark.unit
def test_real_catalog_runtime_bundle_passes_db_url_guard() -> None:
    """Validate the actual catalog YAML files pass the DB URL guard.

    This is the key regression test: if someone re-introduces a DB URL
    in required_env or adds localhost to a hardcoded DB URL, this test
    will catch it at CI time.
    """
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    # generate_compose calls _validate_no_localhost_db_urls for each service
    compose = generate_compose(resolved)
    # If we get here, all services passed validation
    assert "services" in compose
