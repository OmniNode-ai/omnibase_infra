# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the compose generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.generator import generate_compose
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


@pytest.mark.unit
def test_generated_compose_db_urls_use_docker_dns() -> None:
    """DB URLs in the runtime bundle must use Docker-internal addresses."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    # generate_compose will raise ValueError if localhost is detected
    compose = generate_compose(resolved)
    # Also verify the actual URL values in the generated compose
    rt_env = compose["services"]["omninode-runtime"]["environment"]
    assert "postgres:5432" in rt_env["OMNIBASE_INFRA_DB_URL"]
    assert "localhost" not in rt_env["OMNIBASE_INFRA_DB_URL"]
    assert "postgres:5432" in rt_env["OMNIINTELLIGENCE_DB_URL"]
    assert "localhost" not in rt_env["OMNIINTELLIGENCE_DB_URL"]


@pytest.mark.unit
def test_generated_compose_consumer_dsn_uses_docker_dns() -> None:
    """Consumer DSN vars must use Docker-internal addresses."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    aac_env = compose["services"]["agent-actions-consumer"]["environment"]
    assert "postgres:5432" in aac_env["OMNIBASE_INFRA_AGENT_ACTIONS_POSTGRES_DSN"]
    assert "localhost" not in aac_env["OMNIBASE_INFRA_AGENT_ACTIONS_POSTGRES_DSN"]


@pytest.mark.unit
def test_generator_rejects_localhost_in_db_url() -> None:
    """Generator must raise ValueError if a manifest has localhost in a DB URL."""
    from omnibase_infra.docker.catalog.manifest_schema import CatalogManifest
    from omnibase_infra.docker.catalog.resolver import ResolvedStack

    bad_manifest = CatalogManifest(
        name="bad-svc",
        description="test",
        image="test:latest",
        layer="runtime",
        required_env=["POSTGRES_PASSWORD"],
        hardcoded_env={
            "DB_URL": "postgresql://postgres:pw@localhost:5436/mydb",
        },
        operational_defaults={},
        ports=None,
        healthcheck=None,
        volumes=[],
        depends_on=[],
    )
    resolved = ResolvedStack(
        manifests={"bad-svc": bad_manifest},
        required_env={"POSTGRES_PASSWORD"},
        injected_env={},
    )
    with pytest.raises(ValueError, match="localhost detected"):
        generate_compose(resolved)
