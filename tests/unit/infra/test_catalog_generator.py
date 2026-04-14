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
def test_runtime_effects_has_omnimemory_network() -> None:
    """runtime-effects must join omnimemory-network so PluginMemory can reach Memgraph."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    svc_networks = compose["services"]["runtime-effects"]["networks"]
    assert "omnimemory-network" in svc_networks
    assert "omnibase-infra-network" in svc_networks


@pytest.mark.unit
def test_extra_networks_declared_external_in_top_level() -> None:
    """Extra networks must appear as external in the top-level networks block."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    top_networks = compose["networks"]
    assert "omnimemory-network" in top_networks
    assert top_networks["omnimemory-network"]["external"] is True


@pytest.mark.unit
def test_extra_networks_absent_for_services_without_them() -> None:
    """Services without extra_networks must only be on omnibase-infra-network."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["core"])
    compose = generate_compose(resolved)
    pg_networks = compose["services"]["postgres"]["networks"]
    assert pg_networks == ["omnibase-infra-network"]
    assert "omnimemory-network" not in compose["networks"]
