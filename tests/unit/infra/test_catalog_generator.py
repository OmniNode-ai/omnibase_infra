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
def test_generated_runtime_compose_preserves_runtime_image_build() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)

    build = compose["services"]["omninode-runtime"]["build"]
    assert build["context"] == ".."
    assert build["dockerfile"] == "docker/Dockerfile.runtime"
    assert build["args"]["BUILD_SOURCE"] == "${BUILD_SOURCE:-release}"
    assert build["args"]["EXPECTED_BUILD_SOURCE"] == "${EXPECTED_BUILD_SOURCE:-release}"
    assert build["args"]["OMNI_HOME"] == "${OMNI_HOME:-}"
    assert build["args"]["GIT_SHA"] == "${GIT_SHA:-unknown}"
    assert build["args"]["VCS_REF"] == "${VCS_REF:-}"
    assert build["args"]["BUILD_DATE"] == "${BUILD_DATE:-}"


@pytest.mark.unit
def test_generated_compose_preserves_redpanda_ulimits() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["core"])
    compose = generate_compose(resolved)
    redpanda = compose["services"]["redpanda"]

    assert redpanda["ulimits"] == {"nofile": {"soft": 65535, "hard": 65535}}


@pytest.mark.unit
def test_intelligence_migration_uses_lane_namespaced_container_name() -> None:
    """OMN-13201: intelligence-migration must carry the lane prefix.

    Regression for the DEV effects crash-loop's migration gap: the base
    (DEV-lane) intelligence-migration one-shot used the fixed, non-lane-prefixed
    container_name ``omnibase-intelligence-migration`` while every other lane and
    every sibling migration service uses the ``omnibase-infra-`` prefix
    (``omnibase-infra-forward-migration``, ``omnibase-infra-migration-gate``).
    The mismatched name meant the DEV one-shot never ran, so
    ``015_create_db_metadata.sql`` was never applied to the DEV omniintelligence
    database and the runtime entrypoint logged "relation public.db_metadata does
    not exist" on every effects boot. The name must match its siblings.
    """
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    services = compose["services"]
    assert (
        services["intelligence-migration"]["container_name"]
        == "omnibase-infra-intelligence-migration"
    )
    # Sibling migration one-shots define the shared lane-prefix convention.
    assert (
        services["forward-migration"]["container_name"]
        == "omnibase-infra-forward-migration"
    )
    assert (
        services["migration-gate"]["container_name"] == "omnibase-infra-migration-gate"
    )


@pytest.mark.unit
def test_generated_compose_preserves_one_shot_semantics() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    fm = compose["services"]["forward-migration"]
    assert fm["restart"] == "no"
    assert "healthcheck" not in fm  # one-shot entries have no healthcheck
    assert fm["environment"]["NODE_POSTGRES_DB"] == "omnidash_analytics"


@pytest.mark.unit
def test_generated_migration_gate_requires_projection_tables() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    gate = compose["services"]["migration-gate"]
    assert gate["environment"]["NODE_POSTGRES_DB"] == "omnidash_analytics"
    assert (
        gate["environment"]["REQUIRED_PROJECTION_TABLES"]
        == "delegation_events node_service_registry"
    )


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
def test_generated_runtime_services_export_onex_state_dir() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)

    for service_name in ("omninode-runtime", "runtime-effects", "runtime-worker"):
        runtime_env = compose["services"][service_name]["environment"]
        assert runtime_env["ONEX_STATE_ROOT"] == "/app/data/.onex_state"
        assert runtime_env["ONEX_STATE_DIR"] == runtime_env["ONEX_STATE_ROOT"]


@pytest.mark.unit
def test_generated_compose_includes_network_and_volumes() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["core"])
    compose = generate_compose(resolved)
    assert "omnibase-infra-network" in compose["networks"]
    assert "postgres_data" in compose.get("volumes", {})


@pytest.mark.unit
def test_runtime_effects_stays_on_default_network() -> None:
    """runtime-effects should only join the default runtime bridge."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    svc_networks = compose["services"]["runtime-effects"]["networks"]
    assert svc_networks == ["omnibase-infra-network"]


@pytest.mark.unit
def test_runtime_bundle_has_no_external_networks() -> None:
    """The reduced runtime bundle should not declare external helper networks."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    compose = generate_compose(resolved)
    top_networks = compose["networks"]
    assert top_networks == {
        "omnibase-infra-network": {"name": "omnibase-infra-network", "driver": "bridge"}
    }


@pytest.mark.unit
def test_extra_networks_absent_for_services_without_them() -> None:
    """Services without extra_networks must only be on omnibase-infra-network."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["core"])
    compose = generate_compose(resolved)
    pg_networks = compose["services"]["postgres"]["networks"]
    assert pg_networks == ["omnibase-infra-network"]
    assert compose["networks"] == {
        "omnibase-infra-network": {"name": "omnibase-infra-network", "driver": "bridge"}
    }
