# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for catalog extra_networks support (OMN-8713)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver

REPO_ROOT = Path(__file__).parent.parent.parent
CATALOG_DIR = str(REPO_ROOT / "docker" / "catalog")


@pytest.mark.integration
def test_runtime_effects_has_no_extra_networks() -> None:
    """runtime-effects should stay on the default runtime network only."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    manifest = resolver._manifests.get("runtime-effects")
    if manifest is None:
        pytest.skip("runtime-effects manifest not found in catalog — skipping")

    assert manifest.extra_networks == []


@pytest.mark.integration
def test_generate_compose_runtime_effects_stays_on_default_network() -> None:
    """runtime-effects should not join any external networks."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    manifest = resolver._manifests.get("runtime-effects")
    if manifest is None:
        pytest.skip("runtime-effects manifest not found in catalog — skipping")

    from omnibase_infra.docker.catalog.resolver import ResolvedStack

    stack = ResolvedStack(
        manifests={"runtime-effects": manifest},
        required_env=set(manifest.required_env),
        injected_env={},
    )
    compose = generate_compose(stack)

    svc_networks = compose["services"]["runtime-effects"]["networks"]  # type: ignore[index]
    assert svc_networks == ["omnibase-infra-network"]


@pytest.mark.integration
def test_generate_compose_has_no_external_networks_for_runtime_effects() -> None:
    """runtime-effects should not force external networks into the compose graph."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    manifest = resolver._manifests.get("runtime-effects")
    if manifest is None:
        pytest.skip("runtime-effects manifest not found in catalog — skipping")

    from omnibase_infra.docker.catalog.resolver import ResolvedStack

    stack = ResolvedStack(
        manifests={"runtime-effects": manifest},
        required_env=set(manifest.required_env),
        injected_env={},
    )
    compose = generate_compose(stack)

    top_networks = compose["networks"]  # type: ignore[index]
    assert top_networks == {
        "omnibase-infra-network": {"name": "omnibase-infra-network", "driver": "bridge"}
    }


@pytest.mark.integration
def test_generate_compose_no_extra_networks_no_external_entry() -> None:
    """Services without extra_networks must not produce external network entries."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    # postgres has no extra_networks
    manifest = resolver._manifests.get("postgres")
    if manifest is None:
        pytest.skip("postgres manifest not found in catalog — skipping")

    assert not manifest.extra_networks

    from omnibase_infra.docker.catalog.resolver import ResolvedStack

    stack = ResolvedStack(
        manifests={"postgres": manifest},
        required_env=set(manifest.required_env),
        injected_env={},
    )
    compose = generate_compose(stack)

    top_networks = compose["networks"]  # type: ignore[index]
    # Only the default bridge network — no external entries
    for net_name, net_cfg in top_networks.items():  # type: ignore[union-attr]
        assert net_cfg.get("external") is not True, (
            f"Network '{net_name}' incorrectly marked external for a service "
            "that declares no extra_networks"
        )


@pytest.mark.integration
def test_generate_compose_default_network_not_overwritten_as_external() -> None:
    """omnibase-infra-network in extra_networks must not overwrite the bridge definition."""
    from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
    from omnibase_infra.docker.catalog.manifest_schema import CatalogManifest
    from omnibase_infra.docker.catalog.resolver import ResolvedStack

    manifest = CatalogManifest(
        name="test-svc",
        description="",
        image="test:latest",
        layer=EnumInfraLayer.RUNTIME,
        required_env=[],
        hardcoded_env={},
        operational_defaults={},
        ports=None,
        healthcheck=None,
        volumes=[],
        tmpfs=[],
        depends_on=[],
        extra_networks=["omnibase-infra-network"],  # reserved name in extra_networks
    )
    stack = ResolvedStack(
        manifests={"test-svc": manifest},
        required_env=set(),
        injected_env={},
    )
    compose = generate_compose(stack)

    top_networks = compose["networks"]  # type: ignore[index]
    default_net = top_networks["omnibase-infra-network"]
    assert default_net.get("driver") == "bridge", (
        "Default network must remain a bridge, not be overwritten as external"
    )
    assert "external" not in default_net
