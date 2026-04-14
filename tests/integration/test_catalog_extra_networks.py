# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for catalog extra_networks support (OMN-8713).

Verifies that services declaring extra_networks in their manifest YAML:
  - Join those networks in the generated compose service definition
  - Cause the extra network to appear as external in the top-level networks block
  - Do not pollute services that lack extra_networks
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver

REPO_ROOT = Path(__file__).parent.parent.parent
CATALOG_DIR = str(REPO_ROOT / "docker" / "catalog")


@pytest.mark.integration
def test_runtime_effects_joins_omnimemory_network() -> None:
    """runtime-effects manifest declares extra_networks: [omnimemory-network].

    After generate_compose the service must appear in both omnibase-infra-network
    and omnimemory-network.
    """
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    # Resolve a bundle that includes runtime-effects; fall back to direct manifest
    # lookup if no bundle wraps it.
    manifest = resolver._manifests.get("runtime-effects")
    if manifest is None:
        pytest.skip("runtime-effects manifest not found in catalog — skipping")

    assert "omnimemory-network" in manifest.extra_networks, (
        "runtime-effects manifest must declare extra_networks: [omnimemory-network]"
    )


@pytest.mark.integration
def test_generate_compose_extra_network_in_service_networks() -> None:
    """generate_compose must include extra_networks in service's networks list."""
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
    assert "omnibase-infra-network" in svc_networks
    assert "omnimemory-network" in svc_networks


@pytest.mark.integration
def test_generate_compose_extra_network_marked_external() -> None:
    """Extra networks must appear in the top-level networks block as external."""
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
    assert "omnimemory-network" in top_networks
    assert top_networks["omnimemory-network"] == {
        "name": "omnimemory-network",
        "external": True,
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
