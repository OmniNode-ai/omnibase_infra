# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-8738: Validator tests for compose-migrated service manifests.

Each test asserts that the new manifest YAML parses into a valid CatalogManifest
and resolves cleanly through CatalogResolver. Tests are written TDD-first and
must fail until the corresponding manifest files are added.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.resolver import CatalogResolver, _load_manifest

CATALOG_DIR = Path(__file__).resolve().parents[3] / "docker" / "catalog"
SERVICES_DIR = CATALOG_DIR / "services"

# ── Expected manifests from OMN-8738 migration ──────────────────────────────

OMNIMEMORY_MANIFESTS = [
    "omnimemory-qdrant",
    "omnimemory-memgraph",
    "omnimemory-valkey",
    "omnimemory-kreuzberg",
]

OMNIMARKET_PROJECTION_MANIFESTS = [
    "omnimarket-projection-session-outcome",
    "omnimarket-projection-llm-cost",
    "omnimarket-projection-savings",
    "omnimarket-projection-delegation",
    "omnimarket-projection-baselines",
    "omnimarket-projection-registration",
]

OMNIDASH_MANIFESTS = [
    "omnidash",
]

ALL_NEW_MANIFESTS = (
    OMNIMEMORY_MANIFESTS + OMNIMARKET_PROJECTION_MANIFESTS + OMNIDASH_MANIFESTS
)


# ── Parse tests ──────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("manifest_name", ALL_NEW_MANIFESTS)
def test_new_manifest_file_exists(manifest_name: str) -> None:
    """Each OMN-8738 manifest YAML file must exist in catalog/services/."""
    manifest_path = SERVICES_DIR / f"{manifest_name}.yaml"
    assert manifest_path.exists(), (
        f"OMN-8738: manifest file missing: {manifest_path}. "
        "Create it from the source compose definition."
    )


@pytest.mark.unit
@pytest.mark.parametrize("manifest_name", ALL_NEW_MANIFESTS)
def test_new_manifest_parses_as_catalog_manifest(manifest_name: str) -> None:
    """Each OMN-8738 manifest must parse into a valid CatalogManifest (no loose dicts)."""
    manifest_path = SERVICES_DIR / f"{manifest_name}.yaml"
    if not manifest_path.exists():
        pytest.skip(f"Manifest not yet created: {manifest_name}")
    manifest = _load_manifest(manifest_path)
    assert manifest.name == manifest_name, (
        f"Manifest 'name' field must equal '{manifest_name}', got '{manifest.name}'"
    )
    assert manifest.description, f"{manifest_name}: description must be non-empty"
    assert manifest.image, f"{manifest_name}: image must be non-empty"
    assert manifest.layer is not None, f"{manifest_name}: layer must be set"


@pytest.mark.unit
@pytest.mark.parametrize("manifest_name", ALL_NEW_MANIFESTS)
def test_new_manifest_yaml_loads_cleanly(manifest_name: str) -> None:
    """Each OMN-8738 manifest YAML must be valid YAML with required top-level keys."""
    manifest_path = SERVICES_DIR / f"{manifest_name}.yaml"
    if not manifest_path.exists():
        pytest.skip(f"Manifest not yet created: {manifest_name}")
    with open(manifest_path) as f:
        raw = yaml.safe_load(f)
    required_keys = {"name", "description", "image", "layer"}
    missing = required_keys - set(raw.keys())
    assert not missing, f"{manifest_name}: missing required keys: {missing}"


# ── Bundle integration tests ─────────────────────────────────────────────────


@pytest.mark.unit
def test_omnimemory_bundle_resolves_all_four_services() -> None:
    """The omnimemory bundle must resolve all 4 migrated services."""
    resolver = CatalogResolver(catalog_dir=str(CATALOG_DIR))
    resolved = resolver.resolve(bundles=["omnimemory"])
    for svc in OMNIMEMORY_MANIFESTS:
        assert svc in resolved.service_names, (
            f"omnimemory bundle missing service '{svc}'"
        )


@pytest.mark.unit
def test_omnimarket_projections_bundle_resolves_all_six_consumers() -> None:
    """The omnimarket-projections bundle must resolve all 6 projection consumers."""
    resolver = CatalogResolver(catalog_dir=str(CATALOG_DIR))
    resolved = resolver.resolve(bundles=["omnimarket-projections"])
    for svc in OMNIMARKET_PROJECTION_MANIFESTS:
        assert svc in resolved.service_names, (
            f"omnimarket-projections bundle missing service '{svc}'"
        )


@pytest.mark.unit
def test_omnidash_bundle_resolves() -> None:
    """The omnidash bundle must resolve the omnidash service."""
    resolver = CatalogResolver(catalog_dir=str(CATALOG_DIR))
    resolved = resolver.resolve(bundles=["omnidash"])
    assert "omnidash" in resolved.service_names
