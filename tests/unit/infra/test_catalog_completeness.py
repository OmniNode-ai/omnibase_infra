# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for catalog completeness — every compose entry has a manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer

CATALOG_DIR = Path(__file__).resolve().parents[3] / "docker" / "catalog" / "services"
COMPOSE_FILE = (
    Path(__file__).resolve().parents[3] / "docker" / "docker-compose.infra.yml"
)
BUNDLES_FILE = (
    Path(__file__).resolve().parents[3] / "docker" / "catalog" / "bundles.yaml"
)

# The compose uses 'omninode-contract-resolver' but the manifest is named
# 'contract-resolver' for brevity. This mapping handles that.
COMPOSE_TO_MANIFEST_NAME = {
    "omninode-contract-resolver": "omninode-contract-resolver",
}


@pytest.mark.unit
def test_every_compose_entry_has_a_manifest() -> None:
    """Every entry in docker-compose.infra.yml must have a catalog manifest."""
    content = COMPOSE_FILE.read_text().replace("!!merge ", "")
    compose = yaml.safe_load(content)
    compose_services = set(compose.get("services", {}).keys())

    manifest_names: set[str] = set()
    for manifest_file in CATALOG_DIR.glob("*.yaml"):
        with open(manifest_file) as f:
            manifest = yaml.safe_load(f)
        manifest_names.add(manifest["name"])

    missing = compose_services - manifest_names
    assert not missing, f"Entries in compose but missing manifests: {missing}"


@pytest.mark.unit
def test_bundles_reference_only_existing_entries() -> None:
    with open(BUNDLES_FILE) as f:
        bundles = yaml.safe_load(f)
    manifest_names: set[str] = set()
    for p in CATALOG_DIR.glob("*.yaml"):
        with open(p) as mf:
            manifest_names.add(yaml.safe_load(mf)["name"])
    for bundle_name, bundle_def in bundles.items():
        for svc in bundle_def.get("services", []):
            assert svc in manifest_names, (
                f"Bundle '{bundle_name}' references unknown entry '{svc}'"
            )


@pytest.mark.unit
def test_no_entry_has_empty_default_env() -> None:
    """No manifest should declare an env var with empty-string default."""
    for manifest_file in CATALOG_DIR.glob("*.yaml"):
        with open(manifest_file) as f:
            manifest = yaml.safe_load(f)
        for var in manifest.get("required_env", []):
            assert var, f"{manifest['name']}: empty string in required_env"
        for var, val in manifest.get("hardcoded_env", {}).items():
            assert val != "", f"{manifest['name']}: {var} has empty hardcoded value"


@pytest.mark.unit
def test_all_manifests_have_valid_layer() -> None:
    """Every manifest must use a constrained layer value."""
    valid_layers = {e.value for e in EnumInfraLayer}
    for manifest_file in CATALOG_DIR.glob("*.yaml"):
        with open(manifest_file) as f:
            manifest = yaml.safe_load(f)
        assert manifest.get("layer") in valid_layers, (
            f"{manifest['name']}: layer '{manifest.get('layer')}' not in {valid_layers}"
        )


@pytest.mark.unit
def test_all_manifests_have_description() -> None:
    """Every manifest must have a non-empty description."""
    for manifest_file in CATALOG_DIR.glob("*.yaml"):
        with open(manifest_file) as f:
            manifest = yaml.safe_load(f)
        desc = manifest.get("description", "")
        assert desc and len(desc) > 0, f"{manifest['name']}: missing description"
