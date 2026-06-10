# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the canary runtime bundle (OMN-10729).

Verifies that the canary bundle is well-formed: it resolves to exactly
runtime-canary (plus core deps), declares RUNTIME_PROFILE=canary, and
binds to the expected port :8088.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.resolver import CatalogResolver

CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")
BUNDLES_FILE = (
    Path(__file__).resolve().parents[3] / "docker" / "catalog" / "bundles.yaml"
)
SERVICES_DIR = Path(__file__).resolve().parents[3] / "docker" / "catalog" / "services"


def _load_manifest(name: str) -> dict[str, object]:
    path = SERVICES_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _load_bundles() -> dict[str, dict[str, object]]:
    with open(BUNDLES_FILE) as f:
        return yaml.safe_load(f)


@pytest.mark.unit
def test_canary_bundle_exists() -> None:
    bundles = _load_bundles()
    assert "canary" in bundles, "canary bundle missing from bundles.yaml"


@pytest.mark.unit
def test_canary_bundle_contains_runtime_canary_service() -> None:
    bundles = _load_bundles()
    services = bundles["canary"]["services"]
    assert "runtime-canary" in services


@pytest.mark.unit
def test_canary_bundle_includes_core() -> None:
    bundles = _load_bundles()
    includes = bundles["canary"].get("includes", [])
    assert "core" in includes, (
        "canary bundle must include core for postgres/redpanda/valkey"
    )


@pytest.mark.unit
def test_runtime_canary_manifest_exists() -> None:
    path = SERVICES_DIR / "runtime-canary.yaml"
    assert path.exists(), "docker/catalog/services/runtime-canary.yaml missing"


@pytest.mark.unit
def test_runtime_canary_has_correct_runtime_profile() -> None:
    manifest = _load_manifest("runtime-canary")
    hardcoded = manifest.get("hardcoded_env", {})
    assert hardcoded.get("RUNTIME_PROFILE") == "canary", (
        f"expected RUNTIME_PROFILE=canary, got {hardcoded.get('RUNTIME_PROFILE')}"
    )


@pytest.mark.unit
def test_runtime_canary_uses_port_8088() -> None:
    manifest = _load_manifest("runtime-canary")
    ports = manifest.get("ports", {})
    assert ports.get("external") == 8088, (
        f"expected external port 8088, got {ports.get('external')}"
    )


@pytest.mark.unit
def test_runtime_canary_has_unique_group_id() -> None:
    manifest = _load_manifest("runtime-canary")
    defaults = manifest.get("operational_defaults", {})
    group_id = defaults.get("ONEX_GROUP_ID")
    assert group_id == "onex-runtime-canary", (
        f"expected ONEX_GROUP_ID=onex-runtime-canary, got {group_id}"
    )
    main_manifest = _load_manifest("omninode-runtime")
    main_defaults = main_manifest.get("operational_defaults", {})
    assert group_id != main_defaults.get("ONEX_GROUP_ID"), (
        "canary and main runtime must have distinct ONEX_GROUP_IDs"
    )


@pytest.mark.unit
def test_runtime_canary_has_distinct_container_name() -> None:
    manifest = _load_manifest("runtime-canary")
    assert manifest.get("container_name") == "omninode-runtime-canary"


@pytest.mark.unit
def test_canary_bundle_resolves_without_error() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["canary"])
    assert "runtime-canary" in resolved.service_names


@pytest.mark.unit
def test_canary_bundle_required_env_subset_of_core_baseline() -> None:
    """canary bundle must not demand env vars beyond the standard runtime baseline."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["canary"])
    _EXPECTED_CANARY_ENV = frozenset(
        {
            "POSTGRES_PASSWORD",
            "VALKEY_PASSWORD",
            "KEYCLOAK_ADMIN_CLIENT_SECRET",
            "ONEX_SERVICE_CLIENT_SECRET",
            "ONEX_REGISTRATION_AUTO_ACK",
            "LLM_CODER_URL",
            "LLM_CODER_FAST_URL",
            "LLM_EMBEDDING_URL",
            "LLM_DEEPSEEK_R1_URL",
            "LLM_ENDPOINT_CIDR_ALLOWLIST",
            "LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST",
            "LOCAL_LLM_SHARED_SECRET",
            "OMNIDASH_ANALYTICS_DB_URL",
            # core bundle Infisical vars (transitive via includes: [core])
            "INFISICAL_CLIENT_ID",
            "INFISICAL_CLIENT_SECRET",
            "INFISICAL_PROJECT_ID",
            "INFISICAL_ENCRYPTION_KEY",
            "INFISICAL_AUTH_SECRET",
            "INFISICAL_REDIS_URL",
        }
    )
    unexpected = resolved.required_env - _EXPECTED_CANARY_ENV
    assert not unexpected, f"canary bundle requires unexpected env vars: {unexpected}"
