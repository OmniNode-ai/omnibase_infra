# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the catalog resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.resolver import CatalogResolver

CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")


@pytest.mark.unit
def test_resolver_selects_runtime_includes_core_and_valkey() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    service_names = resolved.service_names
    assert "postgres" in service_names  # from core (included by runtime)
    assert "redpanda" in service_names  # from core
    assert "valkey" in service_names  # explicit in runtime
    assert "omninode-runtime" in service_names


@pytest.mark.unit
def test_resolver_collects_all_required_env() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    assert "POSTGRES_PASSWORD" in resolved.required_env
    assert "VALKEY_PASSWORD" in resolved.required_env


@pytest.mark.unit
def test_resolver_injects_bundle_env_for_memgraph() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime", "memgraph"])
    assert resolved.injected_env["OMNIMEMORY_ENABLED"] == "true"
    assert "OMNIMEMORY_DB_URL" in resolved.required_env


@pytest.mark.unit
def test_resolver_does_not_leak_feature_env_when_bundle_unselected() -> None:
    """Feature vars must NOT appear when their bundle is not selected."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["core"])
    feature_vars = {
        "OMNIMEMORY_ENABLED",
        "OMNIMEMORY_DB_URL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_TRACES_EXPORTER",
        # INFISICAL_ADDR and INFISICAL_CLIENT_ID moved to core bundle
        # as part of Infisical-first config (OMN-5831)
    }
    leaked = feature_vars & set(resolved.injected_env.keys())
    assert not leaked, f"Feature vars leaked into core bundle: {leaked}"
