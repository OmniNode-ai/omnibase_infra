# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for NodeRegistryApiEffect contract-driven configuration.

Ticket: OMN-1441 — Refactor Registry API as Contract-Driven ONEX Node

Verifies:
1. load_registry_api_config() returns expected keys from contract.yaml
2. ServiceRegistryDiscovery reads MAX_NODE_TYPE_FILTER_FETCH from the contract
3. ServiceRegistryDiscovery reads DEFAULT_WIDGET_MAPPING_PATH from the contract
4. Backward-compatible constants remain importable from service module
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_registry_api_effect import (
    RegistryInfraRegistryApiEffect,
    load_registry_api_config,
)


@pytest.mark.unit
class TestLoadRegistryApiConfig:
    """Tests for contract-driven configuration via load_registry_api_config."""

    def setup_method(self) -> None:
        """Clear lru_cache before each test to ensure fresh contract reads."""
        load_registry_api_config.cache_clear()

    def test_get_config_returns_dict(self) -> None:
        """load_registry_api_config() must return a non-empty dict."""
        cfg = load_registry_api_config()
        assert isinstance(cfg, dict)
        assert len(cfg) > 0, "contract.yaml config section must not be empty"

    def test_get_config_has_max_node_type_filter_fetch(self) -> None:
        """contract.yaml must define max_node_type_filter_fetch."""
        cfg = load_registry_api_config()
        assert "max_node_type_filter_fetch" in cfg, (
            "contract.yaml missing config.max_node_type_filter_fetch"
        )
        assert int(cfg["max_node_type_filter_fetch"]) > 0

    def test_get_config_has_default_widget_mapping_path(self) -> None:
        """contract.yaml must define default_widget_mapping_path."""
        cfg = load_registry_api_config()
        assert "default_widget_mapping_path" in cfg, (
            "contract.yaml missing config.default_widget_mapping_path"
        )
        assert isinstance(cfg["default_widget_mapping_path"], str)
        assert cfg["default_widget_mapping_path"]  # non-empty

    def test_get_config_has_pagination_defaults(self) -> None:
        """contract.yaml must define pagination defaults."""
        cfg = load_registry_api_config()
        assert "default_page_limit" in cfg
        assert "max_page_limit" in cfg
        assert int(cfg["default_page_limit"]) > 0
        assert int(cfg["max_page_limit"]) >= int(cfg["default_page_limit"])

    def test_get_config_is_cached(self) -> None:
        """load_registry_api_config() must return the same object on repeated calls (lru_cache)."""
        cfg1 = load_registry_api_config()
        cfg2 = load_registry_api_config()
        assert cfg1 is cfg2, "load_registry_api_config() should return cached result"

    def test_max_node_type_filter_fetch_default_value(self) -> None:
        """max_node_type_filter_fetch should be 10000 (matches former hardcoded constant)."""
        cfg = load_registry_api_config()
        assert int(cfg["max_node_type_filter_fetch"]) == 10000

    def test_default_widget_mapping_path_points_to_configs(self) -> None:
        """default_widget_mapping_path should reference the configs directory."""
        cfg = load_registry_api_config()
        path_str: str = str(cfg["default_widget_mapping_path"])
        assert "configs" in path_str or "widget_mapping" in path_str


@pytest.mark.unit
class TestServiceModuleConstants:
    """Verify backward-compatible constants in service.py use contract values."""

    def test_max_node_type_filter_fetch_importable(self) -> None:
        """MAX_NODE_TYPE_FILTER_FETCH must be importable from service module."""
        from omnibase_infra.services.registry_api.service import (
            MAX_NODE_TYPE_FILTER_FETCH,
        )

        assert isinstance(MAX_NODE_TYPE_FILTER_FETCH, int)
        assert MAX_NODE_TYPE_FILTER_FETCH == 10000

    def test_default_widget_mapping_path_importable(self) -> None:
        """DEFAULT_WIDGET_MAPPING_PATH must be importable and point to widget_mapping.yaml."""
        from pathlib import Path

        from omnibase_infra.services.registry_api.service import (
            DEFAULT_WIDGET_MAPPING_PATH,
        )

        assert isinstance(DEFAULT_WIDGET_MAPPING_PATH, Path)
        assert DEFAULT_WIDGET_MAPPING_PATH.name == "widget_mapping.yaml"

    def test_default_widget_mapping_path_exists(self) -> None:
        """The widget_mapping.yaml file must exist at the resolved path."""
        from omnibase_infra.services.registry_api.service import (
            DEFAULT_WIDGET_MAPPING_PATH,
        )

        assert DEFAULT_WIDGET_MAPPING_PATH.exists(), (
            f"widget_mapping.yaml not found at {DEFAULT_WIDGET_MAPPING_PATH}"
        )


@pytest.mark.unit
class TestRegistryInfraRegistryApiEffect:
    """Tests for the infrastructure registry factory."""

    def test_get_node_type(self) -> None:
        """Node type must be EFFECT."""
        assert RegistryInfraRegistryApiEffect.get_node_type() == "EFFECT"

    def test_get_node_name(self) -> None:
        """Node name must match contract.yaml."""
        assert (
            RegistryInfraRegistryApiEffect.get_node_name() == "node_registry_api_effect"
        )

    def test_get_required_protocols_is_empty(self) -> None:
        """All deps are optional (partial-success semantics); required list is empty."""
        assert RegistryInfraRegistryApiEffect.get_required_protocols() == []

    def test_get_optional_protocols_non_empty(self) -> None:
        """Optional protocols list must be non-empty."""
        protocols = RegistryInfraRegistryApiEffect.get_optional_protocols()
        assert len(protocols) > 0

    def test_get_capabilities_includes_discovery(self) -> None:
        """Capabilities must include registry.discovery."""
        caps = RegistryInfraRegistryApiEffect.get_capabilities()
        assert "registry.discovery" in caps

    def test_get_supported_operations_complete(self) -> None:
        """All 10 operations declared in contract.yaml must be listed."""
        ops = RegistryInfraRegistryApiEffect.get_supported_operations()
        expected = {
            "list_nodes",
            "get_node",
            "list_instances",
            "get_widget_mapping",
            "get_discovery",
            "get_health",
            "list_contracts",
            "get_contract",
            "list_topics",
            "get_topic",
        }
        assert set(ops) == expected
