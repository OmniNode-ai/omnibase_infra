# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for NodeContractResolverBridge.

Ticket: OMN-2756 — Phase 2: Transitional node-shaped HTTP bridge for contract.resolve

Verifies:
1. Node is properly declarative (extends NodeEffect)
2. load_contract_resolver_bridge_config() returns expected keys from contract.yaml
3. RegistryInfraContractResolverBridge metadata matches contract.yaml
4. FastAPI app creates successfully with cors_origins parameter
5. GET /health returns 200 with expected body
6. POST /api/nodes/contract.resolve returns 200 with resolved output
7. POST /api/nodes/contract.resolve returns 422 on invalid input
8. Resolved hash is deterministic for the same input
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_contract_resolver_bridge import (
    NodeContractResolverBridge,
    RegistryInfraContractResolverBridge,
    load_contract_resolver_bridge_config,
)

# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadContractResolverBridgeConfig:
    """Tests for contract-driven configuration via load_contract_resolver_bridge_config."""

    def setup_method(self) -> None:
        """Clear lru_cache before each test."""
        load_contract_resolver_bridge_config.cache_clear()

    def test_returns_dict(self) -> None:
        """load_contract_resolver_bridge_config() must return a non-empty dict."""
        cfg = load_contract_resolver_bridge_config()
        assert isinstance(cfg, dict)
        assert len(cfg) > 0, "contract.yaml config section must not be empty"

    def test_has_port(self) -> None:
        """contract.yaml must define config.port."""
        cfg = load_contract_resolver_bridge_config()
        assert "port" in cfg
        assert int(cfg["port"]) == 8091

    def test_has_host(self) -> None:
        """contract.yaml must define config.host."""
        cfg = load_contract_resolver_bridge_config()
        assert "host" in cfg
        assert isinstance(cfg["host"], str)

    def test_has_request_timeout(self) -> None:
        """contract.yaml must define config.request_timeout_seconds."""
        cfg = load_contract_resolver_bridge_config()
        assert "request_timeout_seconds" in cfg
        assert int(cfg["request_timeout_seconds"]) > 0

    def test_has_emit_kafka_events(self) -> None:
        """contract.yaml must define config.emit_kafka_events."""
        cfg = load_contract_resolver_bridge_config()
        assert "emit_kafka_events" in cfg
        assert isinstance(cfg["emit_kafka_events"], bool)

    def test_is_cached(self) -> None:
        """load_contract_resolver_bridge_config() must return the same object (lru_cache)."""
        cfg1 = load_contract_resolver_bridge_config()
        cfg2 = load_contract_resolver_bridge_config()
        assert cfg1 is cfg2


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegistryInfraContractResolverBridge:
    """Tests for the infrastructure registry factory."""

    def test_get_node_type(self) -> None:
        """Node type must be EFFECT."""
        assert RegistryInfraContractResolverBridge.get_node_type() == "EFFECT"

    def test_get_node_name(self) -> None:
        """Node name must match contract.yaml."""
        assert (
            RegistryInfraContractResolverBridge.get_node_name()
            == "node_contract_resolver_bridge"
        )

    def test_get_required_protocols_is_empty(self) -> None:
        """All deps are optional; required list must be empty."""
        assert RegistryInfraContractResolverBridge.get_required_protocols() == []

    def test_get_optional_protocols_includes_event_bus(self) -> None:
        """Optional protocols must include ProtocolEventBus."""
        protocols = RegistryInfraContractResolverBridge.get_optional_protocols()
        assert "ProtocolEventBus" in protocols

    def test_get_capabilities_includes_contract_resolve(self) -> None:
        """Capabilities must include contract.resolve."""
        caps = RegistryInfraContractResolverBridge.get_capabilities()
        assert "contract.resolve" in caps
        assert "contract.health" in caps

    def test_get_supported_operations(self) -> None:
        """Supported operations must match contract.yaml io_operations."""
        ops = RegistryInfraContractResolverBridge.get_supported_operations()
        assert "contract_resolve" in ops
        assert "health_check" in ops

    def test_create_returns_node_instance(self) -> None:
        """create() must return a NodeContractResolverBridge instance."""
        from unittest.mock import MagicMock

        container = MagicMock()
        node = RegistryInfraContractResolverBridge.create(container)
        assert isinstance(node, NodeContractResolverBridge)


# ---------------------------------------------------------------------------
# Node class tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNodeContractResolverBridgeDeclarative:
    """Verify NodeContractResolverBridge is purely declarative."""

    def test_extends_node_effect(self) -> None:
        """NodeContractResolverBridge must extend NodeEffect."""
        from omnibase_core.nodes.node_effect import NodeEffect

        assert issubclass(NodeContractResolverBridge, NodeEffect)

    def test_no_custom_methods(self) -> None:
        """NodeContractResolverBridge must not define custom business logic methods."""
        from omnibase_core.nodes.node_effect import NodeEffect

        # Collect methods defined directly on the bridge (not inherited from NodeEffect)
        bridge_methods = {
            name
            for name, _ in vars(NodeContractResolverBridge).items()
            if not name.startswith("__")
        }
        effect_methods = {
            name for name, _ in vars(NodeEffect).items() if not name.startswith("__")
        }
        custom_methods = bridge_methods - effect_methods
        assert custom_methods == set(), (
            f"NodeContractResolverBridge must be declarative — "
            f"found unexpected custom methods: {custom_methods}"
        )


# ---------------------------------------------------------------------------
# FastAPI application tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContractResolverBridgeApp:
    """Tests for the FastAPI application factory."""

    def _make_app(self) -> object:
        """Create a test app with fixed CORS origins."""
        from unittest.mock import MagicMock

        from omnibase_infra.services.contract_resolver.main import create_app

        container = MagicMock()
        return create_app(
            container=container,
            cors_origins=["http://localhost:3000"],
        )

    def test_create_app_succeeds(self) -> None:
        """create_app() must return a FastAPI instance."""
        from fastapi import FastAPI

        app = self._make_app()
        assert isinstance(app, FastAPI)

    def test_create_app_missing_cors_raises(self) -> None:
        """create_app() must raise ProtocolConfigurationError if CORS not configured."""
        import os
        from unittest.mock import MagicMock, patch

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.services.contract_resolver.main import create_app

        container = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            # Remove CORS_ORIGINS if present
            env = {k: v for k, v in os.environ.items() if k != "CORS_ORIGINS"}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ProtocolConfigurationError):
                    create_app(container=container)

    def test_health_endpoint_returns_200(self) -> None:
        """GET /health must return 200 with status=ok."""
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "node_contract_resolver_bridge"
        assert data["version"] == "1.0.0"

    def test_contract_resolve_invalid_body_returns_422(self) -> None:
        """POST /api/nodes/contract.resolve with invalid body must return 422."""
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            response = client.post(
                "/api/nodes/contract.resolve",
                json={"invalid_field": "should_be_rejected"},
            )
        assert response.status_code == 422

    def test_contract_resolve_empty_patches_returns_200(self) -> None:
        """POST /api/nodes/contract.resolve with empty patches must return 200."""
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            response = client.post(
                "/api/nodes/contract.resolve",
                json={
                    "base_profile_ref": {
                        "profile": "effect_idempotent",
                        "version": "1.0.0",
                    },
                    "patches": [],
                    "options": {
                        "include_diff": False,
                        "include_overlay_refs": True,
                        "normalize_for_hash": True,
                    },
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "resolved_hash" in data
        assert len(data["resolved_hash"]) == 64, "SHA-256 hex digest must be 64 chars"
        assert "resolved_contract" in data
        assert "overlay_refs" in data
        assert data["overlay_refs"] == []  # no patches applied

    def test_contract_resolve_hash_is_deterministic(self) -> None:
        """Same input must produce identical resolved_hash on repeated calls."""
        from fastapi.testclient import TestClient

        app = self._make_app()
        payload = {
            "base_profile_ref": {
                "profile": "effect_idempotent",
                "version": "1.0.0",
            },
            "patches": [],
            "options": {
                "include_diff": False,
                "include_overlay_refs": False,
                "normalize_for_hash": True,
            },
        }

        with TestClient(app) as client:
            resp1 = client.post("/api/nodes/contract.resolve", json=payload)
            resp2 = client.post("/api/nodes/contract.resolve", json=payload)

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["resolved_hash"] == resp2.json()["resolved_hash"]

    def test_root_endpoint_returns_service_info(self) -> None:
        """GET / must return service info dict."""
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "resolve" in data
        assert data["resolve"] == "/api/nodes/contract.resolve"

    def test_openapi_schema_available(self) -> None:
        """GET /openapi.json must return a valid OpenAPI schema."""
        from fastapi.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "/api/nodes/contract.resolve" in schema.get("paths", {})
        assert "/health" in schema.get("paths", {})
