# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for GET /v1/introspection/manifest endpoint (OMN-11198)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.services.health_checker import ServiceHealth


def _make_manifest(contract_count: int = 2) -> ModelAutoWiringManifest:
    """Build a minimal ModelAutoWiringManifest for testing."""
    from pathlib import Path

    from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
        ModelContractVersion,
    )
    from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
        ModelDiscoveredContract,
    )

    contracts = tuple(
        ModelDiscoveredContract(
            name=f"test_contract_{i}",
            node_type="EFFECT_GENERIC",
            contract_version=ModelContractVersion(major=1, minor=0, patch=0),
            contract_path=Path(f"/fake/path/{i}/contract.yaml"),
            entry_point_name=f"test_contract_{i}",
            package_name=f"pkg_{i}",
        )
        for i in range(contract_count)
    )
    return ModelAutoWiringManifest(contracts=contracts)


@pytest.mark.unit
class TestIntrospectionManifestEndpoint:
    """Tests for GET /v1/introspection/manifest on ServiceHealth."""

    @pytest.mark.asyncio
    async def test_returns_503_when_manifest_not_attached(self) -> None:
        """Returns 503 if attach_manifest() was never called (startup not complete)."""
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime, port=0)

        mock_request = MagicMock()
        response = await server._handle_introspection_manifest(mock_request)

        assert response.status == 503
        body = json.loads(response.text)
        assert body["status"] == "starting"

    @pytest.mark.asyncio
    async def test_returns_200_with_manifest_json_when_attached(self) -> None:
        """Returns 200 with full manifest JSON after attach_manifest() is called."""
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime, port=0)

        manifest = _make_manifest(contract_count=2)
        server.attach_manifest(manifest)

        mock_request = MagicMock()
        response = await server._handle_introspection_manifest(mock_request)

        assert response.status == 200
        assert response.content_type == "application/json"
        body = json.loads(response.text)
        assert "contracts" in body
        assert len(body["contracts"]) == 2

    @pytest.mark.asyncio
    async def test_manifest_json_contains_contract_names(self) -> None:
        """Returned JSON includes the contract names from the attached manifest."""
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime, port=0)

        manifest = _make_manifest(contract_count=1)
        server.attach_manifest(manifest)

        mock_request = MagicMock()
        response = await server._handle_introspection_manifest(mock_request)

        body = json.loads(response.text)
        contract_names = [c["name"] for c in body["contracts"]]
        assert "test_contract_0" in contract_names

    @pytest.mark.asyncio
    async def test_attach_manifest_replaces_previous(self) -> None:
        """Calling attach_manifest() a second time replaces the prior manifest."""
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime, port=0)

        server.attach_manifest(_make_manifest(contract_count=1))
        server.attach_manifest(_make_manifest(contract_count=3))

        mock_request = MagicMock()
        response = await server._handle_introspection_manifest(mock_request)

        body = json.loads(response.text)
        assert len(body["contracts"]) == 3

    def test_attach_manifest_stores_reference(self) -> None:
        """attach_manifest() stores the manifest on _manifest attribute."""
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime, port=0)

        assert server._manifest is None
        manifest = _make_manifest()
        server.attach_manifest(manifest)
        assert server._manifest is manifest
