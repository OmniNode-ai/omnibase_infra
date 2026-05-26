# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for /v1/introspection/manifest (OMN-11198).

The unit suite covers the in-process handler with a mocked runtime; this
integration test boots a real aiohttp web.Application against the
ServiceHealth router with a real ModelAutoWiringManifest attached and
exercises the HTTP surface end-to-end. The goal is to prove that attach
+ route registration + JSON serialization wire together without crash
and that the 503 -> 200 transition matches the contract described in
the OMN-11198 ticket.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.services.health_checker import ServiceHealth

pytestmark = pytest.mark.integration


def _make_manifest(contract_count: int = 2) -> ModelAutoWiringManifest:
    contracts = tuple(
        ModelDiscoveredContract(
            name=f"integration_contract_{i}",
            node_type="EFFECT_GENERIC",
            contract_version=ModelContractVersion(major=1, minor=0, patch=0),
            contract_path=Path(f"/fake/integration/{i}/contract.yaml"),
            entry_point_name=f"integration_contract_{i}",
            package_name=f"integration_pkg_{i}",
        )
        for i in range(contract_count)
    )
    return ModelAutoWiringManifest(contracts=contracts)


def _build_service(manifest: ModelAutoWiringManifest | None) -> ServiceHealth:
    runtime = MagicMock()
    service = ServiceHealth(runtime=runtime, port=0)
    if manifest is not None:
        service.attach_manifest(manifest)
    return service


def _build_app(service: ServiceHealth) -> web.Application:
    app = web.Application()
    app.router.add_get(
        "/v1/introspection/manifest",
        service._handle_introspection_manifest,
    )
    return app


@pytest.mark.asyncio
async def test_introspection_returns_503_before_attach_manifest() -> None:
    service = _build_service(manifest=None)
    app = _build_app(service)
    async with TestServer(app) as server, TestClient(server) as client:
        response = await client.get("/v1/introspection/manifest")
        assert response.status == 503
        body = json.loads(await response.text())
        assert body["status"] == "starting"


@pytest.mark.asyncio
async def test_introspection_returns_200_after_attach_manifest() -> None:
    service = _build_service(manifest=_make_manifest(contract_count=2))
    app = _build_app(service)
    async with TestServer(app) as server, TestClient(server) as client:
        response = await client.get("/v1/introspection/manifest")
        assert response.status == 200
        body = json.loads(await response.text())
        names = {c["name"] for c in body["contracts"]}
        assert names == {"integration_contract_0", "integration_contract_1"}


@pytest.mark.asyncio
async def test_introspection_serves_attached_manifest_contents() -> None:
    """The JSON body must exactly mirror the attached manifest."""
    manifest = _make_manifest(contract_count=3)
    service = _build_service(manifest=manifest)
    app = _build_app(service)
    async with TestServer(app) as server, TestClient(server) as client:
        response = await client.get("/v1/introspection/manifest")
        assert response.status == 200
        body = json.loads(await response.text())
        expected = json.loads(manifest.model_dump_json())
        assert body == expected
