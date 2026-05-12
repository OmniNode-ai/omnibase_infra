# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for POST /skill endpoint on ServiceHealth (OMN-10860).

Boots a real ServiceHealth HTTP server with a mocked RuntimeHostProcess and
verifies that the /skill endpoint correctly routes requests through the
dispatch_local_ingress_request path end-to-end over a real HTTP connection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import aiohttp
import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.models.model_local_runtime_ingress_error import (
    ModelLocalRuntimeIngressError,
)
from omnibase_infra.runtime.models.model_local_runtime_ingress_response import (
    ModelLocalRuntimeIngressResponse,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from omnibase_infra.services.service_health import ServiceHealth
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.integration

_CORR_ID = UUID("aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb")

_SKILL_TEST_CONFIG: dict[str, object] = {
    "service_name": "skill-endpoint-test",
    "node_name": "test-node",
    "env": "test",
    "version": "v1",
}


def _ok_ingress_response(
    command_name: str = "test_command",
) -> ModelLocalRuntimeIngressResponse:
    from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
        ModelDispatchBusTerminalResult,
    )

    return ModelLocalRuntimeIngressResponse(
        ok=True,
        command_name=command_name,
        correlation_id=_CORR_ID,
        dispatch_result=ModelDispatchBusTerminalResult(
            correlation_id=_CORR_ID,
            status="completed",
            payload={"result": "ok"},
            error_message=None,
        ),
        output_payloads=[{"result": "ok"}],
    )


def _error_ingress_response(
    command_name: str = "bad_command",
) -> ModelLocalRuntimeIngressResponse:
    return ModelLocalRuntimeIngressResponse(
        ok=False,
        command_name=command_name,
        error=ModelLocalRuntimeIngressError(
            code="unknown_command", message="Unknown command"
        ),  # type: ignore[arg-type]
    )


class TestServiceHealthSkillEndpointIntegration:
    """Integration tests for the POST /skill HTTP endpoint."""

    async def _boot_health_server(
        self,
        runtime: RuntimeHostProcess,
    ) -> tuple[ServiceHealth, int]:
        """Start a ServiceHealth server and return it + its bound port."""
        health_server = ServiceHealth(runtime=runtime, port=0, version="test-1.0.0")
        await health_server.start()
        site = health_server._site
        assert site is not None
        internal_server = site._server
        assert internal_server is not None
        sockets = getattr(internal_server, "sockets", None)
        assert sockets and len(sockets) > 0
        port: int = sockets[0].getsockname()[1]
        return health_server, port

    @pytest.mark.asyncio
    async def test_skill_endpoint_dispatches_and_returns_ok(self) -> None:
        """POST /skill with a valid request reaches dispatch and returns ok=True."""
        event_bus = EventBusInmemory()
        runtime = RuntimeHostProcess(event_bus=event_bus, config=_SKILL_TEST_CONFIG)
        ingress_ok = _ok_ingress_response("test_command")

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            seed_mock_handlers(runtime)
            await runtime.start()

            runtime.dispatch_local_ingress_request = AsyncMock(return_value=ingress_ok)  # type: ignore[method-assign]

            health_server: ServiceHealth | None = None
            try:
                health_server, port = await self._boot_health_server(runtime)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/skill",
                        json={
                            "command_name": "test_command",
                            "payload": {"dry_run": True},
                            "correlation_id": str(_CORR_ID),
                            "timeout_ms": 5000,
                        },
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()

                assert data["ok"] is True
                assert data["command_name"] == "test_command"
                runtime.dispatch_local_ingress_request.assert_awaited_once()
            finally:
                if health_server is not None:
                    await health_server.stop()
                await runtime.stop()

    @pytest.mark.asyncio
    async def test_skill_endpoint_returns_400_on_invalid_json(self) -> None:
        """POST /skill with malformed JSON returns 400 with validation_error."""
        event_bus = EventBusInmemory()
        runtime = RuntimeHostProcess(event_bus=event_bus, config=_SKILL_TEST_CONFIG)

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            seed_mock_handlers(runtime)
            await runtime.start()

            health_server: ServiceHealth | None = None
            try:
                health_server, port = await self._boot_health_server(runtime)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/skill",
                        data=b"not-json{{{",
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        assert resp.status == 400
                        data = await resp.json()

                assert data["ok"] is False
                assert data["error"]["code"] == "validation_error"
            finally:
                if health_server is not None:
                    await health_server.stop()
                await runtime.stop()

    @pytest.mark.asyncio
    async def test_skill_endpoint_propagates_correlation_id_from_header(self) -> None:
        """X-Correlation-ID header is propagated into the ingress request."""
        event_bus = EventBusInmemory()
        runtime = RuntimeHostProcess(event_bus=event_bus, config=_SKILL_TEST_CONFIG)
        corr_id = uuid4()
        ingress_ok = _ok_ingress_response("session_orchestrator")

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            seed_mock_handlers(runtime)
            await runtime.start()

            runtime.dispatch_local_ingress_request = AsyncMock(return_value=ingress_ok)  # type: ignore[method-assign]

            health_server: ServiceHealth | None = None
            try:
                health_server, port = await self._boot_health_server(runtime)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/skill",
                        json={"command_name": "session_orchestrator", "payload": {}},
                        headers={"X-Correlation-ID": str(corr_id)},
                    ) as resp:
                        assert resp.status == 200

                call_args = runtime.dispatch_local_ingress_request.call_args
                assert call_args is not None
                ingress_req = call_args[0][0]
                assert ingress_req.correlation_id == corr_id
            finally:
                if health_server is not None:
                    await health_server.stop()
                await runtime.stop()
