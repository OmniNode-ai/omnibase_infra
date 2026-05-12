# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the POST /skill endpoint on ServiceHealth (OMN-10860)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from aiohttp import web

from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.runtime.models.model_local_runtime_ingress_error import (
    ModelLocalRuntimeIngressError,
)
from omnibase_infra.runtime.models.model_local_runtime_ingress_response import (
    ModelLocalRuntimeIngressResponse,
)
from omnibase_infra.services.service_health import ServiceHealth

_CORR_ID = UUID("12345678-1234-5678-1234-567812345678")


def _make_dispatch_result() -> ModelDispatchBusTerminalResult:
    return ModelDispatchBusTerminalResult(
        correlation_id=_CORR_ID,
        status="completed",
        payload={"result": "ok"},
        error_message=None,
    )


def _make_ok_ingress_response(
    command_name: str = "test_command",
) -> ModelLocalRuntimeIngressResponse:
    return ModelLocalRuntimeIngressResponse(
        ok=True,
        command_name=command_name,
        correlation_id=_CORR_ID,
        dispatch_result=_make_dispatch_result(),
        output_payloads=[{"result": "ok"}],
    )


def _make_error_ingress_response(
    code: str = "unknown_command",
    message: str = "Unknown command",
    command_name: str = "bad_command",
) -> ModelLocalRuntimeIngressResponse:
    return ModelLocalRuntimeIngressResponse(
        ok=False,
        command_name=command_name,
        error=ModelLocalRuntimeIngressError(code=code, message=message),  # type: ignore[arg-type]
    )


@pytest.fixture
def mock_runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.dispatch_local_ingress_request = AsyncMock(
        return_value=_make_ok_ingress_response()
    )
    return runtime


@pytest.mark.unit
class TestServiceHealthSkillEndpoint:
    """Tests for the POST /skill endpoint."""

    @pytest.mark.asyncio
    async def test_skill_success(self, mock_runtime: MagicMock) -> None:
        """Successful dispatch returns 200 with ok=True."""
        server = ServiceHealth(runtime=mock_runtime, port=0)

        with patch(
            "omnibase_infra.services.service_health.web.Application"
        ) as mock_app_cls:
            with patch(
                "omnibase_infra.services.service_health.web.AppRunner"
            ) as mock_runner_cls:
                with patch(
                    "omnibase_infra.services.service_health.web.TCPSite"
                ) as mock_site_cls:
                    app_instance = MagicMock()
                    routes: list[tuple[str, str, object]] = []

                    def _add_post(path: str, handler: object) -> None:
                        routes.append(("POST", path, handler))

                    app_instance.router = MagicMock()
                    app_instance.router.add_get = MagicMock()
                    app_instance.router.add_post = MagicMock(side_effect=_add_post)
                    mock_app_cls.return_value = app_instance

                    runner_instance = MagicMock()
                    runner_instance.setup = AsyncMock()
                    mock_runner_cls.return_value = runner_instance

                    site_instance = MagicMock()
                    site_instance.start = AsyncMock()
                    mock_site_cls.return_value = site_instance

                    await server.start()

        # POST /skill should be registered
        assert app_instance.router.add_post.called
        call_args = app_instance.router.add_post.call_args_list
        paths = [c.args[0] for c in call_args]
        assert "/skill" in paths

    @pytest.mark.asyncio
    async def test_handle_skill_ok_response(self, mock_runtime: MagicMock) -> None:
        """_handle_skill returns 200 with ok=True for a successful ingress response."""
        server = ServiceHealth(runtime=mock_runtime, port=0)

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.json = AsyncMock(return_value={"command_name": "test_command"})

        response = await server._handle_skill(mock_request)

        assert response.status == 200
        body = json.loads(response.text)
        assert body["ok"] is True
        assert body["command_name"] == "test_command"

    @pytest.mark.asyncio
    async def test_handle_skill_invalid_json(self, mock_runtime: MagicMock) -> None:
        """_handle_skill returns 400 when request body is not valid JSON."""
        server = ServiceHealth(runtime=mock_runtime, port=0)

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.json = AsyncMock(side_effect=ValueError("bad json"))

        response = await server._handle_skill(mock_request)

        assert response.status == 400
        body = json.loads(response.text)
        assert body["ok"] is False
        assert body["error"]["code"] == "validation_error"

    @pytest.mark.asyncio
    async def test_handle_skill_validation_error(self, mock_runtime: MagicMock) -> None:
        """_handle_skill returns 400 when body fails ModelRuntimeSkillRequest validation."""
        server = ServiceHealth(runtime=mock_runtime, port=0)

        mock_request = MagicMock()
        mock_request.headers = {}
        # command_name is required; empty dict should fail
        mock_request.json = AsyncMock(return_value={})

        response = await server._handle_skill(mock_request)

        assert response.status == 400
        body = json.loads(response.text)
        assert body["ok"] is False
        assert body["error"]["code"] == "validation_error"

    @pytest.mark.asyncio
    async def test_handle_skill_unknown_command(self, mock_runtime: MagicMock) -> None:
        """_handle_skill returns 200 with ok=False for unknown_command ingress error."""
        mock_runtime.dispatch_local_ingress_request = AsyncMock(
            return_value=_make_error_ingress_response(
                code="unknown_command", message="Unknown command 'bogus'"
            )
        )
        server = ServiceHealth(runtime=mock_runtime, port=0)

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.json = AsyncMock(return_value={"command_name": "bogus"})

        response = await server._handle_skill(mock_request)

        assert response.status == 200
        body = json.loads(response.text)
        assert body["ok"] is False
        assert body["error"]["code"] == "unknown_command"

    @pytest.mark.asyncio
    async def test_handle_skill_dispatch_exception(
        self, mock_runtime: MagicMock
    ) -> None:
        """_handle_skill returns 500 when dispatch_local_ingress_request raises."""
        mock_runtime.dispatch_local_ingress_request = AsyncMock(
            side_effect=RuntimeError("broker down")
        )
        server = ServiceHealth(runtime=mock_runtime, port=0)

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.json = AsyncMock(return_value={"command_name": "test_command"})

        response = await server._handle_skill(mock_request)

        assert response.status == 500
        body = json.loads(response.text)
        assert body["ok"] is False
        assert body["error"]["code"] == "dispatch_error"

    @pytest.mark.asyncio
    async def test_handle_skill_correlation_id_propagated(
        self, mock_runtime: MagicMock
    ) -> None:
        """Caller-supplied X-Correlation-ID is forwarded into the ingress request."""
        caller_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        server = ServiceHealth(runtime=mock_runtime, port=0)

        mock_request = MagicMock()
        mock_request.headers = {"X-Correlation-ID": caller_id}
        mock_request.json = AsyncMock(return_value={"command_name": "test_command"})

        await server._handle_skill(mock_request)

        call_args = mock_runtime.dispatch_local_ingress_request.call_args
        ingress_req = call_args.args[0]
        assert str(ingress_req.correlation_id) == caller_id

    @pytest.mark.asyncio
    async def test_handle_skill_timeout_clamped(self, mock_runtime: MagicMock) -> None:
        """timeout_ms is clamped to 600_000 (ModelLocalRuntimeIngressRequest upper bound)."""
        server = ServiceHealth(runtime=mock_runtime, port=0)

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.json = AsyncMock(
            return_value={"command_name": "test_command", "timeout_ms": 900_000}
        )

        await server._handle_skill(mock_request)

        call_args = mock_runtime.dispatch_local_ingress_request.call_args
        ingress_req = call_args.args[0]
        assert ingress_req.timeout_ms == 600_000
