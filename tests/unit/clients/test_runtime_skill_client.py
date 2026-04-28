# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_core.models.runtime import (
    ModelRuntimeSkillRequest,
    ModelRuntimeSkillResponse,
)
from omnibase_infra.clients.runtime_skill_client import (
    LocalRuntimeSkillClient,
    default_runtime_socket_path,
)


@pytest.mark.unit
def test_default_runtime_socket_path_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "test-runtime.sock"
    monkeypatch.setenv("ONEX_LOCAL_RUNTIME_SOCKET_PATH", str(socket_path))
    assert default_runtime_socket_path() == str(socket_path)


@pytest.mark.asyncio
async def test_dispatch_async_round_trip(tmp_path: Path) -> None:
    socket_path = Path(f"/tmp/runtime-skill-client-{uuid4().hex}.sock")  # noqa: S108

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request = json.loads((await reader.readline()).decode("utf-8"))
        correlation_id = UUID(request["correlation_id"])
        response = ModelRuntimeSkillResponse(
            ok=True,
            command_name=request["command_name"],
            contract_name=request["command_name"],
            command_topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
            terminal_event="onex.evt.omnimarket.session-orchestrator-completed.v1",
            correlation_id=correlation_id,
            dispatch_result=ModelDispatchBusTerminalResult(
                correlation_id=correlation_id,
                status="completed",
                payload={"status": "complete"},
            ),
            output_payloads=[{"status": "complete"}],
        )
        writer.write(
            response.model_dump_json(exclude_none=True).encode("utf-8") + b"\n"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(handle_client, path=str(socket_path))
    client = LocalRuntimeSkillClient(socket_path=str(socket_path))
    request = ModelRuntimeSkillRequest(
        command_name="session_orchestrator",
        payload={"dry_run": True},
        correlation_id=uuid4(),
    )

    try:
        response = await client.dispatch_async(request)
    finally:
        server.close()
        await server.wait_closed()
        socket_path.unlink(missing_ok=True)

    assert response.ok is True
    assert response.command_name == "session_orchestrator"
    assert response.output_payloads == [{"status": "complete"}]


@pytest.mark.asyncio
async def test_dispatch_async_returns_runtime_unavailable_for_missing_socket() -> None:
    client = LocalRuntimeSkillClient(socket_path="/tmp/does-not-exist.sock")  # noqa: S108
    response = await client.dispatch_async(
        ModelRuntimeSkillRequest(command_name="session_orchestrator")
    )

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "runtime_unavailable"
