# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runtime local ingress over a Unix socket."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.runtime_local_ingress import (
    RuntimeLocalIngressRoute,
    RuntimeLocalIngressServer,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_unix_socket_request_dispatches_through_runtime_host(
    tmp_path: Path,
) -> None:
    socket_path = Path(f"/tmp/runtime-local-ingress-{uuid4().hex}.sock")  # noqa: S108
    dispatch_result = ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        correlation_id=uuid4(),
    )
    dispatch_engine = AsyncMock()
    dispatch_engine.dispatch = AsyncMock(return_value=dispatch_result)

    process = RuntimeHostProcess(
        config=make_runtime_config(local_ingress={"enabled": True}),
        dispatch_engine=dispatch_engine,
    )
    process._is_running = True
    process._local_ingress_routes = {
        "session_orchestrator": RuntimeLocalIngressRoute(
            node_name="node_session_orchestrator",
            contract_name="session_orchestrator",
            command_topic="onex.cmd.omnimarket.session-orchestrator-start.v1",
            event_type="omnimarket.session-orchestrator-start",
            terminal_event="onex.evt.omnimarket.session-orchestrator-completed.v1",
            contract_path="/tmp/node_session_orchestrator/contract.yaml",  # noqa: S108
            package_name="omnimarket",
        )
    }
    server = RuntimeLocalIngressServer(
        str(socket_path),
        process._dispatch_local_ingress_request,
    )

    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        correlation_id = uuid4()
        writer.write(
            json.dumps(
                {
                    "node_alias": "session_orchestrator",
                    "payload": {"dry_run": True},
                    "correlation_id": str(correlation_id),
                }
            ).encode("utf-8")
            + b"\n"
        )
        await writer.drain()
        response = json.loads((await reader.readline()).decode("utf-8"))
        writer.close()
        await writer.wait_closed()

        assert response["ok"] is True
        assert response["correlation_id"] == str(correlation_id)
        assert response["topic"] == "onex.cmd.omnimarket.session-orchestrator-start.v1"
        dispatch_engine.dispatch.assert_awaited_once()
    finally:
        await server.stop()
