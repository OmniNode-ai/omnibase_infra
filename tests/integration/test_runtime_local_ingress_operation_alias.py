# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for local ingress handler operation aliases."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.runtime.models.model_local_runtime_ingress_request import (
    ModelLocalRuntimeIngressRequest,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.runtime_local_ingress import (
    discover_runtime_local_ingress_routes,
)
from tests.helpers.runtime_helpers import make_runtime_config

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_runtime_local_ingress_dispatches_handler_operation_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "fakepkg"
    node_dir = package_root / "nodes" / "node_delegate_skill_orchestrator"
    node_dir.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (node_dir / "contract.yaml").write_text(
        """
name: node_delegate_skill_orchestrator
event_bus:
  subscribe_topics:
    - onex.cmd.omnimarket.delegate-skill.v1
terminal_event: onex.evt.omnimarket.delegate-skill-completed.v1
handler_routing:
  handlers:
    - operation: delegate_skill.orchestrate
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )

    routes = discover_runtime_local_ingress_routes(("fakepkg",))
    route = routes["delegate_skill.orchestrate"]
    correlation_id = uuid4()
    broker = SimpleNamespace(
        dispatch_request=AsyncMock(
            return_value=(
                route,
                ModelDispatchBusTerminalResult(
                    status="completed",
                    payload={"status": "completed"},
                    completed_at=datetime.now(UTC),
                    correlation_id=correlation_id,
                ),
            )
        )
    )
    process = RuntimeHostProcess(
        config=make_runtime_config(), dispatch_engine=AsyncMock()
    )
    process._is_running = True
    process._local_ingress_routes = routes
    process._pattern_b_broker = cast("object", broker)

    response = await process.dispatch_local_ingress_request(
        ModelLocalRuntimeIngressRequest(
            command_name="delegate_skill.orchestrate",
            payload={"prompt": "write tests", "task_type": "test"},
            correlation_id=correlation_id,
        )
    )

    assert response.ok is True
    assert response.command_name == "node_delegate_skill_orchestrator"
    assert response.command_topic == "onex.cmd.omnimarket.delegate-skill.v1"
    assert response.terminal_event == "onex.evt.omnimarket.delegate-skill-completed.v1"
    assert response.output_payloads == [{"status": "completed"}]

    dispatched = broker.dispatch_request.await_args.args[0]
    assert dispatched.command_name == "node_delegate_skill_orchestrator"
    assert dispatched.correlation_id == correlation_id
