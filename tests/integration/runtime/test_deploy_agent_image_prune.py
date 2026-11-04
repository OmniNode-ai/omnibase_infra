# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatch image maintenance through the real local-runtime bus and contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_infra.models.model_deploy_agent_image_prune_result import (
    ModelDeployAgentImagePruneResult,
)
from tests.unit.scripts.test_deploy_agent_image_prune import (
    CONTRACT,
    IMAGE_ID,
    docker_stub,
    make_request,
)


@pytest.mark.parametrize(
    ("operation", "fail"),
    [
        ("deploy_agent.images.plan", False),
        ("deploy_agent.images.prune", False),
        ("deploy_agent.images.prune", True),
    ],
)
def test_actual_runtime_dispatches_contract_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, fail: bool
) -> None:
    log = docker_stub(tmp_path, monkeypatch)
    if fail:
        monkeypatch.setenv("FAIL_INVENTORY", "1")
    input_path = tmp_path / "request.json"
    input_path.write_text(make_request(operation=operation).model_dump_json())
    runtime = RuntimeLocal(
        CONTRACT,
        input_path=input_path,
        state_root=tmp_path / "state",
        backend_overrides={"event_bus": "inmemory"},
        host_handlers=True,
    )
    runtime.run()
    assert isinstance(runtime.handler_result, ModelDeployAgentImagePruneResult), (
        runtime.last_error
    )
    assert runtime.handler_result.success is not fail
    assert (runtime.exit_code != 0) is fail
    if not fail:
        assert runtime.handler_result.remove_image_ids == (IMAGE_ID,)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(call[:2] == ["image", "prune"] for call in calls) is (
        operation == "deploy_agent.images.prune" and not fail
    )
