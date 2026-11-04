# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatch image maintenance through the real local-runtime bus and contract."""

from __future__ import annotations

import json
import shlex
import subprocess
from importlib.metadata import entry_points
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


@pytest.mark.parametrize("fail", [False, True])
def test_host_service_uses_its_explicit_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """Run the shipped timer command without attaching host Docker I/O at boot."""
    node_name = "node_deploy_agent_image_prune_effect"
    assert node_name not in {ep.name for ep in entry_points(group="onex.nodes")}
    log = docker_stub(tmp_path, monkeypatch)
    if fail:
        monkeypatch.setenv("FAIL_INVENTORY", "1")
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("DEPLOY_AGENT_GC_REPO", str(root))
    unit = root / "scripts/deploy-agent/deploy/onex-deploy-agent-image-prune.service"
    command = next(
        line.removeprefix("ExecStart=")
        for line in unit.read_text().splitlines()
        if line.startswith("ExecStart=")
    )
    # systemd expands these escapes before invoking the declared shell.
    command = command.replace("$$", "$").replace("%h", str(tmp_path))
    argv = shlex.split(command)
    assert "--contract " in argv[-1]
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == int(fail), result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(call[:2] == ["image", "prune"] for call in calls) is not fail
