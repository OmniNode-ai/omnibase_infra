# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-to-handler routing of the start-first effect (OMN-22151).

Loads node_runtime_start_first_effect's contract from disk, resolves its routed
operation to the declared handler, and drives HandlerRuntimeStartFirst.handle()
through a fake host execution seam that answers Docker the way a dev lane does.
No Docker daemon is touched.
"""

from __future__ import annotations

import importlib
import subprocess
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers import handler_runtime_start_first as effect_mod
from omnibase_infra.models.model_runtime_start_first_request import (
    ModelRuntimeStartFirstRequest,
)

pytestmark = pytest.mark.unit

CONTRACT = (
    Path(effect_mod.__file__).resolve().parents[1]
    / "nodes"
    / "node_runtime_start_first_effect"
    / "contract.yaml"
)
OLD = "omnibase-infra-omninode-runtime"
NEW = f"{OLD}{effect_mod.REPLACEMENT_SUFFIX}"
READY_URL = "http://localhost:8085/ready"


class _Lane(Enum):
    DEV = "dev"


def _host(ready_after: int | None) -> tuple[SimpleNamespace, list[str]]:
    running = {OLD}
    events: list[str] = []
    probes = [0]

    def run(
        cmd: list[str], timeout: int, *, env: object = None
    ) -> subprocess.CompletedProcess[str]:
        out, code = "", 0
        if cmd[:2] == ["docker", "inspect"]:
            out, code = ("true", 0) if cmd[-1] in running else ("", 1)
        elif "run" in cmd:
            running.add(cmd[cmd.index("--name") + 1])
            events.append("start")
        elif cmd[:2] == ["docker", "exec"]:
            probes[0] += 1
            ready = ready_after is not None and probes[0] >= ready_after
            if ready and "first-ready" not in events:
                events.append("first-ready" if OLD in running else "first-ready-alone")
            out = "200" if ready else "503"
        elif cmd[:3] == ["docker", "rm", "-f"]:
            running.discard(cmd[-1])
            events.append(f"remove {cmd[-1]}")
        else:
            running.discard(cmd[-1])
            events.append(f"stop {cmd[-1]}")
        return subprocess.CompletedProcess(cmd, code, stdout=out, stderr="")

    execution = SimpleNamespace(
        EnumRuntimeLane=_Lane,
        _compose_argv=lambda lane: ["docker", "compose"],
        _compose_env=dict,
        _run=run,
    )
    return execution, events


def _handler_for(
    operation: str, execution: SimpleNamespace
) -> effect_mod.HandlerRuntimeStartFirst:
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["input_model"]["name"] == "ModelRuntimeStartFirstRequest"
    ticks = iter(range(0, 10_000, 5))
    for entry in contract["handler_routing"]["handlers"]:
        if entry["operation"] == operation:
            module = importlib.import_module(entry["handler"]["module"])
            handler = getattr(module, entry["handler"]["name"])(
                execution=execution,
                clock=lambda: float(next(ticks)),
                sleep=lambda seconds: None,
            )
            assert isinstance(handler, effect_mod.HandlerRuntimeStartFirst)
            return handler
    raise AssertionError(f"no route for {operation}")


def _request(bound: float = 60.0) -> ModelRuntimeStartFirstRequest:
    return ModelRuntimeStartFirstRequest(
        service="omninode-runtime",
        old_container=OLD,
        ready_url=READY_URL,
        bound_seconds=bound,
    )


def test_contract_routes_replace_to_an_infra_effect() -> None:
    handler = _handler_for("replace", _host(1)[0])
    assert isinstance(handler, effect_mod.HandlerRuntimeStartFirst)
    assert handler.handler_type is EnumHandlerType.INFRA_HANDLER
    assert handler.handler_category is EnumHandlerTypeCategory.EFFECT


def test_replace_stops_the_old_runtime_only_after_the_replacement_is_ready() -> None:
    execution, events = _host(ready_after=3)
    result = _handler_for("replace", execution).handle(_request())
    assert result.status == "pass"
    assert result.old_running_at_first_ready is True
    assert events == [f"remove {NEW}", "start", "first-ready", f"stop {OLD}"]


def test_replace_never_ready_removes_the_replacement_and_keeps_the_old() -> None:
    execution, events = _host(ready_after=None)
    result = _handler_for("replace", execution).handle(_request(bound=30.0))
    assert result.status == "fail"
    assert result.failure == "replacement-never-ready"
    assert events[-1] == f"remove {NEW}"
    assert f"stop {OLD}" not in events
