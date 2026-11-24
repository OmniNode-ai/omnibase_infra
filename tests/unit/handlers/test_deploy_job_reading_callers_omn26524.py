# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every handler operation of node_deploy_job_reading_effect has a caller in src (OMN-26524).

OMN-26485 added ``check_routing_key_checks`` and ``alarm_routed_commands_without_job_record``
to the node contract and stated that no caller wiring came with them, so both read NOT
RUNNING on dev. The host composition root of the node (``runtime_deploy_job_reading``,
which the deploy agent loads by path) is where the node's other operations are called; the
contract names each caller in ``settings.operation_callers`` and the deploy agent's idle
tick reaches them through the one entry it already calls.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
_NODE = _SRC / "omnibase_infra/nodes/node_deploy_job_reading_effect"
_ROUTING_OPERATIONS = (
    "check_routing_key_checks",
    "alarm_routed_commands_without_job_record",
    "request_overdue_dev_merge_build",
)


def _contract() -> dict[str, Any]:
    loaded = yaml.safe_load((_NODE / "contract.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _handlers() -> list[dict[str, Any]]:
    return list(_contract()["handler_routing"]["handlers"])


def test_every_handler_of_the_node_is_named_by_src_outside_its_own_module() -> None:
    """A handler named only by its module and the contract is a handler nothing calls."""
    uncalled: list[str] = []
    for entry in _handlers():
        handler = entry["handler"]
        name, module = handler["name"], handler["module"]
        own = _SRC / (str(module).replace(".", "/") + ".py")
        callers = [
            path
            for path in _SRC.rglob("*.py")
            if path != own and name in path.read_text(encoding="utf-8")
        ]
        if not callers:
            uncalled.append(f"{entry['operation']} -> {name}")
    assert uncalled == []


def test_the_contract_declares_one_caller_per_routing_operation() -> None:
    settings = _contract()["settings"]
    declared = settings["operation_callers"]
    assert set(declared) >= set(_ROUTING_OPERATIONS)
    runtime = ast.parse(
        (_NODE / "runtime_deploy_job_reading.py").read_text(encoding="utf-8")
    )
    functions = {
        node.name for node in ast.walk(runtime) if isinstance(node, ast.FunctionDef)
    }
    for operation in _ROUTING_OPERATIONS:
        assert declared[operation] in functions, operation


def test_the_idle_tick_entry_reaches_each_declared_caller() -> None:
    """``check_pin`` is the one entry the agent's idle tick calls; it runs the callers."""
    settings = _contract()["settings"]
    declared = settings["operation_callers"]
    runtime = ast.parse(
        (_NODE / "runtime_deploy_job_reading.py").read_text(encoding="utf-8")
    )
    entry = next(
        node
        for node in ast.walk(runtime)
        if isinstance(node, ast.FunctionDef) and node.name == "check_pin"
    )
    called = {
        node.func.id
        for node in ast.walk(entry)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for operation in _ROUTING_OPERATIONS:
        assert declared[operation] in called, operation
