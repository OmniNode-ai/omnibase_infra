# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19972: the delegation dispatch terminal node runs from its contract on disk.

The handler and its models are resolved from ``contract.yaml``'s
``handler_routing``, ``input_model`` and ``output_model``, not imported by hand.
The routes come from the real ingress discovery over an installed delegate-skill
contract that declares ``runtime_dispatch.run_step_event``. An escalated run's
terminal comes back as the caller's result and its eight numbered steps for that
topic; with the declaration removed, the same terminal yields no steps.
"""

from __future__ import annotations

import asyncio
import importlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from pydantic import BaseModel

from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.models.delegation.model_delegation_dispatch_terminal_result import (
    ModelDelegationDispatchTerminalResult,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    discover_runtime_local_ingress_routes,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_delegation_dispatch_terminal_effect"
    / "contract.yaml"
)
RUN_STEP_TOPIC = "onex.evt.omnimarket.delegation-run-step.v1"
ESCALATED_RUN: dict[str, object] = {
    "routing_disposition": "routed",
    "backend_ref": "backend-b",
    "model_used": "model-b",
    "cost_tier_name": "cheap_cloud",
    "latency_ms": 900,
    "escalation_history": [
        {
            "backend_ref": "backend-a",
            "model_used": "model-a",
            "tier": "local",
            "latency_ms": 400,
            "acceptance_decision": "escalate",
            "acceptance_reason": "quality_gate_rejection",
            "failure_reasons": ["answer too short"],
        },
        {
            "backend_ref": "backend-b",
            "model_used": "model-b",
            "tier": "cheap_cloud",
            "latency_ms": 900,
            "acceptance_decision": "accept",
        },
    ],
}


def _load(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


def _install_skill_contract(root: Path, run_step_event: str | None) -> None:
    contract = root / "nodes" / "node_delegate_skill_orchestrator" / "contract.yaml"
    contract.parent.mkdir(parents=True, exist_ok=True)
    declared = f"  run_step_event: {run_step_event}\n" if run_step_event else ""
    contract.write_text(
        "name: delegate_skill_orchestrator\n"
        "event_bus:\n"
        "  subscribe_topics:\n"
        "    - onex.cmd.omnimarket.delegate-skill.v1\n"
        "runtime_dispatch:\n"
        "  command_topic: onex.cmd.omnimarket.delegate-skill.v1\n"
        "  terminal_events:\n"
        "    success: onex.evt.omnimarket.delegate-skill-completed.v1\n" + declared,
        encoding="utf-8",
    )


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_step_event: str | None
) -> ModelDelegationDispatchTerminalResult:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    (route,) = contract["handler_routing"]["handlers"]
    assert route["operation"] == "delegation.dispatch_terminal"
    handler_class = _load(route["handler"]["module"], route["handler"]["name"])
    request_model = _load(
        contract["input_model"]["module"], contract["input_model"]["name"]
    )
    output_model = _load(
        contract["output_model"]["module"], contract["output_model"]["name"]
    )
    assert isinstance(handler_class, type)
    assert isinstance(request_model, type) and issubclass(request_model, BaseModel)
    assert output_model is ModelDelegationDispatchTerminalResult

    _install_skill_contract(tmp_path, run_step_event)
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress._resolve_package_root",
        lambda _name: tmp_path,
    )
    routes = discover_runtime_local_ingress_routes(["omnimarket"])
    correlation_id = uuid4()
    request = request_model.model_validate(
        {
            "terminal": ModelDispatchBusTerminalResult(
                correlation_id=correlation_id,
                status="completed",
                payload=ESCALATED_RUN,
                completed_at=datetime.now(UTC),
            ),
            "correlation_id": correlation_id,
            "task_type": "document",
            "tenant_id": "tenant-a",
            "started_at": datetime.now(UTC),
            "started_monotonic": 0.0,
            "engine_package": "omnimarket",
            "publish_steps": True,
        }
    )
    finished = asyncio.run(handler_class().handle(request, routes=routes))
    assert isinstance(finished, ModelDelegationDispatchTerminalResult)
    return finished


def test_the_contract_routed_handler_numbers_an_escalated_run_for_the_declared_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished = _run(tmp_path, monkeypatch, RUN_STEP_TOPIC)

    assert finished.result["status"] == "completed"
    assert finished.result["model_name"] == "model-b"
    assert finished.run_step_topic == RUN_STEP_TOPIC
    assert [(s.step_sequence, s.step_kind, s.outcome) for s in finished.run_steps] == [
        (1, "request", "succeeded"),
        (2, "routing_decision", "succeeded"),
        (3, "provider_call", "succeeded"),
        (4, "quality_gate", "failed"),
        (5, "routing_decision", "succeeded"),
        (6, "provider_call", "succeeded"),
        (7, "quality_gate", "succeeded"),
        (8, "terminal", "succeeded"),
    ]
    assert {s.tenant_id for s in finished.run_steps} == {"tenant-a"}


def test_without_a_declared_topic_the_same_run_has_its_result_and_no_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished = _run(tmp_path, monkeypatch, None)

    assert finished.result["status"] == "completed"
    assert finished.run_step_topic is None
    assert finished.run_steps == ()
