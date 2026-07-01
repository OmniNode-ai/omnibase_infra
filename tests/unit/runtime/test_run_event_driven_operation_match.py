# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end boot tests for OMN-13141: _run_event_driven executor half.

Disjoint companion to OMN-13137 (#1973). OMN-13137 fixed only the VALIDATOR
half (`_validate_routing` is strategy-aware). This file covers the EXECUTOR
half: `_run_event_driven` wiring loop.

Root cause (L816): for `routing_strategy: operation_match`, handler entries carry
no `event_model` block, so `_resolve_routing_entries` yields
`event_model_module=""`. The wiring loop then calls
`importlib.import_module("")`, which raises `ValueError: Empty module name`.
That `ValueError` is NOT caught by the `(ImportError, AttributeError)` clause and
propagates to the outer `except Exception` in `run_async` -> result FAILED. The
node never boots end-to-end.

Validator-half test 4 (`test_validate_routing_operation_match.py`) only calls
`_validate_routing` — it does NOT boot through `run_async()`, so it cannot catch
this path. These tests boot the real `run_async()` event-driven path.

Fix: skip the event-model import when `event_model_module` is empty (operation
routes by `operation`, not a payload model); wire the adapter with no input
model. payload_type_match keeps importing its event model (regression guard).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from omnibase_core.enums.enum_workflow_result import EnumWorkflowResult
from omnibase_infra.runtime.runtime_local import RuntimeLocal

# ---------------------------------------------------------------------------
# Test 1 — operation_match contract boots via run_async without ValueError
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_operation_match_contract_boots_via_run_async(tmp_path: Path) -> None:
    """An operation_match contract (no event_model) boots end-to-end.

    Before OMN-13141 the wiring loop calls importlib.import_module("") for the
    empty event_model_module and raises ValueError: Empty module name, which the
    outer except in run_async swallows -> result FAILED. After the fix the
    handler is wired without importing a payload model and the workflow reaches
    its terminal event -> COMPLETED.
    """

    class _OperationInput(BaseModel):
        correlation_id: str

    class _OperationTerminal(BaseModel):
        correlation_id: str
        status: str = "success"

    class _OperationHandler:
        async def handle(self, correlation_id: str) -> _OperationTerminal:
            return _OperationTerminal(correlation_id=correlation_id)

    mod_input = types.ModuleType("_test_opmatch_input")
    mod_input._OperationInput = _OperationInput  # type: ignore[attr-defined]
    mod_handler = types.ModuleType("_test_opmatch_handler")
    mod_handler._OperationHandler = _OperationHandler  # type: ignore[attr-defined]

    sys.modules["_test_opmatch_input"] = mod_input
    sys.modules["_test_opmatch_handler"] = mod_handler

    try:
        # operation_match handler entry deliberately omits event_model — this is
        # the canonical shape that produced event_model_module="" -> L816 crash.
        contract_yaml = (
            "name: test_operation_match_boot\n"
            "contract_version: {major: 1, minor: 0, patch: 0}\n"
            "node_type: orchestrator\n"
            "description: operation_match boot test\n"
            "initial_command: cmd.op.start.v1\n"
            "terminal_event: evt.op.done.v1\n"
            "event_bus:\n"
            "  subscribe_topics:\n"
            "    - cmd.op.start.v1\n"
            "  publish_topics:\n"
            "    - evt.op.done.v1\n"
            "input_model:\n"
            "  module: _test_opmatch_input\n"
            "  class: _OperationInput\n"
            "handler_routing:\n"
            "  routing_strategy: operation_match\n"
            "  handlers:\n"
            "    - operation: do_op\n"
            "      handler:\n"
            "        name: _OperationHandler\n"
            "        module: _test_opmatch_handler\n"
        )
        workflow = tmp_path / "operation_match_boot.yaml"
        workflow.write_text(contract_yaml)

        runtime = RuntimeLocal(
            workflow_path=workflow,
            state_root=tmp_path / "state",
            timeout=10,
        )
        result = await runtime.run_async()

        # The node must not FAIL from the empty-module ValueError. Before the fix
        # this asserts COMPLETED but receives FAILED.
        assert result != EnumWorkflowResult.FAILED, (
            "operation_match contract failed to boot — likely the empty "
            "event_model_module ValueError at the wiring loop (OMN-13141)"
        )
        assert result == EnumWorkflowResult.COMPLETED

        # No FAILED state persisted from an import crash.
        import json

        state_file = tmp_path / "state" / "workflow_result.json"
        assert state_file.exists()
        state_data = json.loads(state_file.read_text())
        assert state_data["result"] == "completed"
    finally:
        sys.modules.pop("_test_opmatch_input", None)
        sys.modules.pop("_test_opmatch_handler", None)


# ---------------------------------------------------------------------------
# Test 2 — operation_match wiring does not raise ValueError: Empty module name
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_operation_match_wiring_does_not_import_empty_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring loop must never call importlib.import_module("").

    Guards the exact failure mode: a spy on importlib.import_module asserts the
    empty string is never passed (operation_match entries have no event model to
    import). The handler module is still imported, so the spy sees non-empty
    names only.
    """
    import importlib

    class _OpHandler:
        async def handle(self, correlation_id: str) -> None:
            return None

    mod_handler = types.ModuleType("_test_opmatch_spy_handler")
    mod_handler._OpHandler = _OpHandler  # type: ignore[attr-defined]
    sys.modules["_test_opmatch_spy_handler"] = mod_handler

    real_import_module = importlib.import_module
    empty_calls: list[str] = []

    def _spy_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "":
            empty_calls.append(name)
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local.importlib.import_module",
        _spy_import_module,
    )

    try:
        contract_yaml = (
            "name: test_operation_match_spy\n"
            "contract_version: {major: 1, minor: 0, patch: 0}\n"
            "node_type: orchestrator\n"
            "description: operation_match spy test\n"
            "terminal_event: evt.op.done.v1\n"
            "event_bus:\n"
            "  subscribe_topics:\n"
            "    - cmd.op.start.v1\n"
            "  publish_topics:\n"
            "    - evt.op.done.v1\n"
            "handler_routing:\n"
            "  routing_strategy: operation_match\n"
            "  handlers:\n"
            "    - operation: do_op\n"
            "      handler:\n"
            "        name: _OpHandler\n"
            "        module: _test_opmatch_spy_handler\n"
        )
        workflow = tmp_path / "operation_match_spy.yaml"
        workflow.write_text(contract_yaml)

        runtime = RuntimeLocal(
            workflow_path=workflow,
            state_root=tmp_path / "state",
            timeout=10,
        )
        await runtime.run_async()

        assert empty_calls == [], (
            "importlib.import_module('') was called during operation_match "
            "wiring — empty event_model_module must be skipped (OMN-13141)"
        )
    finally:
        sys.modules.pop("_test_opmatch_spy_handler", None)


# ---------------------------------------------------------------------------
# Test 3 — payload_type_match still imports its event model (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_payload_type_match_still_imports_event_model(tmp_path: Path) -> None:
    """payload_type_match path is unchanged — its event model is still imported.

    A payload_type_match entry pointing at a non-importable event_model module
    must still FAIL (the import error is real). This guards against the fix
    over-broadly skipping imports for the payload strategy.
    """

    class _PtInput(BaseModel):
        correlation_id: str

    class _PtHandler:
        async def handle(self, correlation_id: str) -> None:
            return None

    mod_handler = types.ModuleType("_test_ptmatch_handler")
    mod_handler._PtHandler = _PtHandler  # type: ignore[attr-defined]
    sys.modules["_test_ptmatch_handler"] = mod_handler

    try:
        contract_yaml = (
            "name: test_payload_type_match_bad_model\n"
            "contract_version: {major: 1, minor: 0, patch: 0}\n"
            "node_type: orchestrator\n"
            "description: payload_type_match bad event model\n"
            "terminal_event: evt.pt.done.v1\n"
            "event_bus:\n"
            "  subscribe_topics:\n"
            "    - cmd.pt.start.v1\n"
            "  publish_topics:\n"
            "    - evt.pt.done.v1\n"
            "handler_routing:\n"
            "  routing_strategy: payload_type_match\n"
            "  handlers:\n"
            "    - event_model:\n"
            "        name: NotThere\n"
            "        module: _test_nonexistent_event_model_module\n"
            "      handler:\n"
            "        name: _PtHandler\n"
            "        module: _test_ptmatch_handler\n"
        )
        workflow = tmp_path / "payload_type_match_bad.yaml"
        workflow.write_text(contract_yaml)

        runtime = RuntimeLocal(
            workflow_path=workflow,
            state_root=tmp_path / "state",
            timeout=10,
        )
        result = await runtime.run_async()

        # The event model import genuinely fails -> FAILED is correct here.
        assert result == EnumWorkflowResult.FAILED
    finally:
        sys.modules.pop("_test_ptmatch_handler", None)
