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


# ---------------------------------------------------------------------------
# Test 4 — OMN-13277: event_model-only contract (no top-level input_model)
#          seeds the FULL payload from the routing entry's event_model.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_model_only_contract_seeds_full_payload(tmp_path: Path) -> None:
    """A contract WITHOUT a top-level ``input_model`` seeds the full payload.

    OMN-13277 root cause: ``_run_event_driven`` resolved the initial-payload
    model ONLY from the top-level ``input_model``. When absent it published a
    degenerate ``{correlation_id}``-only payload, silently dropping every other
    caller-supplied field (the OMN-13253 failure mode). After the hardening the
    payload model is resolved from the routing entry's ``event_model`` block, so
    the ``--input`` file's required field (``ticket_id``) reaches the handler.
    """

    class _StartCommand(BaseModel):
        correlation_id: str = ""
        ticket_id: str

    class _Terminal(BaseModel):
        correlation_id: str = ""
        ticket_id: str
        status: str = "success"

    captured: dict[str, str] = {}

    class _EchoHandler:
        async def handle(self, payload: _StartCommand) -> _Terminal:
            # Capture exactly what the runtime seeded so the test can prove the
            # caller's ticket_id survived (not dropped to {correlation_id}).
            captured["ticket_id"] = payload.ticket_id
            return _Terminal(
                correlation_id=payload.correlation_id,
                ticket_id=payload.ticket_id,
            )

    mod_model = types.ModuleType("_test_em_only_model")
    mod_model._StartCommand = _StartCommand  # type: ignore[attr-defined]
    mod_handler = types.ModuleType("_test_em_only_handler")
    mod_handler._EchoHandler = _EchoHandler  # type: ignore[attr-defined]
    sys.modules["_test_em_only_model"] = mod_model
    sys.modules["_test_em_only_handler"] = mod_handler

    try:
        # NOTE: deliberately NO top-level input_model. The only place the payload
        # model can be resolved from is the handler routing entry's event_model.
        contract_yaml = (
            "name: test_event_model_only_seed\n"
            "contract_version: {major: 1, minor: 0, patch: 0}\n"
            "node_type: orchestrator\n"
            "description: event_model-only payload seeding test\n"
            "terminal_event: evt.em.done.v1\n"
            "event_bus:\n"
            "  subscribe_topics:\n"
            "    - cmd.em.start.v1\n"
            "  publish_topics:\n"
            "    - evt.em.done.v1\n"
            "handler_routing:\n"
            "  routing_strategy: payload_type_match\n"
            "  handlers:\n"
            "    - event_model:\n"
            "        name: _StartCommand\n"
            "        module: _test_em_only_model\n"
            "      handler:\n"
            "        name: _EchoHandler\n"
            "        module: _test_em_only_handler\n"
        )
        workflow = tmp_path / "event_model_only.yaml"
        workflow.write_text(contract_yaml)

        input_file = tmp_path / "input.json"
        input_file.write_text('{"ticket_id": "OMN-99999"}')

        runtime = RuntimeLocal(
            workflow_path=workflow,
            state_root=tmp_path / "state",
            timeout=10,
            input_path=input_file,
        )
        result = await runtime.run_async()

        assert result == EnumWorkflowResult.COMPLETED
        # The caller's ticket_id must reach the handler — proving the full
        # payload was seeded, not a degenerate {correlation_id} dict.
        assert captured.get("ticket_id") == "OMN-99999", (
            "event_model-only contract dropped caller input — payload was not "
            "seeded from the routing entry's event_model (OMN-13277 regression)"
        )
    finally:
        sys.modules.pop("_test_em_only_model", None)
        sys.modules.pop("_test_em_only_handler", None)


# ---------------------------------------------------------------------------
# Test 5 — OMN-13277: a contract resolving NO payload model fails loud.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_resolvable_payload_model_fails_loud(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A contract with NO resolvable payload model fails loud, naming the contract.

    OMN-13277: when NEITHER a top-level ``input_model`` NOR an ``event_model`` /
    ``handler.input_model`` block can be resolved, the runtime must refuse to
    publish a degenerate ``{correlation_id}``-only payload. It raises a typed
    ``ModelOnexError`` (recorded as FAILED at the run_async boundary) whose
    message names the contract and the missing declarations.
    """
    import logging

    class _OpHandler:
        async def handle(self, correlation_id: str) -> None:
            return None

    mod_handler = types.ModuleType("_test_no_model_handler")
    mod_handler._OpHandler = _OpHandler  # type: ignore[attr-defined]
    sys.modules["_test_no_model_handler"] = mod_handler

    try:
        # operation_match entry: no event_model, no handler.input_model, and no
        # top-level input_model => nothing to resolve a payload model from.
        contract_yaml = (
            "name: test_no_resolvable_model\n"
            "contract_version: {major: 1, minor: 0, patch: 0}\n"
            "node_type: orchestrator\n"
            "description: no resolvable payload model test\n"
            "terminal_event: evt.nm.done.v1\n"
            "event_bus:\n"
            "  subscribe_topics:\n"
            "    - cmd.nm.start.v1\n"
            "  publish_topics:\n"
            "    - evt.nm.done.v1\n"
            "handler_routing:\n"
            "  routing_strategy: operation_match\n"
            "  handlers:\n"
            "    - operation: do_op\n"
            "      handler:\n"
            "        name: _OpHandler\n"
            "        module: _test_no_model_handler\n"
        )
        workflow = tmp_path / "no_resolvable_model.yaml"
        workflow.write_text(contract_yaml)

        runtime = RuntimeLocal(
            workflow_path=workflow,
            state_root=tmp_path / "state",
            timeout=10,
        )
        with caplog.at_level(logging.ERROR):
            result = await runtime.run_async()

        # Fail loud: FAILED result, never a degenerate publish + downstream
        # validation error.
        assert result == EnumWorkflowResult.FAILED

        messages = " ".join(rec.getMessage() for rec in caplog.records)
        assert "could not resolve an initial-payload model" in messages
        assert "test_no_resolvable_model" in messages, (
            "fail-loud message must name the offending contract (OMN-13277)"
        )
    finally:
        sys.modules.pop("_test_no_model_handler", None)
