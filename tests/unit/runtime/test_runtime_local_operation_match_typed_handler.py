# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Real-dispatch-path coverage for typed single-model-parameter handlers reached
via ``operation_match`` routing (OMN-8724).

``operation_match`` routing entries declare no event model, so the runtime wires
the adapter with ``input_model_cls=None`` and forwards the raw decoded dict
(OMN-13141). When such a handler declares a single positional parameter annotated
with a concrete ``BaseModel`` subclass — e.g. ``handle(self, request:
GoldenChainSweepRequest)`` — the adapter must validate the dict into that type
before calling. Before OMN-8724 it forwarded the bare dict and the handler
crashed on the first attribute access (``AttributeError: 'dict' object has no
attribute '<field>'``) the moment it ran through ``onex node`` / RuntimeLocal,
even though the module ``__main__`` entrypoint (which builds the model itself)
worked.

These tests boot the real ``RuntimeLocal._run_event_driven`` path end-to-end —
not the handler in isolation — so they would have caught the dispatch-boundary
crash. Handler-isolation tests pass while the live ``onex node`` path fails;
this is the regression guard for the live path.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_workflow_result import EnumWorkflowResult
from omnibase_infra.runtime.runtime_local import RuntimeLocal
from omnibase_infra.runtime.runtime_local_adapter import _invoke_handle_method

# ---------------------------------------------------------------------------
# Typed request/result models + a single-model-parameter handler that mirrors
# NodeGoldenChainSweep.handle(self, request: GoldenChainSweepRequest).
# ---------------------------------------------------------------------------


class _TypedRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: str = ""
    items: list[str] = Field(default_factory=list)


class _TypedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str = ""
    item_count: int = 0
    status: str = "success"


class _TypedRequestHandler:
    """Sync handler whose sole positional parameter is a concrete BaseModel.

    Accesses an attribute on the request immediately — the exact shape that
    crashed in node_golden_chain_sweep when handed a raw dict.
    """

    def handle(self, request: _TypedRequest) -> _TypedResult:
        return _TypedResult(
            correlation_id=request.correlation_id,
            item_count=len(request.items),
        )


# ---------------------------------------------------------------------------
# Unit-level: _invoke_handle_method coerces a dict into the annotated model.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_invoke_coerces_dict_into_single_model_parameter() -> None:
    """A raw dict payload is validated into the handler's annotated BaseModel."""
    handler = _TypedRequestHandler()
    payload = {"correlation_id": "cid-coerce", "items": ["a", "b", "c"]}

    result = _invoke_handle_method(handler.handle, payload)

    assert isinstance(result, _TypedResult)
    assert result.correlation_id == "cid-coerce"
    assert result.item_count == 3


@pytest.mark.unit
def test_invoke_does_not_double_wrap_model_payload() -> None:
    """When the payload is already the model instance, it is passed through."""
    handler = _TypedRequestHandler()
    request = _TypedRequest(correlation_id="cid-model", items=["x"])

    result = _invoke_handle_method(handler.handle, request)

    assert isinstance(result, _TypedResult)
    assert result.correlation_id == "cid-model"
    assert result.item_count == 1


@pytest.mark.unit
def test_invoke_dict_coercion_raises_on_invalid_payload() -> None:
    """An invalid dict surfaces a pydantic ValidationError, not AttributeError.

    Fail-fast: bad data must fail validation at the boundary, not silently slip
    a malformed object into the handler.
    """
    from pydantic import ValidationError

    handler = _TypedRequestHandler()
    # extra="forbid" rejects unknown keys → ValidationError (not AttributeError).
    with pytest.raises(ValidationError):
        _invoke_handle_method(handler.handle, {"unexpected_field": 1})


# ---------------------------------------------------------------------------
# Real-dispatch-path: boot an operation_match contract whose handler takes a
# single typed-model parameter, through RuntimeLocal.run_async().
# ---------------------------------------------------------------------------


_CMD_TOPIC = "onex.cmd.test.typed-operation-match-start.v1"
_EVT_TOPIC = "onex.evt.test.typed-operation-match-completed.v1"


def _typed_operation_match_contract() -> str:
    return (
        "name: test_typed_operation_match\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        "node_type: compute\n"
        "description: operation_match handler with a single typed-model parameter\n"
        "terminal_event: " + _EVT_TOPIC + "\n"
        "event_bus:\n"
        "  subscribe_topics:\n"
        "    - " + _CMD_TOPIC + "\n"
        "  publish_topics:\n"
        "    - " + _EVT_TOPIC + "\n"
        "handler_routing:\n"
        "  routing_strategy: operation_match\n"
        "  handlers:\n"
        "    - operation: run_typed\n"
        "      handler:\n"
        "        name: _TypedRequestHandler\n"
        "        module: _test_typed_op_match_handler\n"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_operation_match_typed_handler_boots_via_run_async(
    tmp_path: Path,
) -> None:
    """A typed single-model-parameter handler completes via the real dispatch path.

    Regression guard for OMN-8724: before the fix, the adapter forwarded the raw
    decoded dict to ``handle(self, request: _TypedRequest)`` and the run ended
    FAILED with ``AttributeError: 'dict' object has no attribute ...``. The fix
    validates the dict into ``_TypedRequest`` first, so the run COMPLETES and the
    terminal event carries the handler's result.
    """
    mod = types.ModuleType("_test_typed_op_match_handler")
    mod._TypedRequestHandler = _TypedRequestHandler  # type: ignore[attr-defined]
    sys.modules["_test_typed_op_match_handler"] = mod

    try:
        contract = tmp_path / "contract.yaml"
        contract.write_text(_typed_operation_match_contract())

        runtime = RuntimeLocal(
            workflow_path=contract,
            state_root=tmp_path / "state",
            timeout=10,
        )
        result = await runtime.run_async()

        assert result == EnumWorkflowResult.COMPLETED, (
            f"typed operation_match handler did not complete: "
            f"result={result.value} last_error={runtime.last_error!r}"
        )
        assert runtime.last_error is None

        state_data = json.loads(
            (tmp_path / "state" / "workflow_result.json").read_text()
        )
        assert state_data["result"] == "completed"
        assert state_data["exit_code"] == 0
        # Terminal payload carries the handler's typed result, proving the dict
        # was validated into _TypedRequest and processed.
        assert state_data["terminal_payload"]["status"] == "success"
    finally:
        sys.modules.pop("_test_typed_op_match_handler", None)
