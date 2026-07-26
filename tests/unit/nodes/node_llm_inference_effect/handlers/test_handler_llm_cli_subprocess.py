# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatch-entrypoint (def-B) tests for HandlerLlmCliSubprocess.

OMN-8735 follow-up: auto-wiring compliance — no-args construction.
OMN-14804 (child of OMN-14510 missing-handle burn-down): HandlerLlmCliSubprocess
    was contract-declared on ``node_llm_inference_effect`` for three CLI operations
    (``inference.gemini_cli`` / ``inference.claude_cli`` / ``inference.opencode_cli``)
    yet exposed ONLY its business method ``execute_cli_inference()`` — NEITHER
    ``handle`` NOR ``handle_async``. Auto-wiring's ``_make_dispatch_callback`` binds
    ``_missing_handle`` in that case, which raises ``ModelOnexError`` on the FIRST
    real dispatch while the contract validates, the node boots, and CI stays green.

    These tests drive the REAL production dispatch callback over the REAL handler
    class (no fake handler, no patched entrypoint). Before OMN-14804 they FAIL for
    the missing-handle reason; they pass only once a genuine def-B ``handle`` exists.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.llm.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_infra.models.llm.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.models.llm.model_llm_message import ModelLlmMessage
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
    HandlerLlmCliSubprocess,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback


@pytest.mark.unit
def test_handler_llm_cli_subprocess_constructs_with_no_args() -> None:
    """HandlerLlmCliSubprocess must construct with no arguments for auto-wiring."""
    handler = HandlerLlmCliSubprocess()
    assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
    assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


@pytest.mark.unit
def test_handler_exposes_a_dispatch_entrypoint() -> None:
    """The bare invariant: auto-wiring can only bind ``handle`` / ``handle_async``.

    RED against the pre-OMN-14804 handler, which had only ``execute_cli_inference()``.
    """
    assert callable(getattr(HandlerLlmCliSubprocess, "handle", None)) or callable(
        getattr(HandlerLlmCliSubprocess, "handle_async", None)
    ), (
        "HandlerLlmCliSubprocess exposes neither handle() nor handle_async(); "
        "auto-wiring binds _missing_handle and every dispatch raises ModelOnexError."
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_real_dispatch_callback_runs_cli_and_emits_response() -> None:
    """LOAD-BEARING: a request dispatched through the REAL auto-wiring callback
    reaches the handler and yields a ``ModelLlmInferenceResponse`` output event.

    The contract declares these CLI operations with NO ``event_model`` (they route
    via ``operation_match``), so this exercises the untyped def-B arm (OMN-14716):
    ``_make_dispatch_callback`` signature-introspects ``handle(request: ModelX)`` and
    coerces the envelope payload into ``ModelLlmInferenceRequest`` at the adapter
    boundary. Against the entrypoint-less handler this raises ``ModelOnexError``
    (``_missing_handle``) instead of returning a result — that raise IS the bug.
    """
    callback = _make_dispatch_callback(HandlerLlmCliSubprocess())
    cid = uuid4()
    request = ModelLlmInferenceRequest(
        base_url="http://localhost:1",
        model="gemini-cli",
        messages=(ModelLlmMessage(role="user", content="What is 2+2?"),),
        correlation_id=cid,
    )
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=request,
        correlation_id=cid,
        event_type="ModelLlmInferenceRequest",
    )
    completed = subprocess.CompletedProcess(
        args=["gemini", "-p", "What is 2+2?"],
        returncode=0,
        stdout="Four\n",
        stderr="",
    )

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=completed),
    ):
        result = await callback(envelope)

    assert result is not None, "Dispatch produced no result — the handler never ran."
    assert len(result.output_events) == 1, (
        "Expected exactly one ModelLlmInferenceResponse output event from the def-B "
        f"CLI-subprocess handler. Got {result.output_events!r}"
    )
    event = result.output_events[0]
    assert isinstance(event, ModelLlmInferenceResponse)
    assert event.generated_text == "Four"  # stdout stripped
    assert event.correlation_id == cid


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_handler_receives_typed_request_not_raw_envelope() -> None:
    """Seam check: the def-B ``handle`` receives a validated ModelLlmInferenceRequest,
    NOT a raw envelope, on the untyped (operation_match / no event_model) arm.

    Before OMN-14716 the untyped arm handed the handler the raw ModelEventEnvelope,
    which a def-B handler breaks on at first attribute access. This guards that the
    ``ModelLlmInferenceRequest`` annotation is runtime-resolvable so the coercion
    fires (it would silently no-op if the model were import-guarded behind
    ``TYPE_CHECKING`` and the annotation failed ``eval_str`` resolution).
    """
    seen: list[object] = []
    handler = HandlerLlmCliSubprocess()
    original = handler.handle

    async def _spy(request: ModelLlmInferenceRequest) -> ModelHandlerOutput[None]:
        seen.append(request)
        return await original(request)

    handler.handle = _spy  # type: ignore[method-assign]
    callback = _make_dispatch_callback(handler)
    cid = uuid4()
    request = ModelLlmInferenceRequest(
        base_url="http://localhost:1",
        model="gemini-cli",
        messages=(ModelLlmMessage(role="user", content="ping"),),
        correlation_id=cid,
    )
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=request,
        correlation_id=cid,
        event_type="ModelLlmInferenceRequest",
    )
    completed = subprocess.CompletedProcess(
        args=["gemini", "-p", "ping"], returncode=0, stdout="pong\n", stderr=""
    )

    with (
        patch("shutil.which", return_value="/usr/bin/gemini"),
        patch("subprocess.run", return_value=completed),
    ):
        await callback(envelope)

    assert len(seen) == 1, "Handler was not invoked exactly once."
    assert isinstance(seen[0], ModelLlmInferenceRequest), (
        "def-B handle received a raw envelope instead of a validated "
        f"ModelLlmInferenceRequest: {type(seen[0]).__name__}."
    )
