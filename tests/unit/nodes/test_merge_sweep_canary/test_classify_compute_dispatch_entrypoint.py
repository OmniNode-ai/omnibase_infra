# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-dispatch entrypoint proof for the classify-compute def-B flip (OMN-14824).

Drives ``HandlerClassifyPRs`` through the REAL auto-wiring dispatch bind
(``_make_dispatch_callback``) — the same path the runtime uses — so the tests
prove the canonical ``handle(request) -> response`` shape is dispatch-reachable,
not merely importable. Before the def-B flip the class exposed a multi-positional
``handle(prs, correlation_id, require_approval)`` that ``_make_dispatch_callback``
could not adapt (it is not a single typed-request def-B handle), so dispatching a
materialized wire envelope raised ``TypeError``; after the flip the same path
coerces the wire payload into ``ModelClassifyInput`` and returns a
``ModelDispatchResult`` carrying the ``ModelClassifyResult``.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.nodes.node_merge_sweep_classify_compute.handlers.handler_classify_prs import (
    HandlerClassifyPRs,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_classify_input import (
    ModelClassifyInput,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_classify_result import (
    ModelClassifyResult,
)
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_info import (
    ModelPRInfo,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback

pytestmark = pytest.mark.unit


def _pr(**overrides: object) -> ModelPRInfo:
    defaults: dict[str, object] = {
        "number": 1,
        "repo": "OmniNode-ai/test",
        "title": "test PR",
        "mergeable": "MERGEABLE",
        "review_decision": "APPROVED",
        "ci_status": "SUCCESS",
        "is_draft": False,
        "has_auto_merge": False,
    }
    defaults.update(overrides)
    return ModelPRInfo(**defaults)


def _wire_envelope(payload: dict[str, object]) -> dict[str, object]:
    """Raw materialized wire envelope; correlation_id triggers payload unwrap."""
    return {"payload": payload, "correlation_id": payload["correlation_id"]}


@pytest.mark.asyncio
async def test_handler_exposes_handle_entrypoint() -> None:
    """The canonical handler is dispatch-adaptable (single typed-request handle)."""
    callback = _make_dispatch_callback(HandlerClassifyPRs())  # type: ignore[arg-type]
    assert callable(callback)


@pytest.mark.asyncio
async def test_real_dispatch_callback_returns_success() -> None:
    """A materialized wire envelope routes through handle() and returns SUCCESS."""
    cid = str(uuid4())
    payload = {
        "correlation_id": cid,
        "require_approval": True,
        "prs": [
            _pr(number=1).model_dump(mode="json"),  # Track A
            _pr(number=2, ci_status="FAILURE").model_dump(mode="json"),  # Track B
            _pr(number=3, is_draft=True).model_dump(mode="json"),  # SKIP
        ],
    }
    callback = _make_dispatch_callback(HandlerClassifyPRs())  # type: ignore[arg-type]
    result = await callback(_wire_envelope(payload))  # type: ignore[arg-type]

    assert isinstance(result, ModelDispatchResult)
    assert result.status is EnumDispatchStatus.SUCCESS
    assert result.output_count == 1
    (classify_result,) = result.output_events
    assert isinstance(classify_result, ModelClassifyResult)
    assert len(classify_result.track_a) == 1
    assert len(classify_result.track_b) == 1
    assert len(classify_result.skipped) == 1
    assert classify_result.total_classified == 3


@pytest.mark.asyncio
async def test_handle_owns_behavior_direct_call() -> None:
    """handle() itself owns the classification behavior — no retained splat shim."""
    result = await HandlerClassifyPRs().handle(
        ModelClassifyInput(
            correlation_id=uuid4(),
            require_approval=False,
            prs=(_pr(number=7, review_decision=""),),
        )
    )
    assert isinstance(result, ModelClassifyResult)
    assert len(result.track_a) == 1  # approved via not require_approval
