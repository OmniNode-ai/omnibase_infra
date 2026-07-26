# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14832 - HandlerScopeExtract canonical def-B dispatch proof.

RED-against-EXISTS-but-WRONG proof for the canonical def-B flip (equivalence
path) under the canonical-shape ratchet epic OMN-14355 (Class-B Tier-1 fan-out).

Before this ticket ``HandlerScopeExtract.handle`` was declared multi-positional
``handle(self, content, plan_file_path, correlation_id, output_path) ->
ModelScopeExtracted``. It exposed a callable ``handle`` (so it was NOT
entrypoint-less) but the multi-positional signature is NOT adaptable to the
canonical definition-B shape: the shared runtime adapter
``_resolve_def_b_input_model_type`` returns ``None`` for it, so the dispatcher
would hand the handler the RAW materialized envelope instead of a validated
``ModelScopeExtractInput`` -- and calling the 4-positional ``handle`` with that
single arg raises ``TypeError``. The canonical-shape ratchet classified the node
``nonadaptable`` (baselined in ``NON_CANONICAL``).

The flip retypes the entrypoint to the contract input model
``handle(self, request: ModelScopeExtractInput) -> ModelScopeExtracted`` and
unpacks ``request.*`` at the top; the regex extraction body is behaviorally
identical (proven separately by the golden-equivalence replay).

``test_handle_is_canonical_def_b_typed_entrypoint`` is the RED discriminator: it
asserts the REAL runtime helper resolves ``ModelScopeExtractInput`` from the
entrypoint signature -- FALSE on the pre-flip tree, TRUE on the flip. The
parametrized dispatch tests drive the REAL production ``_make_dispatch_callback``
over the SELECTED input corpus (the exact set bound by ``input_hash`` into the
adequacy + equivalence receipts under ``scripts/ci/adequacy_receipts/``) and
assert a SUCCESS dispatch carrying the projected ``ModelScopeExtracted``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.nodes.node_scope_extract_compute.handlers import (
    HandlerScopeExtract,
)
from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extract_input import (
    ModelScopeExtractInput,
)
from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extracted import (
    ModelScopeExtracted,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _handler_accepts_event_envelope,
    _make_dispatch_callback,
    _resolve_def_b_input_model_type,
)

pytestmark = [pytest.mark.unit]

_GOLDEN_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "golden"
    / "node_scope_extract_compute"
)


def _selected_goldens() -> list[Path]:
    files = sorted(_GOLDEN_DIR.glob("*.json"))
    assert files, f"no golden fixtures under {_GOLDEN_DIR}"
    return files


@pytest.mark.unit
def test_handle_is_canonical_def_b_typed_entrypoint() -> None:
    """RED discriminator: the entrypoint is the canonical def-B typed shape.

    Pre-flip (multi-positional ``handle``) the runtime resolves NO typed input
    model (returns ``None``) -- this test is RED there. Post-flip the runtime
    resolves ``ModelScopeExtractInput`` and the core does not accept a raw
    envelope (definition B: the envelope boundary is the shared runtime adapter).
    """
    handler = HandlerScopeExtract()
    assert _resolve_def_b_input_model_type(handler.handle) is ModelScopeExtractInput, (
        "handle() is not an adaptable def-B typed entrypoint -- the runtime would "
        "hand it the raw envelope instead of a validated ModelScopeExtractInput."
    )
    assert _handler_accepts_event_envelope(handler.handle) is False


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("golden_path", _selected_goldens(), ids=lambda p: p.stem)
async def test_real_dispatch_callback_extracts_scope(golden_path: Path) -> None:
    """LOAD-BEARING: a selected-corpus input dispatched through the REAL auto-wiring
    callback reaches the def-B ``handle`` and yields a SUCCESS dispatch carrying the
    recorded ``ModelScopeExtracted`` scope.

    The contract declares ``operation_match`` (no ``event_model``), so this
    exercises the def-B typed-model coercion arm: the adapter validates the
    payload into ``ModelScopeExtractInput`` and passes the typed model.
    """
    golden: dict[str, Any] = json.loads(golden_path.read_text(encoding="utf-8"))
    request = ModelScopeExtractInput.model_validate(golden["input"])
    callback = _make_dispatch_callback(HandlerScopeExtract(), None)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=request,
        correlation_id=uuid4(),
        event_type="ModelScopeExtractInput",
    )

    result = await callback(envelope)

    assert result is not None, "Dispatch produced no result -- the handler never ran."
    assert result.status is EnumDispatchStatus.SUCCESS, (
        f"Expected SUCCESS dispatch status, got {result.status!r}."
    )
    extracted = [e for e in result.output_events if isinstance(e, ModelScopeExtracted)]
    assert len(extracted) == 1, (
        f"Expected exactly one ModelScopeExtracted result, got {result.output_events!r}"
    )
    scope = extracted[0]
    expected = golden["output"]
    assert list(scope.files) == expected["files"]
    assert list(scope.directories) == expected["directories"]
    assert list(scope.repos) == expected["repos"]
    assert list(scope.systems) == expected["systems"]


@pytest.mark.unit
def test_missing_typed_entrypoint_is_the_red() -> None:
    """Documents the exact RED the flip closes.

    A handler whose ``handle`` takes multiple positional params is not an
    adaptable def-B typed entrypoint -- the runtime resolves no input model, so it
    would hand the handler the raw envelope. Guards against silent regression of
    the multi-positional non-canonical shape through the REAL runtime helper.
    """

    class _LegacyShape:
        async def handle(
            self,
            content: str,
            plan_file_path: str,
            correlation_id: object,
            output_path: str = "",
        ) -> None:
            raise AssertionError("unreachable")

    assert _resolve_def_b_input_model_type(_LegacyShape().handle) is None
