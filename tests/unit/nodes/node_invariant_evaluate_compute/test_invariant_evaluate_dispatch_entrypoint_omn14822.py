# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14822 — the invariant-evaluate handlers are REACHABLE through real dispatch.

RED-against-EXISTS-but-WRONG proof for the canonical def-B flip (hand-flip, OMN-14781).

Before this ticket ``node_invariant_evaluate_compute`` declared three
``operation_match`` operations whose only executable code was module-level async
functions (``handle_invariant_evaluate`` / ``_batch`` / ``_all``). The descriptor class
``HandlerInvariantEvaluate`` exposed NO ``handle`` / ``handle_async``. Auto-wiring's
``_make_dispatch_callback`` looks for ``handle_async`` then ``handle``; finding neither
it binds ``_missing_handle``, which raises::

    ModelOnexError: Auto-wired handler HandlerInvariantEvaluate does not expose a
                    callable handle() or handle_async() dispatch entrypoint.

...on the FIRST real dispatch. So the node was contract-declared, wired, ingress-valid
and CI-green while being non-executable through the real dispatch path.

These tests drive the REAL production dispatch callback over the REAL def-B handler
classes (no fake handler, no patched entrypoint). They fail against the pre-flip
descriptor-only handler (``_missing_handle`` → ``ModelOnexError``) and pass only once the
def-B ``handle`` entrypoints exist. The flip is a hand-flip that moves the three thin
delegations onto ``Handler*.handle``; the shared ``_new_evaluator`` helper and the
``HandlerInvariantEvaluate`` classification properties are byte-identical base_ref↔HEAD,
which the canonical-shape ratchet re-derives from git (the ``.handflip.json`` proof).
The ``InvariantEvaluator`` business logic (``evaluator_invariant.py``) is untouched.

``build_selected_corpus`` below is the exact SELECTED input corpus bound (by
``input_hash``) into both the adequacy receipt
(``scripts/ci/adequacy_receipts/omnibase_infra.nodes.node_invariant_evaluate_compute.json``)
and the hand-flip proof (``...handflip.json``). It uses fixed UUIDs / timestamps so the
input hashes are deterministic and reproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumInvariantType, EnumSeverity
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.invariant import ModelInvariant, ModelInvariantSet
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.nodes.node_invariant_evaluate_compute.handlers.handler_invariant_evaluate import (
    HandlerInvariantEvaluate,
    HandlerInvariantEvaluateAll,
    HandlerInvariantEvaluateBatch,
)
from omnibase_infra.nodes.node_invariant_evaluate_compute.models import (
    ModelInvariantEvaluateAllInput,
    ModelInvariantEvaluateBatchInput,
    ModelInvariantEvaluateInput,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback

pytestmark = [pytest.mark.unit]

# Deterministic ids/timestamps so the SELECTED corpus hashes reproducibly.
_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _inv(
    uid: str, name: str, itype: EnumInvariantType, sev: EnumSeverity, cfg: dict
) -> ModelInvariant:
    return ModelInvariant(id=UUID(uid), name=name, type=itype, severity=sev, config=cfg)


def _iset(uid: str, name: str, invs: list[ModelInvariant]) -> ModelInvariantSet:
    return ModelInvariantSet(
        id=UUID(uid), created_at=_TS, name=name, target="test", invariants=invs
    )


_FIELD = _inv(
    "00000000-0000-0000-0000-000000000001",
    "Field Check",
    EnumInvariantType.FIELD_PRESENCE,
    EnumSeverity.CRITICAL,
    {"fields": ["status"]},
)
_THRESHOLD = _inv(
    "00000000-0000-0000-0000-000000000002",
    "Threshold Check",
    EnumInvariantType.THRESHOLD,
    EnumSeverity.WARNING,
    {"metric_name": "score", "min_value": 0.5},
)


def build_selected_corpus() -> list[tuple[str, str, object]]:
    """The (case_id, operation, input) SELECTED corpus bound into both receipts.

    Deterministic: fixed invariant/set ids + timestamps → reproducible input hashes.
    """
    single_set = _iset(
        "00000000-0000-0000-0000-000000000010", "Mixed", [_FIELD, _THRESHOLD]
    )
    return [
        (
            "E1_field_pass",
            "invariant.evaluate",
            ModelInvariantEvaluateInput(invariant=_FIELD, output={"status": "ok"}),
        ),
        (
            "E2_field_fail",
            "invariant.evaluate",
            ModelInvariantEvaluateInput(invariant=_FIELD, output={"other": "x"}),
        ),
        (
            "E3_threshold_pass",
            "invariant.evaluate",
            ModelInvariantEvaluateInput(invariant=_THRESHOLD, output={"score": 0.8}),
        ),
        (
            "B1_batch_mixed",
            "invariant.evaluate_batch",
            ModelInvariantEvaluateBatchInput(
                invariant_set=single_set, output={"status": "ok", "score": 0.8}
            ),
        ),
        (
            "A1_all_fail_soft",
            "invariant.evaluate_all",
            ModelInvariantEvaluateAllInput(
                invariant_set=single_set, output={"score": 0.2}, fail_fast=False
            ),
        ),
        (
            "A2_all_fail_fast",
            "invariant.evaluate_all",
            ModelInvariantEvaluateAllInput(
                invariant_set=single_set, output={"score": 0.2}, fail_fast=True
            ),
        ),
    ]


_CORPUS = build_selected_corpus()
_EVAL_CASES = [
    (cid, payload) for cid, op, payload in _CORPUS if op == "invariant.evaluate"
]
_EVAL_IDS = [c[0] for c in _EVAL_CASES]


@pytest.mark.unit
def test_handlers_expose_handle_entrypoint_and_no_legacy_shim() -> None:
    """Bare invariant: auto-wiring can only bind handle/handle_async (RED pre-flip).

    Also asserts the flip is a real move, not a retained shim: the legacy module-level
    ``handle_invariant_evaluate*`` functions are gone.
    """
    for cls in (
        HandlerInvariantEvaluate,
        HandlerInvariantEvaluateBatch,
        HandlerInvariantEvaluateAll,
    ):
        assert callable(getattr(cls, "handle", None)) or callable(
            getattr(cls, "handle_async", None)
        ), f"{cls.__name__} exposes neither handle() nor handle_async()."

    import omnibase_infra.nodes.node_invariant_evaluate_compute.handlers.handler_invariant_evaluate as mod

    for legacy in (
        "handle_invariant_evaluate",
        "handle_invariant_evaluate_batch",
        "handle_invariant_evaluate_all",
    ):
        assert not hasattr(mod, legacy), (
            f"legacy free function {legacy} must be removed (no retained shim)."
        )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("case_id", "payload"), _EVAL_CASES, ids=_EVAL_IDS)
async def test_real_dispatch_callback_reaches_handle(
    case_id: str, payload: ModelInvariantEvaluateInput
) -> None:
    """LOAD-BEARING: a real payload dispatched through the REAL auto-wiring callback
    reaches ``handle`` and yields a SUCCESS dispatch result.

    Against the descriptor-only pre-flip handler this raises ModelOnexError
    (_missing_handle) rather than returning a result — that raise IS the bug this
    flip closes. ``operation_match`` declares no ``event_model``, so this exercises
    the untyped def-B coercion arm (OMN-14716) exactly as production does.
    """
    callback = _make_dispatch_callback(HandlerInvariantEvaluate(), None)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=payload,
        correlation_id=uuid4(),
        event_type="ModelInvariantEvaluateInput",
    )
    result = await callback(envelope)

    assert result is not None, "Dispatch produced no result — handle never ran."
    assert result.status is EnumDispatchStatus.SUCCESS, (
        f"Expected SUCCESS dispatch status, got {result.status!r}."
    )
    assert result.output_count == 1
    attestation = result.output_events[0]
    assert attestation.invariant_name == payload.invariant.name


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_handle_raises_before_flip_is_the_red() -> None:
    """Documents the exact RED the flip closes: a descriptor-only handler exposing
    only the legacy operation function binds ``_missing_handle`` and raises."""

    class _DescriptorOnlyLikeLegacy:
        """Stand-in with only the legacy operation method (the pre-flip shape)."""

        async def handle_invariant_evaluate(
            self, request: ModelInvariantEvaluateInput
        ) -> object:
            raise AssertionError("unreachable: auto-wiring never binds this name")

    callback = _make_dispatch_callback(_DescriptorOnlyLikeLegacy(), None)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=ModelInvariantEvaluateInput(invariant=_FIELD, output={"status": "ok"}),
        correlation_id=uuid4(),
        event_type="ModelInvariantEvaluateInput",
    )
    with pytest.raises(ModelOnexError, match="does not expose a callable handle"):
        await callback(envelope)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_selected_corpus_behavior_equivalence() -> None:
    """Behavior equivalence over the full SELECTED corpus, all three operations, driven
    through the REAL def-B ``handle`` entrypoints (parity attestation for the receipts)."""
    handlers = {
        "invariant.evaluate": HandlerInvariantEvaluate(),
        "invariant.evaluate_batch": HandlerInvariantEvaluateBatch(),
        "invariant.evaluate_all": HandlerInvariantEvaluateAll(),
    }
    results: dict[str, object] = {}
    for case_id, operation, payload in _CORPUS:
        results[case_id] = await handlers[operation].handle(payload)

    # evaluate op
    assert results["E1_field_pass"].passed is True
    assert results["E2_field_fail"].passed is False
    assert results["E3_threshold_pass"].passed is True
    # batch op — ordered, both pass on compliant output
    batch = results["B1_batch_mixed"]
    assert [r.invariant_name for r in batch] == ["Field Check", "Threshold Check"]
    assert all(r.passed for r in batch)
    # all op — soft: both fail, aggregate counts
    soft = results["A1_all_fail_soft"]
    assert soft.overall_passed is False
    assert soft.critical_failures == 1
    assert soft.warning_failures == 1
    # all op — fail_fast stops at first critical failure
    fast = results["A2_all_fail_fast"]
    assert fast.overall_passed is False
    assert fast.critical_failures == 1
