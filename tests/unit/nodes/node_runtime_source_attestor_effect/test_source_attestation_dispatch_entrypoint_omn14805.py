# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14805 — HandlerSourceAttestation is REACHABLE through the real dispatch path.

RED-against-EXISTS-but-WRONG proof for the canonical def-B flip (hand-flip, OMN-14781).

Before this ticket ``HandlerSourceAttestation`` was contract-declared
(``node_runtime_source_attestor_effect/contract.yaml``, ``operation_match`` operation
``attest_source_hash``), wired, ingress-valid and CI-green while exposing only
``attest()`` — NO ``handle`` / ``handle_async``. Auto-wiring's
``_make_dispatch_callback`` looks for ``handle_async`` then ``handle``; finding neither
it binds ``_missing_handle``, which raises::

    ModelOnexError: Auto-wired handler HandlerSourceAttestation does not expose a
                    callable handle() or handle_async() dispatch entrypoint.

...on the FIRST dispatch of ``onex.evt.omnibase-infra.runtime-booted.v1``. So the node
passed ingress and then died on every real boot event.

These tests drive the REAL production dispatch callback over the REAL handler class (no
fake handler, no patched entrypoint), so they fail against the ``attest``-only handler
(``_missing_handle`` → ``ModelOnexError``) and pass only once the def-B ``handle``
entrypoint exists. The flip is a PURE RENAME ``attest`` → ``handle`` that preserves the
attestation body verbatim; the business-logic helpers (``_resolve_main_head`` /
``_compute_distance`` / ``_emit_friction``) are byte-identical base_ref↔HEAD, which the
canonical-shape ratchet re-derives from git (the ``.handflip.json`` proof).

The four ``ModelRuntimeBootedEvent`` inputs below are the exact SELECTED input corpus
bound (by ``input_hash``) into both the adequacy receipt
(``omnibase_infra.nodes.node_runtime_source_attestor_effect.json``) and the hand-flip
proof (``...handflip.json``) under ``scripts/ci/adequacy_receipts/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import pytest

from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.health.model_runtime_booted_event import (
    ModelRuntimeBootedEvent,
)
from omnibase_infra.nodes.node_runtime_source_attestor_effect.handlers.handler_source_attestation import (
    HandlerSourceAttestation,
    ModelSourceAttestationResult,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback

pytestmark = [pytest.mark.unit]

# Deterministic timestamp so the SELECTED input corpus hashes reproducibly — the
# adequacy receipt + hand-flip proof pin these exact payloads via input_hash.
_FIXED_BOOTED_AT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
# A full 40-char SHA whose 7-char prefix is "a1b2c3d" (the E3 compliant payload).
_FAKE_MAIN_HEAD = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


def _event(container_ref: str, runtime_source_hash: str) -> ModelRuntimeBootedEvent:
    return ModelRuntimeBootedEvent(
        container_ref=container_ref,
        runtime_source_hash=runtime_source_hash,
        booted_at=_FIXED_BOOTED_AT,
    )


# The SELECTED input corpus — id, event, patch (main_head, distance) | None, verdict.
_CASES: list[tuple[str, ModelRuntimeBootedEvent, tuple[str, int] | None, str]] = [
    (
        "E1_unknown",
        _event("omnibase-infra-runtime-unknown", "unknown"),
        None,
        "unknown_hash",
    ),
    ("E2_empty", _event("omnibase-infra-runtime-empty", ""), None, "unknown_hash"),
    (
        "E3_compliant",
        _event("omnibase-infra-runtime-compliant", "a1b2c3d"),
        (_FAKE_MAIN_HEAD, 0),
        "compliant",
    ),
    (
        "E4_drifted",
        _event("omnibase-infra-runtime-drifted", "deadbeef1234567"),
        (_FAKE_MAIN_HEAD, 10),
        "drifted",
    ),
]
_CASE_IDS = [case[0] for case in _CASES]


@pytest.mark.unit
def test_handler_exposes_handle_entrypoint() -> None:
    """The bare invariant: auto-wiring can only bind handle/handle_async.

    RED against the pre-OMN-14805 handler, which exposed only ``attest``.
    """
    assert callable(getattr(HandlerSourceAttestation, "handle", None)) or callable(
        getattr(HandlerSourceAttestation, "handle_async", None)
    ), (
        "HandlerSourceAttestation exposes neither handle() nor handle_async(); "
        "auto-wiring binds _missing_handle and every dispatch raises ModelOnexError."
    )
    # The pure rename removed the legacy operation-named entrypoint.
    assert not hasattr(HandlerSourceAttestation, "attest"), (
        "attest must be renamed to handle (no retained attest, no delegating shim)."
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_id", "event", "patch_args", "verdict"), _CASES, ids=_CASE_IDS
)
async def test_real_dispatch_callback_returns_success(
    case_id: str,
    event: ModelRuntimeBootedEvent,
    patch_args: tuple[str, int] | None,
    verdict: str,
    tmp_path: object,
) -> None:
    """LOAD-BEARING: a real runtime-booted payload dispatched through the REAL
    auto-wiring callback reaches ``handle`` and yields a SUCCESS dispatch result.

    Against the ``attest``-only handler this raises ModelOnexError (_missing_handle)
    rather than returning a result — that raise IS the bug this flip closes. The
    contract declares ``operation_match`` (no ``event_model``), so this exercises the
    untyped def-B coercion arm (OMN-14716) exactly as production does.
    """
    friction_dir = tmp_path / "friction"  # type: ignore[operator]
    handler = HandlerSourceAttestation(
        repo_url="https://example.com/fake.git",
        drift_threshold=5,
        friction_dir=friction_dir,
    )
    # operation_match handler => auto-wiring passes event_model=None.
    callback = _make_dispatch_callback(handler, None)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=event,
        correlation_id=uuid4(),
        event_type="ModelRuntimeBootedEvent",
    )

    if patch_args is not None:
        main_head, distance = patch_args
        with (
            patch.object(handler, "_resolve_main_head", return_value=main_head),
            patch.object(handler, "_compute_distance", return_value=distance),
        ):
            result = await callback(envelope)
    else:
        result = await callback(envelope)

    assert result is not None, "Dispatch produced no result — the handler never ran."
    assert result.status is EnumDispatchStatus.SUCCESS, (
        f"Expected SUCCESS dispatch status, got {result.status!r}."
    )
    assert len(result.output_events) == 1, (
        f"Expected exactly one ModelSourceAttestationResult, got {result.output_events!r}"
    )
    attestation = result.output_events[0]
    assert isinstance(attestation, ModelSourceAttestationResult)
    assert attestation.verdict == verdict
    assert attestation.container_ref == event.container_ref


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_handle_raises_before_flip_is_the_red() -> None:
    """Documents the exact RED the flip closes.

    A handler exposing neither handle nor handle_async binds ``_missing_handle`` and
    raises ModelOnexError on first dispatch. Post-flip HandlerSourceAttestation DOES
    expose handle, so we assert the negative directly: a bare object with no
    entrypoint still reproduces the missing-handle ModelOnexError through the REAL
    callback (guards against silent regression of the _missing_handle path).
    """

    class _EntrypointlessLikeLegacy:
        """Stand-in with only the legacy operation method (the pre-flip shape)."""

        def attest(
            self, event: ModelRuntimeBootedEvent
        ) -> ModelSourceAttestationResult:
            raise AssertionError("unreachable: auto-wiring never binds attest()")

    callback = _make_dispatch_callback(_EntrypointlessLikeLegacy(), None)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=_event("c", "unknown"),
        correlation_id=uuid4(),
        event_type="ModelRuntimeBootedEvent",
    )
    with pytest.raises(ModelOnexError, match="does not expose a callable handle"):
        await callback(envelope)
