# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14826 — HandlerPrStateProjection canonical def-B dispatch proof.

RED-against-EXISTS-but-WRONG proof for the canonical def-B flip (hand-flip,
OMN-14781 path) under the canonical-shape ratchet epic OMN-14355. Mirrors the
sibling canary OMN-14821 (node_build_loop_projection_compute), which this node
explicitly mirrors.

Before this ticket ``HandlerPrStateProjection.handle`` was declared
``handle(self, message: object) -> ModelHandlerOutput`` — it exposed a callable
``handle`` (so it was NOT entrypoint-less) but the ``object``-typed parameter is
NOT adaptable to the canonical definition-B shape: the shared runtime adapter
``_resolve_def_b_input_model_type`` returns ``None`` for it, so the dispatcher
would hand the handler the RAW materialized envelope instead of a validated
``ModelEventMessage``. The handler compensated with a defensive
``_coerce_event_message`` and carried a dead legacy ProtocolHandler ``execute``
path. The canonical-shape ratchet classified the node ``nonadaptable`` (baselined
in ``NON_CANONICAL``).

The flip retypes the entrypoint parameter to the contract input model
``ModelEventMessage`` (canonical def-B), removes the dead ``execute`` /
``_safe_correlation_id`` path, and drops the lone ``ModelEventEnvelope`` comment
(C-core). The projection business logic (``project`` / ``_extract_payload`` /
``_require_str`` / ``_require_int`` / ``_first_str`` / ``_extract_timestamp`` /
``_extract_correlation_id`` / ``_coerce_event_message``) is preserved
byte-identical base_ref<->HEAD, which the ratchet re-derives from git (the
``.handflip.json`` proof).

``test_handle_is_canonical_def_b_typed_entrypoint`` is the RED discriminator: it
asserts the REAL runtime helper resolves ``ModelEventMessage`` from the entrypoint
signature and that the legacy ``execute`` path is gone — both FALSE on the
pre-flip tree, TRUE on the flip. The parametrized dispatch tests drive the REAL
production ``_make_dispatch_callback`` over the SELECTED input corpus and assert a
SUCCESS dispatch carrying the projected ``pr_state.upsert`` intent; that corpus is
the exact set bound (by ``input_hash``) into the adequacy receipt and the
hand-flip proof under ``scripts/ci/adequacy_receipts/``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_pr_state_projection_compute import (
    HandlerPrStateProjection,
    ModelPayloadPrStateUpsert,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _handler_accepts_event_envelope,
    _make_dispatch_callback,
    _resolve_def_b_input_model_type,
)

pytestmark = [pytest.mark.unit]

_TOPIC = "onex.evt.github.pr-status.v1"
# Deterministic header fields so the SELECTED input corpus hashes reproducibly —
# the adequacy receipt + hand-flip proof pin these exact payloads via input_hash.
_FIXED_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_FIXED_MSG_ID = UUID("00000000-0000-0000-0000-0000000000bb")
_FIXED_CID = UUID("11111111-1111-1111-1111-111111111111")


def _message(body: dict[str, object]) -> ModelEventMessage:
    return ModelEventMessage(
        topic=_TOPIC,
        partition=0,
        offset="0",
        key=None,
        value=json.dumps(body).encode("utf-8"),
        headers=ModelEventHeaders(
            source="test-pr-poller",
            event_type="github.pr-status",
            timestamp=_FIXED_TS,
            correlation_id=_FIXED_CID,
            message_id=_FIXED_MSG_ID,
        ),
    )


# The SELECTED input corpus — id, ModelEventMessage, expected (repo, pr_number,
# triage_state, is_draft). Each case exercises a distinct branch of the projection
# logic (all-fields / envelope-unwrap / fallbacks / is_draft+alt-timestamp+bad-cid).
_CASES: list[tuple[str, ModelEventMessage, tuple[str, int, str, bool]]] = [
    (
        "P1_full",
        _message(
            {
                "repo": "OmniNode-ai/omnibase_infra",
                "pr_number": 2260,
                "triage_state": "ready_to_merge",
                "title": "fix(OMN-14375): pr_state projection",
                "as_of": "2026-01-01T00:00:00+00:00",
                "correlation_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1",
            }
        ),
        ("OmniNode-ai/omnibase_infra", 2260, "ready_to_merge", False),
    ),
    (
        "P2_envelope_wrapped",
        _message(
            {
                "envelope_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbb2",
                "payload": {
                    "repo": "OmniNode-ai/omniclaude",
                    "pr_number": 42,
                    "triage_state": "needs_review",
                },
            }
        ),
        ("OmniNode-ai/omniclaude", 42, "needs_review", False),
    ),
    (
        "P3_minimal",
        _message({"repo": "OmniNode-ai/omnimarket", "pr_number": 1}),
        ("OmniNode-ai/omnimarket", 1, "needs_review", False),
    ),
    (
        "P4_isdraft_alt_keys",
        _message(
            {
                "repo": "OmniNode-ai/omnidash",
                "pr_number": 7,
                "is_draft": True,
                "timestamp": "2026-02-02T03:04:05Z",
                "correlation_id": "not-a-uuid",
            }
        ),
        ("OmniNode-ai/omnidash", 7, "needs_review", True),
    ),
]
_CASE_IDS = [c[0] for c in _CASES]


@pytest.fixture
def handler() -> HandlerPrStateProjection:
    return HandlerPrStateProjection(container=MagicMock())


@pytest.mark.unit
def test_handle_is_canonical_def_b_typed_entrypoint(
    handler: HandlerPrStateProjection,
) -> None:
    """RED discriminator: the entrypoint is the canonical def-B typed shape.

    Pre-flip (``message: object``) the runtime resolves NO typed input model
    (returns ``None``) and the legacy ``execute`` path exists — this test is RED
    there. Post-flip the runtime resolves ``ModelEventMessage`` and ``execute``
    is gone.
    """
    assert _resolve_def_b_input_model_type(handler.handle) is ModelEventMessage, (
        "handle() is not an adaptable def-B typed entrypoint — the runtime would "
        "hand it the raw envelope instead of a validated ModelEventMessage."
    )
    # The core does NOT accept a raw envelope; the envelope boundary is the
    # shared runtime adapter (definition B, C-core).
    assert _handler_accepts_event_envelope(handler.handle) is False
    # The dead legacy ProtocolHandler entrypoint was removed (single entrypoint).
    assert not hasattr(handler, "execute"), (
        "legacy execute() ProtocolHandler path must be removed by the def-B flip."
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("case_id", "message", "expected"), _CASES, ids=_CASE_IDS)
async def test_real_dispatch_callback_projects_intent(
    case_id: str,
    message: ModelEventMessage,
    expected: tuple[str, int, str, bool],
) -> None:
    """LOAD-BEARING: a real PR status event dispatched through the REAL auto-wiring
    callback reaches the def-B ``handle`` and yields a SUCCESS dispatch carrying
    the projected ``pr_state.upsert`` intent.

    The contract declares ``operation_match`` (no ``event_model``), so this
    exercises the untyped def-B coercion arm: the adapter validates the extracted
    payload into ``ModelEventMessage`` and passes the typed model.
    """
    handler = HandlerPrStateProjection(container=MagicMock())
    # operation_match handler => auto-wiring passes event_model=None.
    callback = _make_dispatch_callback(handler, None)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=message,
        correlation_id=uuid4(),
        event_type="ModelEventMessage",
    )

    result = await callback(envelope)

    assert result is not None, "Dispatch produced no result — the handler never ran."
    assert result.status is EnumDispatchStatus.SUCCESS, (
        f"Expected SUCCESS dispatch status, got {result.status!r}."
    )
    assert len(result.output_intents) == 1, (
        f"Expected exactly one projected pr_state.upsert intent, got "
        f"{result.output_intents!r}"
    )
    intent = result.output_intents[0]
    assert isinstance(intent, ModelIntent)
    assert intent.intent_type == "pr_state.upsert"
    assert intent.target.startswith("postgres://pr_state/")
    payload = intent.payload
    assert isinstance(payload, ModelPayloadPrStateUpsert)
    repo, pr_number, triage_state, is_draft = expected
    assert payload.repo == repo
    assert payload.pr_number == pr_number
    assert payload.triage_state == triage_state
    assert payload.is_draft is is_draft


@pytest.mark.unit
def test_missing_typed_entrypoint_is_the_red() -> None:
    """Documents the exact RED the flip closes.

    A handler whose ``handle`` takes an ``object`` param is not an adaptable
    def-B typed entrypoint — the runtime resolves no input model, so it would
    hand the handler the raw envelope. This guards against silent regression of
    the ``object``-typed non-canonical shape through the REAL runtime helper.
    """

    class _LegacyShape:
        async def handle(self, message: object) -> None:
            raise AssertionError("unreachable")

    assert _resolve_def_b_input_model_type(_LegacyShape().handle) is None
