# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14831 — HandlerValidationLedgerProjection is CANONICAL def-B and behavior-equivalent.

RED-against-EXISTS-but-WRONG proof for the canonical def-B flip (hand-flip, OMN-14781).

Before this ticket the contract-declared handler exposed
``handle(self, message: object)`` — a non-adaptable signature the canonical-shape
ratchet (``canonical_handler_shape.py``) classifies ``nonadaptable`` (the parameter
``message`` is not a runtime-adapter magic name and ``object`` is not a BaseModel-typed
annotation), and the module docstring referenced ``ModelEventEnvelope`` (a C-core
violation). The flip is a SIGNATURE-ONLY rename to
``handle(self, request: ModelEventMessage)`` plus removal of that docstring string; the
entire extraction pipeline (``project`` / ``_extract_metadata`` / ``_extract_uuid`` /
``_extract_timestamp`` / ``_extract_version_from_topic`` /
``_extract_header_correlation_id`` / ``_parse_offset``) is preserved byte-identically,
which the ratchet re-derives from git (the ``.handflip.json`` proof).

These tests drive the REAL production dispatch callback
(``handler_wiring._make_dispatch_callback``) over the REAL handler (no fake handler, no
patched entrypoint) with the contract-declared ``event_model`` (``ModelEventMessage``),
and additionally assert that the def-B ``handle`` output is byte-for-byte derived from
the preserved ``project()`` op-method (equivalence). The four ``ModelEventMessage``
inputs below are the exact SELECTED input corpus bound (by ``input_hash``) into both the
adequacy receipt and the hand-flip proof under ``scripts/ci/adequacy_receipts/``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.validation_ledger import ModelPayloadValidationLedgerAppend
from omnibase_infra.nodes.node_validation_ledger_projection_compute.handlers.handler_validation_ledger_projection import (
    HandlerValidationLedgerProjection,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProtocolHandleable,
    _make_dispatch_callback,
)

pytestmark = [pytest.mark.unit]

# Deterministic header fields so the SELECTED corpus hashes reproducibly — the
# adequacy receipt + hand-flip proof pin these exact payloads via input_hash.
_FIXED_CID = UUID("11111111-1111-4111-8111-111111111111")
_FIXED_MID = UUID("22222222-2222-4222-8222-222222222222")
_FIXED_TS = datetime(2026, 7, 20, 0, 0, 0, tzinfo=UTC)

_STARTED = "onex.evt.validation.cross-repo-run-started.v1"
_BATCH = "onex.evt.validation.cross-repo-violations-batch.v1"
_COMPLETED = "onex.evt.validation.cross-repo-run-completed.v1"

# The contract-declared event_model for this operation_match handler.
_EVENT_MODEL_REF = ModelHandlerRef(
    name="ModelEventMessage",
    module="omnibase_infra.event_bus.models.model_event_message",
)


def _msg(topic: str, value: bytes, event_type: str) -> ModelEventMessage:
    return ModelEventMessage(
        topic=topic,
        value=value,
        headers=ModelEventHeaders(
            correlation_id=_FIXED_CID,
            message_id=_FIXED_MID,
            timestamp=_FIXED_TS,
            source="cross-repo-validator",
            event_type=event_type,
        ),
        offset="7",
        partition=0,
    )


_C0_VALUE = json.dumps(
    {
        "run_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "repo_id": "omnibase_core",
        "event_type": _STARTED,
        "timestamp": "2026-07-20T00:00:00+00:00",
    }
).encode("utf-8")
_C1_VALUE = json.dumps({"repo_id": 123, "timestamp": "not-a-timestamp"}).encode("utf-8")
_C2_VALUE = b"this is not json at all"
_C3_VALUE = b"[1, 2, 3]"

# id, message, expected (repo_id, event_type, event_version, kafka_topic).
_CASES: list[tuple[str, ModelEventMessage, tuple[str, str, str, str]]] = [
    (
        "c0_started_full",
        _msg(_STARTED, _C0_VALUE, _STARTED),
        ("omnibase_core", _STARTED, "v1", _STARTED),
    ),
    (
        "c1_batch_partial",
        _msg(_BATCH, _C1_VALUE, _BATCH),
        ("unknown", _BATCH, "v1", _BATCH),
    ),
    (
        "c2_completed_nonjson",
        _msg(_COMPLETED, _C2_VALUE, _COMPLETED),
        ("unknown", _COMPLETED, "v1", _COMPLETED),
    ),
    (
        "c3_started_listjson",
        _msg(_STARTED, _C3_VALUE, _STARTED),
        ("unknown", _STARTED, "v1", _STARTED),
    ),
]
_CASE_IDS = [c[0] for c in _CASES]

_HANDLER_SRC = Path(
    inspect.getsourcefile(HandlerValidationLedgerProjection) or ""
).read_text(encoding="utf-8")


@pytest.mark.unit
def test_handle_is_canonical_defb_signature() -> None:
    """The canonical-shape invariant: adaptable ``handle(request: ModelEventMessage)``.

    RED against the pre-flip handler, whose ``handle(self, message: object)`` the
    canonical-shape ratchet classifies ``nonadaptable`` and whose module referenced
    ``ModelEventEnvelope`` (a C-core violation).
    """
    sig = inspect.signature(HandlerValidationLedgerProjection.handle)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    assert len(params) == 1, (
        f"def-B handle takes exactly one request param; got {params}"
    )
    (param,) = params
    assert param.name == "request", (
        f"handle's sole param must be the adaptable magic name 'request'; got {param.name!r}"
    )
    # from __future__ import annotations -> the annotation is the string literal.
    assert param.annotation == "ModelEventMessage", (
        f"handle's request must be typed ModelEventMessage (input_model); got "
        f"{param.annotation!r}"
    )
    # C-core: the handler module must not reference the envelope type.
    assert "ModelEventEnvelope" not in _HANDLER_SRC, (
        "handler module references ModelEventEnvelope — fails C-core (envelope_in_core)."
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("case_id", "message", "expected"), _CASES, ids=_CASE_IDS)
async def test_real_dispatch_callback_returns_expected_intent(
    case_id: str,
    message: ModelEventMessage,
    expected: tuple[str, str, str, str],
) -> None:
    """LOAD-BEARING: a real validation event dispatched through the REAL auto-wiring
    callback (with the contract's ``event_model``) reaches def-B ``handle`` and yields a
    SUCCESS dispatch result carrying the expected ``validation_ledger.append`` intent.

    Against the pre-flip ``nonadaptable`` handler the ratchet blocks the flip; this test
    proves the flipped handler is reachable and behavior-correct on the production path.
    """
    exp_repo, exp_event_type, exp_version, exp_topic = expected
    handler = HandlerValidationLedgerProjection()
    callback = _make_dispatch_callback(
        cast("ProtocolHandleable", handler), _EVENT_MODEL_REF
    )
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
        f"Expected exactly one validation_ledger.append intent, got {result.output_intents!r}"
    )
    intent = result.output_intents[0]
    assert isinstance(intent, ModelIntent)
    payload = intent.payload
    assert isinstance(payload, ModelPayloadValidationLedgerAppend)
    assert payload.intent_type == "validation_ledger.append"
    assert payload.repo_id == exp_repo
    assert payload.event_type == exp_event_type
    assert payload.event_version == exp_version
    assert payload.kafka_topic == exp_topic
    assert payload.kafka_partition == 0
    assert payload.kafka_offset == 7
    assert payload.envelope_hash == hashlib.sha256(message.value).hexdigest()
    assert intent.target == (f"postgres://validation_event_ledger/{exp_topic}/0/7")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_matches_project_equivalence() -> None:
    """EQUIVALENCE: def-B ``handle`` output is byte-for-byte derived from the preserved
    ``project()`` op-method (guards against a future ``handle`` rewrite silently dropping
    or corrupting a field).
    """
    import base64

    message = _msg(_STARTED, _C0_VALUE, _STARTED)
    # Reconstruct the exact args handle() forwards into the preserved project().
    header_bytes = {"correlation_id": str(_FIXED_CID).encode("utf-8")}
    expected = HandlerValidationLedgerProjection().project(
        topic=message.topic,
        partition=0,
        offset=7,
        value=message.value,
        _headers=header_bytes,
    )

    output = await HandlerValidationLedgerProjection().handle(message)
    intent = output.result
    assert isinstance(intent, ModelIntent)
    payload = intent.payload
    assert isinstance(payload, ModelPayloadValidationLedgerAppend)

    assert payload.run_id == expected["run_id"]
    assert payload.repo_id == expected["repo_id"]
    assert payload.event_type == expected["event_type"]
    assert payload.event_version == expected["event_version"]
    assert payload.occurred_at == expected["occurred_at"]
    assert payload.kafka_topic == expected["kafka_topic"]
    assert payload.kafka_partition == expected["kafka_partition"]
    assert payload.kafka_offset == expected["kafka_offset"]
    assert payload.envelope_hash == expected["envelope_hash"]
    # bytes cannot cross the intent boundary -> handle base64-encodes project's bytes.
    expected_bytes = expected["envelope_bytes"]
    assert isinstance(expected_bytes, bytes)
    assert payload.envelope_bytes == base64.b64encode(expected_bytes).decode("ascii")
