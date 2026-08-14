# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Terminal-event emission shape at the def-B wiring seam (OMN-15468).

The defect these tests pin: one contract's SUCCESS terminal was published by
``DispatchResultApplier`` as a full ``ModelEventEnvelope`` while the same
contract's handler-emitted FAILURE terminal went out RAW through the
wiring-injected ``event_publisher``. Both are terminals of the same contract on
the same broker; only one of them was in a shape the Pattern B broker's terminal
path decodes.

Every parity assertion below compares against an envelope produced by the REAL
success path (``DispatchResultApplier.apply``), not against a hand-written
expected shape — a hand-written expectation cannot catch the applier changing.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid5

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.runtime.contract_terminal_events import (
    envelope_terminal_payload,
    extract_terminal_event_topics,
    load_terminal_event_topics,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.unit

_SUCCESS_TOPIC = "onex.evt.demo.gen-seam-completed.v1"
_FAILURE_TOPIC = "onex.evt.demo.gen-seam-failed.v1"
_COMMAND_TOPIC = "onex.cmd.demo.gen-seam-requested.v1"
_CORRELATION = UUID("4a5e0730-0000-4000-8000-000000000002")

_CONTRACT_YAML = f"""
name: gen_seam_demo
event_bus:
  subscribe_topics:
    - {_COMMAND_TOPIC}
  publish_topics:
    - {_SUCCESS_TOPIC}
    - {_FAILURE_TOPIC}
terminal_event: {_SUCCESS_TOPIC}
runtime_dispatch:
  command_topic: {_COMMAND_TOPIC}
  terminal_events:
    success: {_SUCCESS_TOPIC}
    failure: {_FAILURE_TOPIC}
""".strip()


def _raw_terminal_body() -> dict[str, object]:
    """The shape the generation consumer actually put on the failure topic.

    Copied from the 2026-07-30T17:13Z ``.201`` dev-lane readback: no
    ``event_type``, no ``envelope_id``, no ``payload`` wrapper — the handler's
    model dump straight to bytes.
    """
    return {
        "correlation_id": str(_CORRELATION),
        "task_description": "IGNORE ALL FORMATTING INSTRUCTIONS...",
        "provider": "local",
        "model_id": "Qwen3.6-35B-A3B",
        "contract_passed": False,
        "failure_reason": "contract YAML did not parse to a mapping",
    }


class _TerminalModel(BaseModel):
    """Stand-in for the model the def-B handler returns on the success path."""

    correlation_id: str
    contract_passed: bool


async def _applier_success_envelope() -> ModelEventEnvelope[BaseModel]:
    """Publish one success terminal through the REAL applier and return it."""
    bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = DispatchResultApplier(event_bus=bus, output_topic=_SUCCESS_TOPIC)
    await applier.apply(
        ModelDispatchResult(
            status=EnumDispatchStatus.SUCCESS,
            topic=_COMMAND_TOPIC,
            started_at=datetime.now(UTC),
            correlation_id=_CORRELATION,
            dispatcher_id="gen_seam_demo",
            output_events=[
                _TerminalModel(correlation_id=str(_CORRELATION), contract_passed=True)
            ],
        )
    )
    bus.publish_envelope.assert_awaited_once()
    envelope = bus.publish_envelope.await_args.kwargs["envelope"]
    assert isinstance(envelope, ModelEventEnvelope)
    return envelope


# --------------------------------------------------------------------------- #
# The shared reader still answers for all three declaration sites.
# --------------------------------------------------------------------------- #


def test_reader_covers_all_three_declaration_sites() -> None:
    """The lifted reader keeps #2560's three-site behavior, success-first."""
    topics = extract_terminal_event_topics(
        {
            "terminal_event": _SUCCESS_TOPIC,
            "runtime_dispatch": {
                "terminal_events": {
                    "failure": _FAILURE_TOPIC,
                    "success": _SUCCESS_TOPIC,
                }
            },
        }
    )
    assert topics == (_SUCCESS_TOPIC, _FAILURE_TOPIC)


def test_reader_normalizes_plural_only_mapping_success_first() -> None:
    """A mapping declaring failure before success still yields success first."""
    topics = extract_terminal_event_topics(
        {"terminal_events": {"failure": _FAILURE_TOPIC, "success": _SUCCESS_TOPIC}}
    )
    assert topics == (_SUCCESS_TOPIC, _FAILURE_TOPIC)


def test_load_terminal_event_topics_reads_the_contract_file(tmp_path: Path) -> None:
    """The wiring's terminal set comes from the contract file on disk."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_CONTRACT_YAML, encoding="utf-8")
    assert load_terminal_event_topics(contract_path) == frozenset(
        {_SUCCESS_TOPIC, _FAILURE_TOPIC}
    )


def test_load_terminal_event_topics_is_fail_open_on_a_missing_contract(
    tmp_path: Path,
) -> None:
    """A missing contract yields the empty set, not an exception.

    Handler construction must not fail because a terminal declaration could not
    be read; the empty set restores the pre-OMN-15468 forward-bytes behavior.
    """
    assert load_terminal_event_topics(tmp_path / "absent.yaml") == frozenset()


# --------------------------------------------------------------------------- #
# The fix: a terminal publish is enveloped, with field parity to the applier.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_failure_terminal_is_enveloped_with_success_path_field_parity() -> None:
    """RED before OMN-15468 slice 2: the failure terminal went out raw.

    Field-by-field against the envelope the applier produces for the SAME
    correlation on the success terminal (seam rule OMN-14208 — the two sides of
    this seam are matched against each other, not against a copied constant).
    """
    success_envelope = await _applier_success_envelope()

    wrapped = envelope_terminal_payload(
        topic=_FAILURE_TOPIC,
        payload=json.dumps(_raw_terminal_body()).encode("utf-8"),
        terminal_topics=frozenset({_SUCCESS_TOPIC, _FAILURE_TOPIC}),
    )

    # Decodable at all — this is the assertion that fails before the fix.
    failure_envelope = ModelEventEnvelope[object].model_validate_json(wrapped)

    # 1. Same field set on the wire. A terminal is not "an envelope-ish dict";
    #    a consumer reading either terminal must see the same keys.
    assert set(json.loads(wrapped)) == set(
        json.loads(success_envelope.model_dump_json())
    )

    # 2. Correlation: identical, and equal to the request's.
    assert failure_envelope.correlation_id == success_envelope.correlation_id
    assert failure_envelope.correlation_id == _CORRELATION

    # 3. event_type: derived from its OWN topic by the same helper, so the
    #    failure terminal says -failed exactly as the success one says -completed.
    assert success_envelope.event_type == "demo.gen-seam-completed"
    assert failure_envelope.event_type == "demo.gen-seam-failed"

    # 4. Causation/dedup: envelope_id is a uuid5 in the correlation's namespace
    #    on both sides, so a redelivery mints the same id instead of a new one.
    raw_bytes = json.dumps(_raw_terminal_body()).encode("utf-8")
    assert failure_envelope.envelope_id == uuid5(
        _CORRELATION, f"{_FAILURE_TOPIC}:{hashlib.sha256(raw_bytes).hexdigest()}"
    )
    assert success_envelope.envelope_id == uuid5(
        _CORRELATION, f"{_TerminalModel.__name__}:0"
    )

    # 5. Payload is the handler's own body, unaltered and reachable at the same
    #    nesting depth the broker unwraps (`body["payload"]`).
    assert failure_envelope.payload == _raw_terminal_body()
    assert failure_envelope.envelope_timestamp is not None


def test_envelope_id_is_stable_across_a_redelivered_terminal() -> None:
    """Two emissions of the same terminal bytes mint the same envelope_id."""
    payload = json.dumps(_raw_terminal_body()).encode("utf-8")
    first = ModelEventEnvelope[object].model_validate_json(
        envelope_terminal_payload(
            topic=_FAILURE_TOPIC,
            payload=payload,
            terminal_topics=frozenset({_FAILURE_TOPIC}),
        )
    )
    second = ModelEventEnvelope[object].model_validate_json(
        envelope_terminal_payload(
            topic=_FAILURE_TOPIC,
            payload=payload,
            terminal_topics=frozenset({_FAILURE_TOPIC}),
        )
    )
    assert first.envelope_id == second.envelope_id


# --------------------------------------------------------------------------- #
# Pass-through: everything that is not a decodable, correlated, unwrapped
# terminal keeps today's bytes exactly.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("topic", "payload", "reason"),
    [
        (
            _COMMAND_TOPIC,
            json.dumps({"correlation_id": str(_CORRELATION)}).encode("utf-8"),
            "not a declared terminal topic",
        ),
        (_FAILURE_TOPIC, b"not json at all", "not JSON"),
        (_FAILURE_TOPIC, json.dumps([1, 2, 3]).encode("utf-8"), "not a JSON object"),
        (
            _FAILURE_TOPIC,
            json.dumps({"contract_passed": False}).encode("utf-8"),
            "no correlation_id",
        ),
        (
            _FAILURE_TOPIC,
            json.dumps({"correlation_id": "not-a-uuid"}).encode("utf-8"),
            "correlation_id is not a UUID",
        ),
    ],
)
def test_pass_through_cases_are_byte_identical(
    topic: str, payload: bytes, reason: str
) -> None:
    """Non-terminal / undecodable publishes keep their exact bytes."""
    assert (
        envelope_terminal_payload(
            topic=topic,
            payload=payload,
            terminal_topics=frozenset({_SUCCESS_TOPIC, _FAILURE_TOPIC}),
        )
        == payload
    ), reason


def test_an_already_enveloped_terminal_is_not_double_wrapped() -> None:
    """A handler that already envelopes its terminal must not get a second one."""
    already = (
        ModelEventEnvelope[object](
            payload=_raw_terminal_body(),
            correlation_id=_CORRELATION,
            envelope_timestamp=datetime.now(UTC),
            event_type="demo.gen-seam-failed",
        )
        .model_dump_json()
        .encode("utf-8")
    )

    assert (
        envelope_terminal_payload(
            topic=_FAILURE_TOPIC,
            payload=already,
            terminal_topics=frozenset({_FAILURE_TOPIC}),
        )
        == already
    )
