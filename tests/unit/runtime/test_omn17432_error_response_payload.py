# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17432 — every runtime error envelope must carry a parseable payload.

``RuntimeHostProcess._create_error_response`` returned four keys and no
``payload``::

    {"success": False, "status": "error", "error": ..., "correlation_id": ...}

A consumer that requires an envelope payload before it will act — the gateway's
``parse_session_event_envelope`` is the live example, raising
``"session-event envelope is missing a 'payload' object"`` *before* it resolves
the pending correlation — treats that envelope as if it had never been sent.
The runtime believes it answered; the caller waits out its full bound and is
told ``503``. An envelope that cannot be parsed is not an answer.

Two things are being fixed here, and they are separable:

* **the shape** — an error envelope carries a ``payload`` object, so a consumer
  that gates on payload presence can resolve the correlation it is holding;
* **the content** — that payload is the same typed
  :class:`ModelBoundaryFailureTerminal` the OMN-16812 consume boundary already
  publishes, attributed through the same ``classify_boundary_failure``. One wire
  shape for "the runtime failed your request", whichever surface noticed. A
  caller that learns to read it once reads it everywhere, and OMN-17397's
  omnimarket subscriber already reads exactly this class.

Note the second point is why the ``error`` string alone was not enough even
where a consumer tolerated the missing payload: ``"Handler execution failed"``
names the surface, never the cause, and carries no retryability — so a caller
obeying it either gives up on a transient failure or retries a permanent one
forever.

Scope note: this file covers the function the ticket names. It is NOT the
surface that produced the observed gateway 503 — see
``tests/unit/runtime/auto_wiring/test_omn17432_gateway_route_terminal.py`` for
that chain and its own proof.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.runtime.boundary_failure_terminal import (
    ModelBoundaryFailureTerminal,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config

_LIVE_ONEX_CODE = "ONEX_CORE_041_INVALID_CONFIGURATION"


def _process() -> RuntimeHostProcess:
    return RuntimeHostProcess(config=make_runtime_config(output_topic="test.output"))


def _configuration_failure() -> ProtocolConfigurationError:
    context = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="handler_execution",
    )
    return ProtocolConfigurationError(
        "No tier has a configured endpoint for task_type='agent_delegation'",
        context=context,
    )


@pytest.mark.unit
def test_error_envelope_carries_a_payload_object() -> None:
    """The single read that decides whether a waiting caller is ever released."""
    correlation_id = uuid4()
    envelope = _process()._create_error_response(
        error="Handler execution failed",
        correlation_id=correlation_id,
        exception=_configuration_failure(),
    )

    payload = envelope.get("payload")
    assert isinstance(payload, dict), (
        "payload-less error envelope — a consumer that gates on payload "
        "presence never resolves the correlation and the caller times out"
    )
    assert envelope["correlation_id"] == correlation_id


@pytest.mark.unit
def test_the_payload_is_the_typed_attributed_failure() -> None:
    """One wire shape for runtime failure, attributed, not a bare string."""
    correlation_id = uuid4()
    envelope = _process()._create_error_response(
        error="Handler execution failed",
        correlation_id=correlation_id,
        exception=_configuration_failure(),
    )

    terminal = ModelBoundaryFailureTerminal.model_validate(envelope["payload"])
    assert terminal.correlation_id == correlation_id
    assert terminal.status == "failed"
    assert terminal.failure_class == "ProtocolConfigurationError"
    assert terminal.failure_code == _LIVE_ONEX_CODE
    # A missing configured endpoint is not fixed by asking again.
    assert terminal.retryable is False


@pytest.mark.unit
def test_a_transient_failure_is_still_reported_retryable() -> None:
    """Contrast case: the attribution must not answer 'permanent' to everything."""
    envelope = _process()._create_error_response(
        error="Handler execution failed",
        correlation_id=uuid4(),
        exception=TimeoutError("upstream did not answer in time"),
    )

    terminal = ModelBoundaryFailureTerminal.model_validate(envelope["payload"])
    assert terminal.retryable is True
    assert terminal.failure_class == "TimeoutError"
    assert terminal.failure_code is None


@pytest.mark.unit
def test_the_historical_top_level_keys_are_preserved() -> None:
    """Additive by construction — existing readers of this envelope keep working.

    ``success``/``status``/``error`` are what today's consumers key on. The
    payload is added beside them, not in place of them, so this change cannot
    strand a reader that has not learned the typed shape yet.
    """
    envelope = _process()._create_error_response(
        error="Invalid JSON in message: boom",
        correlation_id=uuid4(),
        exception=json.JSONDecodeError("boom", "{}", 0),
    )

    assert envelope["success"] is False
    assert envelope["status"] == "error"
    assert envelope["error"] == "Invalid JSON in message: boom"


@pytest.mark.unit
def test_a_missing_correlation_id_is_minted_once_and_agrees_with_the_payload() -> None:
    """The envelope and its payload must not name two different correlations.

    ``_create_error_response`` mints a correlation when the inbound envelope had
    none (a decode failure before any id could be read). Minting it twice — once
    for the envelope, once for the payload — would produce a record that
    correlates to nothing, in two different ways.
    """
    envelope = _process()._create_error_response(
        error="Invalid JSON in message: boom",
        correlation_id=None,
        exception=json.JSONDecodeError("boom", "{}", 0),
    )

    terminal = ModelBoundaryFailureTerminal.model_validate(envelope["payload"])
    assert isinstance(envelope["correlation_id"], UUID)
    assert terminal.correlation_id == envelope["correlation_id"]


@pytest.mark.unit
def test_the_payload_survives_envelope_serialization() -> None:
    """The publish path JSON-serializes the envelope; the payload must round-trip.

    ``_serialize_envelope`` walks the envelope converting UUIDs to strings. A
    payload it could not walk into would reach the bus as an unserializable
    object and the publish would fail — turning a bad answer into no answer.
    """
    process = _process()
    correlation_id = uuid4()
    envelope = process._create_error_response(
        error="Handler execution failed",
        correlation_id=correlation_id,
        exception=_configuration_failure(),
    )

    wire = json.loads(json.dumps(process._serialize_envelope(envelope)))

    assert wire["correlation_id"] == str(correlation_id)
    terminal = ModelBoundaryFailureTerminal.model_validate(wire["payload"])
    assert terminal.correlation_id == correlation_id
    assert terminal.failure_code == _LIVE_ONEX_CODE
