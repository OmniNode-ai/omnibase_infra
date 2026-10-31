# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The delegate request ``onex delegate`` sends, built from its flags.

Every optional field is set only when the caller supplied it: the consumer's
request model declares ``extra="forbid"``, so a null would be a shape change on
every existing caller's payload. Keys a released consumer may not declare ride
in the ``metadata`` map, which every released request consumer accepts.

Moved out of ``omnibase_infra.cli.cli_delegate`` unchanged when the session
model joined the metadata (OMN-20557), so the command line module only shrinks.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Protocol

from omnibase_infra.models.delegation.model_delegate_session_model import (
    ModelDelegateSessionModel,
)

__all__ = [
    "DELEGATE_CALLER_LANE_METADATA_KEY",
    "DELEGATE_TICKET_METADATA_KEY",
    "ProtocolDelegateRequestCaller",
    "build_delegate_request_payload",
]

#: The request ``metadata`` key that names the ticket a delegation works
#: (OMN-19514). omnimarket's delegate-skill handler reads the same key.
DELEGATE_TICKET_METADATA_KEY = "ticket_id"

#: The request ``metadata`` key naming the caller's lane (OMN-19860). The same
#: value as ``omnibase_infra.cli.delegate_caller.DELEGATE_CALLER_LANE_METADATA_KEY``,
#: declared here so this module depends on nothing in the command line package;
#: a test pins the two equal.
DELEGATE_CALLER_LANE_METADATA_KEY = "caller_lane"


class ProtocolDelegateRequestCaller(Protocol):
    """The caller's lane and session, as the command line resolves them."""

    @property
    def lane(self) -> str | None: ...

    @property
    def session_id(self) -> str | None: ...


def build_delegate_request_payload(
    *,
    prompt: str,
    task_type: str,
    source: str,
    max_tokens: int | None,
    correlation_id: uuid.UUID,
    acceptance_criteria: tuple[str, ...] = (),
    quality_contract_mode: str | None = None,
    response_contract: dict[str, object] | None = None,
    system_prompt: str | None = None,
    requested_timeout_seconds: int | None = None,
    backend_id: str | None = None,
    model: str | None = None,
    ticket_id: str | None = None,
    caller: ProtocolDelegateRequestCaller | None = None,
    tenant_id: str | None = None,
    trace_id: uuid.UUID | None = None,
    span_id: uuid.UUID | None = None,
    lineage: Mapping[str, str] | None = None,
    named_tests: Mapping[str, str] | None = None,
    session_model: str | None = None,
) -> dict[str, object]:
    """Build the delegate request, setting only the fields the caller supplied."""
    payload: dict[str, object] = {
        "prompt": prompt,
        "task_type": task_type,
        "source": source,
        "correlation_id": str(correlation_id),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if acceptance_criteria:
        payload["acceptance_criteria"] = list(acceptance_criteria)
    if quality_contract_mode is not None:
        payload["quality_contract_mode"] = quality_contract_mode
    if response_contract is not None:
        payload["response_contract"] = response_contract
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if requested_timeout_seconds is not None:
        payload["requested_timeout_seconds"] = requested_timeout_seconds
    # OMN-19124: the caller's explicit rung pin, written under the field name
    # the node contract already declares. Omitted entirely when unset, like
    # every other optional field here, because ``ModelDelegateSkillRequest``
    # declares ``extra="forbid"`` and a null would be a shape change on every
    # existing caller's payload.
    if backend_id is not None:
        payload["backend_id"] = backend_id
    # OMN-20844: the model the customer's own provider key runs for this call,
    # written under the optional field the node contract declares. Omitted
    # entirely when unset, never null, because ``ModelDelegateSkillRequest``
    # declares ``extra="forbid"`` and existing callers keep their payload shape.
    if model is not None:
        payload["model"] = model
    # OMN-19514: the ticket this delegation works, in the request's metadata
    # map, which every released request consumer already accepts. Omitted
    # entirely when no ticket was named.
    metadata: dict[str, str] = {}
    if ticket_id is not None:
        metadata[DELEGATE_TICKET_METADATA_KEY] = ticket_id
    # OMN-19860: who issued the run. The lane rides in the same metadata map;
    # the session rides in the declared ``session_id`` field, only ever as a
    # UUID. Each is omitted entirely when unresolved.
    if caller is not None and caller.lane is not None:
        metadata[DELEGATE_CALLER_LANE_METADATA_KEY] = caller.lane
    # OMN-20606: the delegation this run falls back or escalates from, already
    # validated by resolve_delegate_lineage. Omitted entirely when unnamed.
    if lineage:
        metadata.update(lineage)
    # OMN-17427: the tests a code answer is graded by, under the metadata keys
    # omnimarket's delegation port reads. Omitted entirely when none are named.
    if named_tests:
        metadata.update(named_tests)
    # OMN-20557: the caller's session model, the run's savings baseline, under
    # the metadata key omnimarket's handler reads. Omitted entirely when unset.
    if session_model is not None:
        entry = ModelDelegateSessionModel(model=session_model).as_metadata()
        if entry.keys() & metadata.keys():
            raise ValueError(
                f"request metadata already names {sorted(entry.keys() & metadata.keys())}; "
                "the session model is written once, never overwritten"
            )
        metadata.update(entry)
    if metadata:
        payload["metadata"] = metadata
    if caller is not None and caller.session_id is not None:
        payload["session_id"] = caller.session_id
    # OMN-17427: the tenant of the install the command runs under, written to
    # the ``tenant_id`` field the request model has declared since OMN-14058.
    # Omitted entirely when unresolved (the in-process first-delegation path),
    # like every other optional field here.
    if tenant_id is not None:
        payload["tenant_id"] = tenant_id
    # OMN-19437: the trace context the handler copies onto the terminal. Omitted
    # entirely when the contract's request model does not declare it yet.
    if trace_id is not None:
        payload["trace_id"] = str(trace_id)
    if span_id is not None:
        payload["span_id"] = str(span_id)
    return payload
