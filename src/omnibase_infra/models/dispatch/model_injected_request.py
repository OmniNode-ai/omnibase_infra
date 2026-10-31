# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed handler request carrying the runtime's injected clock and input id.

A handler core never reads the event envelope. The runtime unwraps the envelope
at the dispatch boundary and hands the handler this request: the domain event,
the clock reading the runtime took for the dispatch, the id of the input that
produced the call and the correlation id that travelled with it.

Handlers that need the time use ``request.now``; they never call a clock or read
``envelope_timestamp``. Handlers that stamp an output with the id of the input
they answered use ``request.input_id``; they never read ``envelope_id``.

Timezone-awareness of ``now`` is deliberately not enforced here: the handlers
that consume it validate it with ``validate_timezone_aware_with_context`` so a
naive reading fails with the handler's own error context.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelInjectedRequest[TEvent: BaseModel](BaseModel):
    """The request a handler receives in place of an event envelope.

    Attributes:
        event: The domain event the handler decides on.
        now: The clock reading injected by the runtime for this dispatch.
        input_id: The id of the input envelope that carried the event.
        correlation_id: The correlation id from the input envelope, or ``None``
            when the input carried none.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    event: TEvent = Field(..., description="The domain event the handler decides on")
    now: datetime = Field(
        ..., description="Clock reading injected by the runtime for this dispatch"
    )
    input_id: UUID = Field(
        ..., description="Id of the input envelope that carried the event"
    )
    correlation_id: UUID | None = Field(
        default=None, description="Correlation id carried by the input envelope"
    )

    @classmethod
    def from_dispatch(
        cls, envelope: object, event: TEvent
    ) -> ModelInjectedRequest[TEvent]:
        """Unwrap a dispatch envelope into the request a handler receives.

        This is the runtime's unwrap step: the envelope is read here, at the
        dispatch boundary, and never reaches the handler. ``envelope`` is the
        materialized dispatch mapping or a ``ModelEventEnvelope``; its
        ``envelope_timestamp``, ``envelope_id`` and ``correlation_id`` become
        ``now``, ``input_id`` and ``correlation_id``. An envelope that carries no
        timestamp or id gets a fresh reading and id, as the runtime's typed
        envelope materialization does.

        Args:
            envelope: The dispatch envelope (mapping or object).
            event: The validated domain event the route delivers.

        Returns:
            The request for the handler, wrapping ``event``.
        """

        def read(name: str) -> object:
            if isinstance(envelope, Mapping):
                return envelope.get(name)
            return getattr(envelope, name, None)

        now = read("envelope_timestamp")
        if isinstance(now, str):
            now = datetime.fromisoformat(now.replace("Z", "+00:00"))
        input_id = read("envelope_id")
        correlation_id = read("correlation_id")
        return cls(
            event=event,
            now=now if isinstance(now, datetime) else datetime.now(UTC),
            input_id=UUID(str(input_id)) if input_id is not None else uuid4(),
            correlation_id=UUID(str(correlation_id)) if correlation_id else None,
        )


__all__ = ["ModelInjectedRequest"]
