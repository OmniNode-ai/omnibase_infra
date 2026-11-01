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
from datetime import datetime
from uuid import UUID

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

    @staticmethod
    def narrow(candidate: object) -> type[ModelInjectedRequest[BaseModel]] | None:
        """Return ``candidate`` when it is a request model class, else ``None``."""
        if isinstance(candidate, type) and issubclass(candidate, ModelInjectedRequest):
            return candidate
        return None

    @classmethod
    def from_dispatch(
        cls, envelope: object, event: TEvent
    ) -> ModelInjectedRequest[TEvent]:
        """Unwrap a dispatch envelope into the request a handler receives.

        This is the runtime's unwrap step: the envelope is read here, at the
        dispatch boundary, and never reaches the handler. ``envelope`` is the
        materialized dispatch mapping or a ``ModelEventEnvelope``; its
        ``envelope_timestamp``, ``envelope_id`` and ``correlation_id`` become
        ``now``, ``input_id`` and ``correlation_id``. Timestamp and input id are
        required: generating either here would change the identity of a replay.
        Malformed metadata fails validation at the dispatch boundary.

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
            try:
                now = datetime.fromisoformat(now)
            except ValueError as exc:
                raise ValueError(
                    "Dispatch envelope_timestamp must be an ISO datetime"
                ) from exc
        if not isinstance(now, datetime):
            raise ValueError(
                "Dispatch envelope_timestamp requires a datetime or ISO timestamp"
            )

        def read_uuid(name: str, *, optional: bool = False) -> UUID | None:
            value = read(name)
            if value is None and optional:
                return None
            if isinstance(value, UUID):
                return value
            if isinstance(value, str):
                try:
                    return UUID(value)
                except ValueError as exc:
                    raise ValueError(f"Dispatch {name} must be a UUID") from exc
            raise ValueError(f"Dispatch {name} must be a UUID")

        return cls.model_validate(
            {
                "event": event,
                "now": now,
                "input_id": read_uuid("envelope_id"),
                "correlation_id": read_uuid("correlation_id", optional=True),
            }
        )


__all__ = ["ModelInjectedRequest"]
