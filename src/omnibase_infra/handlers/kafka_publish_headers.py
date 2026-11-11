# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Wire headers for a Kafka publish that supplied none (OMN-18914).

Moved out of ``EventBusKafka`` unchanged in behaviour (OMN-21228): the plain and
the transactional publish both derive their headers here. Not an ONEX handler
(no ``handler_type``/``handler_category``); same arrangement as
``kafka_permanent_failure``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from uuid import UUID

from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders

logger = logging.getLogger(__name__)


def _uuid_or_none(value: object) -> UUID | None:
    """Return ``value`` as a UUID, or None if it is not one.

    A body is untrusted input. A field that is present but malformed must
    neither reach a UUID column nor fail a publish that would otherwise
    have succeeded, so this narrows rather than raises (OMN-18914 AC3).
    """
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def headers_for_unheadered_publish(
    topic: str,
    value: bytes,
    *,
    environment: str,
    minted_identity_topics: set[str],
) -> ModelEventHeaders:
    """Build wire headers for a caller that supplied none (OMN-18914).

    The identity on the wire is the identity in the BODY wherever the body
    states one. This is not a convenience: ``handler_ledger_projection``
    fills ``public.event_ledger``'s ``correlation_id`` and ``envelope_id``
    columns from these headers and never from the body, so a minted header
    files the record under a correlation no other hop shares. That is what
    left the delegation chain's head hop unjoinable and the chain canary
    reporting ``ledger_chain_incomplete`` -- measured on the dev lane,
    a delegation whose body said ``da706588-…`` reached the wire as
    ``3564c149-…``.

    Only IDENTITY is derived. ``event_type`` stays the topic and ``source``
    stays this bus's environment, deliberately: both already match every
    correctly-headered row in ``event_ledger``, no defect argues against
    them, and a body's own ``event_type`` is a different vocabulary (the
    gateway writes ``omnimarket.delegate-skill`` where the wire says
    ``onex.cmd.omnimarket.delegate-skill.v1``). Changing established
    columns with no defect behind the change is how a fix acquires a
    blast radius it was never measured for. ``ModelEventHeaders`` has no
    ``tenant_id`` field at all, so no tenant can be carried here; that half
    belongs to the gateway publisher (OMN-18915).

    Minting survives where there is nothing to derive, and says so once per
    topic -- a producer publishing without identity is a standing finding
    worth a line, and worth exactly one.
    """
    body: object = None
    try:
        body = json.loads(value)
    except (UnicodeDecodeError, ValueError):
        body = None

    correlation_id: UUID | None = None
    message_id: UUID | None = None
    if isinstance(body, dict):
        correlation_id = _uuid_or_none(body.get("correlation_id"))
        message_id = _uuid_or_none(body.get("envelope_id"))

    if correlation_id is None and topic not in minted_identity_topics:
        minted_identity_topics.add(topic)
        logger.warning(
            "Publishing to '%s' with no caller headers and no derivable "
            "identity in the body; minting a correlation id. Records from "
            "this producer cannot be joined to a delegation chain "
            "(OMN-18914).",
            topic,
            extra={"topic": topic, "environment": environment},
        )

    identity: dict[str, UUID] = {}
    if correlation_id is not None:
        identity["correlation_id"] = correlation_id
    if message_id is not None:
        identity["message_id"] = message_id

    return ModelEventHeaders(
        source=environment,
        event_type=topic,
        timestamp=datetime.now(UTC),
        **identity,
    )
