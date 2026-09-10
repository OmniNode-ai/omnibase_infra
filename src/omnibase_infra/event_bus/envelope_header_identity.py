# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bind wire-header identity and the causal edge to the envelope (OMN-18116).

``ModelEventHeaders`` defaults ``message_id`` and ``correlation_id`` to a fresh
``uuid4``. Every envelope publish seam built its headers without overriding
them, so the identity travelling on the wire was unrelated to the identity
inside the envelope it described.

That is not cosmetic. ``event_ledger`` -- the surface a chain replay reads back
-- populates its per-hop identity column from the ``message_id`` HEADER, not
from the envelope body (``HandlerLedgerProjection``). Measured read-only on the
dev lane over a two-hour window: of 737 ledger rows whose decoded body carried
an ``envelope_id``, the column matched the body on **0** and differed on
**737**. ``correlation_id`` agreed on 4151 rows only because OMN-14962 forced
that agreement, for correlation and nothing else.

A causal edge is checkable only if the identity it references is the identity
the reader reads back, so identity and edge are bound together, here, once. The
alternative -- repeating the rule at each seam -- is the drift the two separate
``ModelEventHeaders`` declarations in this codebase already demonstrate.

Absence is preserved rather than defaulted. A payload that is not an envelope
gets no keys back and keeps the model's own minting behaviour; an envelope with
no parent gets no ``parent_message_id``, which is the checkable statement that
the hop is a chain HEAD.
"""

from __future__ import annotations

from uuid import UUID

__all__ = ["header_identity_fields_from_envelope"]


def header_identity_fields_from_envelope(envelope: object) -> dict[str, UUID]:
    """Return the header fields an envelope determines, and only those.

    Args:
        envelope: The object about to be published. Anything without the
            attributes below contributes nothing, so non-envelope payloads are
            unaffected.

    Returns:
        A mapping suitable for splatting into ``ModelEventHeaders``. A key is
        present only when the envelope actually carries that value, so an
        absent parent never becomes an explicit ``None`` the model would have
        to interpret, and never becomes an invented UUID.
    """
    fields: dict[str, UUID] = {}
    envelope_id = getattr(envelope, "envelope_id", None)
    if isinstance(envelope_id, UUID):
        fields["message_id"] = envelope_id
    correlation_id = getattr(envelope, "correlation_id", None)
    if isinstance(correlation_id, UUID):
        fields["correlation_id"] = correlation_id
    parent_envelope_id = getattr(envelope, "parent_envelope_id", None)
    if isinstance(parent_envelope_id, UUID):
        fields["parent_message_id"] = parent_envelope_id
    return fields
