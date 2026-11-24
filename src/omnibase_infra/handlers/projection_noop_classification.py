# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Did a projection handler attest that its zero-row return was designed? (OMN-26410)

The projection dispatch callback logs ERROR when a handler returns zero rows and
no terminal is owed, because that is how a lost write looks. A handler that has
nothing to write on purpose (the runtime tick of a writer that republishes a
snapshot once per interval, an event outside the projection's scope) returned
the same ``rows_upserted: 0``, so every such delivery read as a failed write: the
EvalResults writer logged the ERROR about once a second on the dev runtime.

This is a closed allowlist, in the direction ``projection_failure_classification``
and the ordering-guard count already take: only a result that positively carries
a known reason, for a delivery that reason can apply to, is an expected no-op.
Anything else -- no key, an unknown or mistyped value, a tick attestation on a
source topic -- is not attested and keeps the ERROR.

The key is optional by construction, as ``ROWS_REFUSED_KEY`` is: a handler that
omits it is read exactly as before, so a writer in ``omnimarket`` may send it
before or after this lands. The writer restates the key and the two values
rather than importing them.
"""

from __future__ import annotations

from typing import Final

from omnibase_infra.enums.enum_projection_noop_reason import EnumProjectionNoopReason
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_RUNTIME_TICK

#: OMN-18992. The optional field a projection handler uses to say "I wrote
#: nothing ON PURPOSE, and here is how many rows an ordering guard refused".
#: Optional by construction: a handler omitting it is read exactly as it is
#: today, which is what makes this consumer-first across the repo boundary.
ROWS_REFUSED_KEY: Final[str] = "rows_refused_by_ordering_guard"


#: The result key a handler uses to attest why it wrote zero rows on purpose.
NOOP_REASON_KEY: Final[str] = "projection_noop_reason"


def read_attested_noop_reason(
    result: object, *, topic: str | None
) -> EnumProjectionNoopReason | None:
    """The reason a handler attested for a zero-row return, or ``None`` if it did not.

    ``None`` keeps the zero-row ERROR. A value that is not exactly one of the
    enum's wire strings is not attested: no case folding, no trimming, no
    coercion of a non-string.
    """
    if not isinstance(result, dict):
        return None
    raw = result.get(NOOP_REASON_KEY)
    if type(raw) is not str:
        return None
    try:
        reason = EnumProjectionNoopReason(raw)
    except ValueError:
        return None
    if reason is EnumProjectionNoopReason.TICK and topic != SUFFIX_RUNTIME_TICK:
        return None
    return reason


def is_windowless_consumer_flow_heartbeat(
    *, contract_name: str, event_type: str, payload: object, result: object
) -> bool:
    """Recognize the consumer-flow contract's deliberate empty heartbeat.

    OMN-18992 / B23. This contract projects ``flow_window``, not node liveness.
    Only its new, explicit zero/refusal result shape can attest the no-op;
    legacy or malformed results keep the zero-write ERROR. Other heartbeat
    projections may still owe rows, so event type alone cannot exempt them.
    """
    return (
        contract_name == "projection_consumer_flow"
        and event_type == "heartbeat"
        and isinstance(payload, dict)
        and payload.get("flow_window") is None
        and isinstance(result, dict)
        and type(result.get("rows_upserted")) is int
        and result["rows_upserted"] == 0
        and result.get("flow_rows") == []
        and type(result.get(ROWS_REFUSED_KEY)) is int
        and result[ROWS_REFUSED_KEY] == 0
    )
