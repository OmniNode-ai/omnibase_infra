# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One builder and one probe for the source message's delivery coordinates.

OMN-18918. THREE consume boundaries reach a dispatch engine in this runtime,
not one: ``EventBusSubcontractWiring``'s callback, and the two in
``handler_wiring`` -- ``_make_event_bus_callback`` and
``_make_raw_event_projection_callback``. The five in-process projection
writers, the ones the OMN-18905 defect is actually about, arrive on the
handler_wiring pair and never touch the subcontract seam; a live subscription
readback on the .201 dev lane showed that seam dispatching 104 calls across
four topics, none of them a projection source.

So the builder lives here rather than beside any one of them. Duplicating it
per boundary is how two of the three end up subtly different, and putting it
in one boundary's module makes the other two import from a peer for no reason.

The probe is here for the same reason and answers the same question at both
protocols: ``delivery`` is OPTIONAL on ``ProtocolDispatchEngine`` and on
``ProtocolContractScopedDispatchEngine``, so an implementor written before it
is still valid and still has the old signature. A caller that passes the
keyword unconditionally turns that optionality into a ``TypeError`` on the
first message.
"""

from __future__ import annotations

import functools
import inspect

from omnibase_core.models.dispatch.model_message_delivery_context import (
    ModelMessageDeliveryContext,
)
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage

__all__ = [
    "delivery_context_from_message",
    "engine_type_accepts_delivery",
]


@functools.lru_cache(maxsize=128)
def engine_type_accepts_delivery(engine_type: type, method_name: str) -> bool:
    """Does this engine's named dispatch method declare keyword-only ``delivery``?

    OMN-18918. ``delivery`` is OPTIONAL on both dispatch protocols, which
    means every implementor written before it -- in this repo, in a consumer
    repo, and in a test double -- is still a valid implementor and still has
    the older signature. A caller that passes the keyword unconditionally
    converts that optionality into a ``TypeError`` at the first message,
    which is a consumer break dressed as an additive change. Two engines in
    this repo's own remote-agent integration test failed exactly that way.

    So the caller probes, the same way the engine probes its dispatchers. The
    name must match AND be keyword-only: a positional match would be
    ambiguous with ``topic`` and ``envelope``.

    ``method_name`` is a parameter because the two protocols name their entry
    differently -- ``dispatch`` on the process-global one, ``dispatch_scoped``
    on the contract-scoped one -- and a probe hardcoded to one of them would
    silently answer False for the other, which reads exactly like a legacy
    engine and injects nothing.

    Cached on the engine CLASS, not the instance: the signature is a property
    of the type, and the probe must not run per message.

    An uninspectable engine returns False and is called exactly as it is
    today. Unknown refuses, which here means "changes nothing".
    """
    try:
        method = getattr(engine_type, method_name)
        parameter = inspect.signature(method).parameters.get("delivery")
    except (ValueError, TypeError, AttributeError):
        return False
    return parameter is not None and parameter.kind is inspect.Parameter.KEYWORD_ONLY


def delivery_context_from_message(
    message: object, topic: str
) -> ModelMessageDeliveryContext | None:
    """Build the delivery coordinates for one consumed message, or ``None``.

    OMN-18918. ``ProtocolEventMessage`` in ``omnibase_core`` -- the type this
    loop is written against -- declares ``topic``, ``key``, ``value``,
    ``headers``, ``ack`` and ``nack``, and NO delivery coordinates at all.
    The concrete object the Kafka bus delivers is
    ``omnibase_infra``'s ``ModelEventMessage``, which does carry them
    (``event_bus_kafka.py`` builds it straight from the consumed record), so
    this narrows to that model rather than widening a shared protocol in a
    lower layer for one consumer. An implementation that is not that model --
    the in-memory bus, a test double -- yields ``None`` and is handled by the
    same refusal as a missing coordinate.

    Both fields are optional on that model and the offset is text, so either
    one absent, or an offset that does not parse as a non-negative integer,
    yields ``None``.

    ``None`` rather than a zero, deliberately. A fabricated coordinate is
    indistinguishable downstream from a measured one, and that is precisely
    the defect this ticket exists to end: every in-process projection writer
    published its snapshot deltas at offset 0, the serving cache refuses a
    delta whose offset does not exceed the one it holds for that key, and so
    each key froze on its first value -- at zero consumer lag, behind a green
    readiness endpoint (OMN-18905). A consumer that cannot say where a
    message came from must say nothing, and the projection seam refuses
    loudly rather than inventing a coordinate.

    No broker timestamp: the protocol does not carry one, and the model's
    field is optional for exactly this reason rather than being defaulted to
    a clock reading here.
    """
    if not isinstance(message, ModelEventMessage):
        return None
    partition = message.partition
    raw_offset = message.offset
    if partition is None or raw_offset is None:
        return None
    try:
        offset = int(raw_offset)
    except (TypeError, ValueError):
        return None
    if offset < 0 or partition < 0:
        return None
    return ModelMessageDeliveryContext(topic=topic, partition=partition, offset=offset)
