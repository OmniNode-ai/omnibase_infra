# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Derive the ONEX ``event_type`` routing key from a topic name.

Single canonical home for the ``onex.{kind}.{producer}.{event-name}.v{n}`` ->
``{producer}.{event-name}`` derivation (OMN-12116). Both the external
``DispatchResultApplier`` publish loop and the state_io in-row outbox publish
path (``_publish_outbox_batch``) call THIS function so they cannot diverge on
whether an emitted envelope carries ``event_type`` — the OMN-14743 defect was
exactly two independent envelope-build+publish sites where only the applier
stamped ``event_type`` (the outbox left it ``None``, so the routing reducer's
type-scoped dispatcher dropped the emission and delegation stalled at RECEIVED).

Pure: no I/O, no imports. Keep it that way so every runtime module can import it
without circular-import risk.
"""

from __future__ import annotations


def derive_event_type_from_topic(topic: str) -> str | None:
    """Derive the ``event_type`` routing key from an ONEX topic name.

    ONEX topics follow the convention::

        onex.{kind}.{producer}.{event-name}.v{n}

    This extracts ``{producer}.{event-name}`` as a dot-path routing key suitable
    for ``ModelEventEnvelope.event_type`` — the alias format dispatcher
    registration keys on (e.g. ``omnibase-infra.delegation-routing-request``).

    Args:
        topic: Full topic name following the ONEX naming convention
            (e.g. ``'onex.evt.omnimarket.swarm-endpoint-health-completed.v1'``).

    Returns:
        Derived event_type as ``'{producer}.{event-name}'``
        (e.g. ``'omnimarket.swarm-endpoint-health-completed'``), or ``None`` if
        the topic does not follow the expected ONEX format (at least 5
        dot-separated segments starting with ``onex``).

    .. versionadded:: OMN-12116 (originally on DispatchResultApplier); lifted to
        a shared helper under OMN-14743 so the outbox path stamps identically.
    """
    parts = topic.split(".")
    if len(parts) >= 5 and parts[0] == "onex":
        # onex.{kind}.{producer}.{event-name}.v{n}
        producer = parts[2]
        event_name = parts[3]
        return f"{producer}.{event_name}"
    return None
