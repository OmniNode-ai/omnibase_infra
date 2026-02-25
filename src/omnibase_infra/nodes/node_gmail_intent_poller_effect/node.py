# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Gmail Intent Poller Effect — declarative effect node for Gmail intent polling.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - On-demand trigger (no input_subscriptions)
    - Drains configured source labels and emits gmail-intent-received events
    - Lightweight shell — all logic in HandlerGmailIntentPoll

Design:
    - Zero custom routing in Python — contract.yaml governs all dispatch
    - Partition key = message_id (declared in contract.yaml)
    - Non-blocking: handler errors collected in result, not raised

Related Tickets:
    - OMN-2730: feat(omnibase_infra): add node_gmail_intent_poller_effect
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeGmailIntentPollerEffect(NodeEffect):
    """Declarative effect node that polls Gmail and emits intent events.

    On-demand trigger (no ``input_subscriptions`` in contract.yaml).
    Drains configured source labels, extracts URLs, and returns one
    ``gmail-intent-received`` event payload per email in
    ``pending_events`` for the runtime to publish to
    ``onex.evt.omnibase_infra.gmail-intent-received.v1``.

    All routing and execution logic is driven by contract.yaml.
    This class contains NO custom routing code.

    Partition Key:
        ``message_id`` — ensures all events for the same message land
        in the same Kafka partition for ordered consumption.

    Supported Operations (defined in contract.yaml handler_routing):
        - ``gmail.poll_inbox``: Drain source labels and emit intent events.
    """

    # Pure declarative shell — all behavior defined in contract.yaml


__all__ = ["NodeGmailIntentPollerEffect"]
