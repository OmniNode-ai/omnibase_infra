# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Gmail Archive Cleanup Effect — declarative effect node.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Triggered by onex.int.platform.runtime-tick.v1 (input_subscriptions)
    - Permanently deletes archived emails older than retention_days
    - Publishes a single summary event to
      onex.evt.omnibase_infra.gmail-archive-purged.v1
    - Lightweight shell — all logic in HandlerGmailArchiveCleanup

Design:
    - Zero custom routing in Python — contract.yaml governs all dispatch
    - Partition key = "gmail-archive-cleanup" (all events same partition)
    - Non-blocking: search/delete failures are collected, not raised

Related Tickets:
    - OMN-2731: Add node_gmail_archive_cleanup_effect
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect

__all__ = ["NodeGmailArchiveCleanupEffect"]


class NodeGmailArchiveCleanupEffect(NodeEffect):
    """Declarative effect node that purges aged emails from archive labels.

    Triggered by ``onex.int.platform.runtime-tick.v1`` (declared in
    contract.yaml ``input_subscriptions``). Permanently deletes messages
    from each configured archive label whose age exceeds ``retention_days``.
    Publishes one summary event to
    ``onex.evt.omnibase_infra.gmail-archive-purged.v1`` per run when any
    messages were deleted or errors occurred.

    All routing and execution logic is driven by contract.yaml.
    This class contains NO custom routing code.

    Partition Key:
        ``"gmail-archive-cleanup"`` — all cleanup events land in the same
        Kafka partition for ordered consumption.

    Supported Operations (defined in contract.yaml handler_routing):
        - ``gmail.purge_archive``: Search and delete aged archive messages.
    """

    # Pure declarative shell — all behavior defined in contract.yaml
