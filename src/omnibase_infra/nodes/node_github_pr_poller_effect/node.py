# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node GitHub PR Poller Effect — declarative effect node for GitHub PR polling.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Triggered by onex.evt.runtime.tick.v1 (input_subscriptions)
    - Produces ModelGitHubPRStatusEvent to onex.evt.github.pr-status.v1
    - Lightweight shell — all logic in HandlerGitHubApiPoll

Design:
    - Zero custom routing in Python — contract.yaml governs all dispatch
    - Partition key = "{repo}:{pr_number}" (declared in contract.yaml)
    - Non-blocking: handler errors are collected, not raised

Related Tickets:
    - OMN-2656: Phase 2 — Effect Nodes & CLIs (omnibase_infra)
    - OMN-2655: Core event models and SPI contracts
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeGitHubPRPollerEffect(NodeEffect):
    """Declarative effect node that polls GitHub for PR triage state.

    Triggered by ``onex.evt.runtime.tick.v1`` (declared in contract.yaml
    ``input_subscriptions``). Publishes one ``ModelGitHubPRStatusEvent``
    per open PR to ``onex.evt.github.pr-status.v1``.

    All routing and execution logic is driven by contract.yaml.
    This class contains NO custom routing code.

    Partition Key:
        ``{repo}:{pr_number}`` — ensures all events for the same PR land
        in the same Kafka partition for ordered consumption.

    Supported Operations (defined in contract.yaml handler_routing):
        - ``github.poll.prs``: Poll GitHub API and emit status events.
    """

    # Pure declarative shell — all behavior defined in contract.yaml


__all__ = ["NodeGitHubPRPollerEffect"]
