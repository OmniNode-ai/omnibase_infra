# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Slack Alerter Effect Node - Declarative Slack webhook alerting.

This module exports the declarative NodeSlackAlerterEffect for sending
infrastructure alerts to Slack via webhooks.

Architecture:
    This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom routing logic - all behavior from handler_routing
    - Lightweight shell that delegates to HandlerSlackWebhook
    - Pattern: "Contract-driven, handlers wired externally"

Example:
    >>> from omnibase_core.models.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.node_slack_alerter_effect import NodeSlackAlerterEffect
    >>> from omnibase_infra.handlers import HandlerSlackWebhook
    >>>
    >>> container = ModelONEXContainer()
    >>> node = NodeSlackAlerterEffect(container)
    >>>
    >>> # Handler receives dependencies via constructor
    >>> handler = HandlerSlackWebhook()
    >>> # result = await handler.handle(alert)

Related Tickets:
    - OMN-1905: Add declarative Slack webhook handler to omnibase_infra
"""

from omnibase_infra.nodes.node_slack_alerter_effect.node import NodeSlackAlerterEffect

__all__ = ["NodeSlackAlerterEffect"]
