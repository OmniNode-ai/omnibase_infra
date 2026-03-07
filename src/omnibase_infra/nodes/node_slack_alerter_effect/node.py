# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Slack Alerter Effect — declarative effect node for Slack alerting (OMN-1905)."""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeSlackAlerterEffect(NodeEffect):
    """Declarative effect node for Slack webhook alerting.

    This effect node is a lightweight shell that defines the I/O contract
    for Slack alert operations. All routing and execution logic is driven
    by contract.yaml - this class contains NO custom routing code.

    Supported Operations (defined in contract.yaml handler_routing):
        - send_alert: Send a formatted alert to Slack
        - send_message: Send a plain text message to Slack

    Dependency Injection:
        The HandlerSlackWebhook is instantiated by callers with its
        dependencies (bot_token from env, optional http_session).
        This node contains NO instance variables for the handler.

    Example:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_slack_alerter_effect import NodeSlackAlerterEffect
        from omnibase_infra.handlers import HandlerSlackWebhook
        from omnibase_infra.handlers.models import ModelSlackAlert, EnumAlertSeverity

        # Create effect node via container
        container = ModelONEXContainer()
        effect = NodeSlackAlerterEffect(container)

        # Handler receives dependencies directly via constructor
        handler = HandlerSlackWebhook()

        # Create and send alert
        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.ERROR,
            message="Circuit breaker opened",
            title="Infrastructure Alert",
            details={"service": "consul", "threshold": "5"},
        )
        result = await handler.handle(alert)

        if result.success:
            print(f"Alert delivered in {result.duration_ms}ms")
        else:
            print(f"Alert failed: {result.error}")
        ```
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeSlackAlerterEffect"]
