# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Slack Alerter Effect - Declarative effect node for Slack alerting.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom routing logic - all behavior from handler_routing
    - Lightweight shell that delegates to handlers via container resolution
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for infrastructure I/O operations.
All handler routing is 100% driven by contract.yaml, not Python code.

Handler Routing Pattern:
    1. Receive alert request (input_model in contract)
    2. Route to appropriate handler based on operation (handler_routing)
    3. Execute infrastructure I/O via handler (Slack webhook)
    4. Return structured response (output_model in contract)

Design Decisions:
    - 100% Contract-Driven: All routing logic in YAML, not Python
    - Zero Custom Routing: Base class handles handler dispatch via contract
    - Declarative Handlers: handler_routing section defines dispatch rules
    - Container DI: Handler dependencies resolved via container

Node Responsibilities:
    - Define I/O model contract (ModelSlackAlert -> ModelSlackAlertResult)
    - Delegate all execution to handlers via base class
    - NO custom logic - pure declarative shell

The actual handler execution and routing is performed by:
    - Direct handler invocation by callers
    - Or orchestrator layer for workflow coordination

Handlers receive their dependencies directly via constructor injection:
    - HandlerSlackWebhook(webhook_url, http_session)

Coroutine Safety:
    This node is async-safe. Handler coordination is performed by the
    caller or orchestrator layer, not by this effect node.

Related Modules:
    - contract.yaml: Handler routing and I/O model definitions
    - ../../handlers/handler_slack_webhook.py: Webhook handler implementation
    - ../../handlers/models/model_slack_alert.py: Alert payload models

Related Tickets:
    - OMN-1905: Add declarative Slack webhook handler to omnibase_infra
"""

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
        dependencies (webhook_url from env, optional http_session).
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
