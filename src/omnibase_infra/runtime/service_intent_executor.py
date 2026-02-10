# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime-level intent executor for contract-driven intent routing.

This module provides the IntentExecutor, which routes intents
produced by handlers to the appropriate effect layer handlers. The routing
is driven by the contract's `intent_routing_table` section, not hardcoded
if/else chains.

Architecture:
    Handler returns intents -> DispatchResultApplier -> IntentExecutor
                                                        |-> resolve effect handler
                                                        |-> execute intent

    The executor resolves intent_type to a target handler via:
    1. Look up intent_type in the routing table
    2. Resolve the target handler from the DI container
    3. Execute the handler with the intent payload

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - ServiceDispatchResultApplier: Calls this executor for intents
    - ModelIntent: Intent envelope from omnibase_core
    - contract.yaml: intent_routing_table section

.. versionadded:: 0.7.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_core.models.reducer.model_intent import ModelIntent

logger = logging.getLogger(__name__)


class IntentExecutor:
    """Runtime-level intent executor with contract-driven routing.

    Routes intents from handlers to effect layer handlers based on the
    `intent_type` field of the intent payload. The routing table maps
    intent_type strings to effect handler callables.

    Thread Safety:
        This class is designed for single-threaded async use. Effect handlers
        handle their own concurrency concerns.

    Attributes:
        _container: ONEX container for handler resolution.
        _effect_handlers: Mapping of intent_type to async handler callables.

    Example:
        ```python
        executor = IntentExecutor(
            container=container,
            effect_handlers={
                "consul.register": consul_register_handler,
                "postgres.upsert_registration": postgres_upsert_handler,
            },
        )
        await executor.execute(intent, correlation_id)
        ```

    .. versionadded:: 0.7.0
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        effect_handlers: dict[str, object] | None = None,
    ) -> None:
        """Initialize the intent executor.

        Args:
            container: ONEX container for handler resolution.
            effect_handlers: Optional mapping of intent_type to handler objects.
                Each handler must implement an async `execute()` or `handle()`
                method that accepts the intent payload.
        """
        self._container = container
        self._effect_handlers: dict[str, object] = effect_handlers or {}

    def register_handler(self, intent_type: str, handler: object) -> None:
        """Register an effect handler for an intent type.

        Args:
            intent_type: The intent_type string to route (e.g., "consul.register").
            handler: Handler object with async execute() or handle() method.
        """
        self._effect_handlers[intent_type] = handler
        logger.debug(
            "Registered effect handler for intent_type=%s handler=%s",
            intent_type,
            type(handler).__name__,
        )

    async def execute(
        self,
        intent: ModelIntent,
        correlation_id: UUID | None = None,
    ) -> None:
        """Execute a single intent by routing to the appropriate effect handler.

        Args:
            intent: The intent to execute.
            correlation_id: Optional correlation ID for tracing.
        """
        # Extract intent_type from payload
        payload = intent.payload
        if payload is None:
            logger.warning(
                "Intent has no payload, skipping execution correlation_id=%s",
                str(correlation_id) if correlation_id else "none",
            )
            return

        # Get intent_type from payload (typed payload pattern)
        intent_type = getattr(payload, "intent_type", None)
        if intent_type is None:
            # Fall back to intent-level intent_type
            intent_type = intent.intent_type

        if intent_type is None:
            logger.warning(
                "Intent has no intent_type, skipping execution correlation_id=%s",
                str(correlation_id) if correlation_id else "none",
            )
            return

        handler = self._effect_handlers.get(intent_type)
        if handler is None:
            logger.warning(
                "No effect handler registered for intent_type=%s correlation_id=%s "
                "(intent skipped — possible misconfiguration)",
                intent_type,
                str(correlation_id) if correlation_id else "none",
            )
            return

        try:
            # Duck-type: try execute() first, then handle()
            execute_fn = getattr(handler, "execute", None)
            if execute_fn is not None and callable(execute_fn):
                await execute_fn(payload, correlation_id=correlation_id)
            else:
                handle_fn = getattr(handler, "handle", None)
                if handle_fn is not None and callable(handle_fn):
                    await handle_fn(payload, correlation_id=correlation_id)
                else:
                    logger.warning(
                        "Effect handler for intent_type=%s has no execute() or "
                        "handle() method correlation_id=%s",
                        intent_type,
                        str(correlation_id) if correlation_id else "none",
                    )
                    return

            logger.info(
                "Intent executed: intent_type=%s handler=%s correlation_id=%s",
                intent_type,
                type(handler).__name__,
                str(correlation_id) if correlation_id else "none",
            )

        except Exception as e:
            logger.warning(
                "Intent execution failed: intent_type=%s error=%s correlation_id=%s",
                intent_type,
                sanitize_error_message(e),
                str(correlation_id) if correlation_id else "none",
                extra={
                    "error_type": type(e).__name__,
                    "intent_type": intent_type,
                },
            )
            raise

    async def execute_all(
        self,
        intents: tuple[ModelIntent, ...] | list[ModelIntent],
        correlation_id: UUID | None = None,
    ) -> None:
        """Execute multiple intents sequentially.

        Intents are executed in order. If an intent fails, earlier intents
        that already executed (e.g., Consul register, PostgreSQL upsert)
        are **not** rolled back. The exception propagates to the caller,
        which prevents Kafka offset commit so the message will be redelivered.
        Effect adapters must therefore be idempotent.

        Args:
            intents: Sequence of intents to execute.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            Exception: Re-raised from the failing intent's effect handler.
                Earlier intents remain committed (no compensation/rollback).
        """
        for intent in intents:
            await self.execute(intent, correlation_id=correlation_id)


__all__: list[str] = ["IntentExecutor"]
