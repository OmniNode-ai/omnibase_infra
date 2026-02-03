# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract registration event router.

Routes Kafka messages to ContractRegistryReducer and executes resulting intents.

This module provides an extracted event router for routing contract lifecycle
events (registration, deregistration, heartbeat) to the ContractRegistryReducer.
The router also runs an internal tick timer for periodic staleness computation.

Design:
    This class encapsulates the message routing logic for contract registry
    projection. By extracting it, we enable:
    - Unit testing without full kernel bootstrap
    - Mocking of dependencies for isolation
    - Clearer separation between bootstrap and event routing

    The router uses ProtocolEventBusLike for event publishing, enabling
    duck typing with any event bus implementation (Kafka, InMemory, etc.).

Message Flow:
    1. Parse message as ModelEventEnvelope[dict]
    2. Validate payload as event type (ModelContractRegisteredEvent,
       ModelContractDeregisteredEvent, ModelNodeHeartbeatEvent)
    3. Call reducer.reduce(state, event, metadata)
    4. Execute returned intents via _execute_intents()
    5. Log errors (no exceptions raised to consumer)

Related:
    - OMN-1869: Wire ServiceKernel to Kafka event bus
    - IntrospectionEventRouter: Reference implementation for event routing
    - ContractRegistryReducer: Pure reducer handling contract events
"""

from __future__ import annotations

__all__ = ["ContractRegistrationEventRouter"]

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError

from omnibase_core.models.events import (
    ModelContractDeregisteredEvent,
    ModelContractRegisteredEvent,
    ModelNodeHeartbeatEvent,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.types import JsonType
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.contract_registry_reducer.models.model_contract_registry_state import (
    ModelContractRegistryState,
)
from omnibase_infra.nodes.contract_registry_reducer.reducer import (
    ContractRegistryEvent,
    ContractRegistryReducer,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_core.models.reducer.model_intent import ModelIntent
    from omnibase_infra.protocols import ProtocolEventBusLike

logger = logging.getLogger(__name__)

# Minimum tick interval to prevent excessive CPU usage
MIN_TICK_INTERVAL_SECONDS = 5

# Scheduler ID for tick events from this router
ROUTER_SCHEDULER_ID = "contract-registration-event-router"


class ProtocolIntentEffect(Protocol):
    """Protocol for intent effect executors.

    Intent effects are responsible for executing side effects (e.g., PostgreSQL
    writes) based on intents emitted by the reducer. Each effect executor is keyed
    by the payload's intent_type field (e.g., "postgres.upsert_contract").
    """

    async def handle(self, payload: object, correlation_id: UUID) -> object:
        """Execute the intent.

        Args:
            payload: The typed payload model (e.g., ModelPayloadUpsertContract).
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Result from the effect executor (typically ModelBackendResult).
        """
        ...


class ContractRegistrationEventRouter:
    """Routes contract lifecycle events to reducer and executes intents.

    This router handles incoming event messages from Kafka, parses them as
    contract lifecycle events, and routes them to the ContractRegistryReducer.
    It also maintains an internal tick timer for periodic staleness computation.

    The router propagates correlation IDs from incoming messages for
    distributed tracing. If no correlation ID is present, it generates
    a new one to ensure all operations can be traced.

    This class follows the container-based dependency injection pattern,
    receiving a ModelONEXContainer for service resolution while also
    accepting explicit dependencies for router-specific configuration.

    Message Flow:
        1. Parse message as ModelEventEnvelope[dict]
        2. Validate payload as event type (ModelContractRegisteredEvent, etc.)
        3. Call reducer.reduce(state, event, metadata)
        4. Execute returned intents via _execute_intents()
        5. Log errors (no exceptions raised to consumer)

    Tick Timer:
        The router runs an internal tick timer at configurable intervals
        (default 60s, minimum 5s). Each tick emits a ModelRuntimeTick event
        to the reducer for staleness computation.

    Attributes:
        _container: ONEX service container for dependency resolution.
        _reducer: The ContractRegistryReducer to route events to.
        _effect_handlers: Dict mapping intent_type to handler instances.
        _event_bus: Event bus implementing ProtocolEventBusLike (optional).
        _tick_interval_seconds: Interval between staleness ticks.
        _state: Current reducer state (mutable, updated after each reduction).
        _shutdown_event: Event for graceful shutdown of tick loop.
        _tick_task: Background task for tick loop.
        _tick_sequence: Monotonically increasing counter for tick events.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> router = ContractRegistrationEventRouter(
        ...     container=container,
        ...     reducer=reducer,
        ...     effect_handlers={"postgres.upsert_contract": upsert_handler},
        ...     event_bus=event_bus,
        ...     tick_interval_seconds=60,
        ... )
        >>> await router.start()
        >>> # Use as callback for event bus subscription
        >>> await event_bus.subscribe(
        ...     topic="contract-registered",
        ...     group_id="contract-registry",
        ...     on_message=router.handle_message,
        ... )

    See Also:
        - IntrospectionEventRouter: Reference implementation for event routing
        - ContractRegistryReducer: Pure reducer for contract events
        - docs/patterns/container_dependency_injection.md for DI patterns
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        reducer: ContractRegistryReducer,
        effect_handlers: dict[str, ProtocolIntentEffect],
        event_bus: ProtocolEventBusLike | None = None,
        tick_interval_seconds: int = 60,
    ) -> None:
        """Initialize ContractRegistrationEventRouter with container-based DI.

        Follows the ONEX container-based DI pattern where the container is passed
        as the first parameter for service resolution, with additional explicit
        parameters for router-specific configuration.

        Args:
            container: ONEX service container for dependency resolution. Provides
                access to service_registry for resolving shared services.
            reducer: The ContractRegistryReducer to route events to.
            effect_handlers: Dict mapping intent_type (e.g., "postgres.upsert_contract")
                to handler instances that implement ProtocolIntentEffect.
            event_bus: Event bus implementing ProtocolEventBusLike for publishing
                (optional, only needed if router publishes output events).
            tick_interval_seconds: Interval between staleness tick events.
                Clamped to minimum of 5 seconds to prevent excessive CPU usage.

        Example:
            >>> from omnibase_core.container import ModelONEXContainer
            >>> container = ModelONEXContainer()
            >>> router = ContractRegistrationEventRouter(
            ...     container=container,
            ...     reducer=reducer,
            ...     effect_handlers={"postgres.upsert_contract": handler},
            ...     tick_interval_seconds=60,
            ... )

        See Also:
            - docs/patterns/container_dependency_injection.md for DI patterns.
        """
        self._container = container
        self._reducer = reducer
        self._effect_handlers = effect_handlers
        self._event_bus = event_bus
        # Clamp tick interval to minimum of 5 seconds
        self._tick_interval_seconds = max(
            MIN_TICK_INTERVAL_SECONDS, tick_interval_seconds
        )
        self._state: ModelContractRegistryState = ModelContractRegistryState()
        self._shutdown_event = asyncio.Event()
        self._tick_task: asyncio.Task[None] | None = None
        self._tick_sequence: int = 0

        logger.debug(
            "ContractRegistrationEventRouter initialized",
            extra={
                "tick_interval_seconds": self._tick_interval_seconds,
                "handler_count": len(self._effect_handlers),
                "handlers": list(self._effect_handlers.keys()),
            },
        )

    @property
    def container(self) -> ModelONEXContainer:
        """Return the ONEX service container.

        The ModelONEXContainer provides protocol-based service resolution.

        Returns:
            The ModelONEXContainer instance passed during initialization.
        """
        return self._container

    @property
    def state(self) -> ModelContractRegistryState:
        """Return the current reducer state.

        Returns:
            Current immutable state of the contract registry reducer.
        """
        return self._state

    @property
    def tick_interval_seconds(self) -> int:
        """Return the configured tick interval.

        Returns:
            Tick interval in seconds (clamped to minimum 5s).
        """
        return self._tick_interval_seconds

    async def start(self) -> None:
        """Start internal tick timer.

        Starts a background task that periodically emits ModelRuntimeTick
        events to the reducer for staleness computation.
        """
        self._shutdown_event.clear()
        self._tick_task = asyncio.create_task(self._tick_loop())
        logger.info(
            "ContractRegistrationEventRouter started with tick_interval=%ds",
            self._tick_interval_seconds,
        )

    async def stop(self) -> None:
        """Stop tick timer.

        Signals the tick loop to stop and waits for it to complete.
        """
        self._shutdown_event.set()
        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None
        logger.info("ContractRegistrationEventRouter stopped")

    def _extract_correlation_id_from_message(self, msg: ModelEventMessage) -> UUID:
        """Extract correlation ID from message headers or generate new one.

        Attempts to extract the correlation_id from message headers to ensure
        proper propagation for distributed tracing. Falls back to generating
        a new UUID if no correlation ID is found.

        Uses duck-typing patterns for type detection instead of isinstance checks
        to align with protocol-based design principles.

        Args:
            msg: The incoming event message.

        Returns:
            UUID: The extracted or generated correlation ID.
        """
        # Try to extract from message headers if available
        if hasattr(msg, "headers") and msg.headers is not None:
            headers = msg.headers
            if (
                hasattr(headers, "correlation_id")
                and headers.correlation_id is not None
            ):
                try:
                    correlation_id = headers.correlation_id
                    # Check for bytes-like (has decode method) - duck typing
                    if hasattr(correlation_id, "decode"):
                        correlation_id = correlation_id.decode("utf-8")
                    return UUID(str(correlation_id))
                except (ValueError, TypeError, UnicodeDecodeError, AttributeError):
                    pass  # Fall through to try payload extraction

        # If we can peek at the payload, try to extract correlation_id
        try:
            if msg.value is not None:
                # Duck-type: check for decode method (bytes-like) first
                if hasattr(msg.value, "decode"):
                    payload_dict = json.loads(msg.value.decode("utf-8"))
                else:
                    try:
                        payload_dict = json.loads(msg.value)
                    except TypeError:
                        payload_dict = msg.value

                if payload_dict:
                    # Check envelope-level correlation_id first
                    if "correlation_id" in payload_dict:
                        return UUID(str(payload_dict["correlation_id"]))
                    # Check payload-level correlation_id
                    payload_content = payload_dict.get("payload")
                    if payload_content and hasattr(payload_content, "get"):
                        nested_corr_id = payload_content.get("correlation_id")
                        if nested_corr_id is not None:
                            return UUID(str(nested_corr_id))
        except (json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError):
            pass

        # Generate new correlation ID as last resort
        return uuid4()

    async def handle_message(self, msg: ModelEventMessage) -> None:
        """Route Kafka message to reducer, execute intents.

        This callback is invoked for each message received on contract topics.
        It parses the raw JSON payload as a contract lifecycle event and routes
        it to the ContractRegistryReducer for processing.

        The method propagates the correlation_id from the incoming message
        for distributed tracing. If no correlation_id is present in the message,
        a new one is generated.

        Error Handling:
            Errors are logged but not raised to the consumer. This ensures
            message processing continues even if individual messages fail.
            Failed messages should be handled via dead-letter queue (DLQ)
            at the event bus level.

        Args:
            msg: ModelEventMessage from Kafka consumer containing the serialized
                event envelope in .value field.
        """
        callback_correlation_id = self._extract_correlation_id_from_message(msg)
        callback_start_time = time.time()

        logger.debug(
            "Contract event message callback invoked (correlation_id=%s)",
            callback_correlation_id,
            extra={
                "message_offset": getattr(msg, "offset", None),
                "message_partition": getattr(msg, "partition", None),
                "message_topic": getattr(msg, "topic", None),
            },
        )

        try:
            # ModelEventMessage has .value as bytes
            if msg.value is None:
                logger.debug(
                    "Message value is None, skipping (correlation_id=%s)",
                    callback_correlation_id,
                )
                return

            # Parse message value using duck-typing patterns
            if hasattr(msg.value, "decode"):
                payload_dict = json.loads(msg.value.decode("utf-8"))
            else:
                try:
                    payload_dict = json.loads(msg.value)
                except TypeError:
                    if hasattr(msg.value, "keys"):
                        payload_dict = msg.value
                    else:
                        logger.debug(
                            "Unexpected message value type: %s (correlation_id=%s)",
                            type(msg.value).__name__,
                            callback_correlation_id,
                        )
                        return

            # Parse as ModelEventEnvelope containing contract event
            raw_envelope = ModelEventEnvelope[dict].model_validate(payload_dict)

            # Try to validate payload as one of the contract event types
            event: ContractRegistryEvent | None = None
            event_type_name: str = ""

            # Try ModelContractRegisteredEvent first
            try:
                event = ModelContractRegisteredEvent.model_validate(
                    raw_envelope.payload
                )
                event_type_name = "ContractRegisteredEvent"
            except ValidationError:
                pass

            # Try ModelContractDeregisteredEvent
            if event is None:
                try:
                    event = ModelContractDeregisteredEvent.model_validate(
                        raw_envelope.payload
                    )
                    event_type_name = "ContractDeregisteredEvent"
                except ValidationError:
                    pass

            # Try ModelNodeHeartbeatEvent
            if event is None:
                try:
                    event = ModelNodeHeartbeatEvent.model_validate(raw_envelope.payload)
                    event_type_name = "NodeHeartbeatEvent"
                except ValidationError:
                    pass

            if event is None:
                # Not a recognized contract event - skip silently
                logger.debug(
                    "Message is not a valid contract event, skipping (correlation_id=%s)",
                    callback_correlation_id,
                )
                return

            logger.info(
                "Parsed %s (correlation_id=%s)",
                event_type_name,
                callback_correlation_id,
                extra={
                    "envelope_id": str(raw_envelope.envelope_id),
                    "event_type": event_type_name,
                },
            )

            # Build event metadata from Kafka message
            event_metadata: dict[str, JsonType] = {
                "topic": msg.topic,
                "partition": msg.partition or 0,
                "offset": int(msg.offset) if msg.offset else 0,
            }

            # Call reducer
            reducer_start_time = time.time()
            output = self._reducer.reduce(self._state, event, event_metadata)
            reducer_duration = time.time() - reducer_start_time

            # Update state
            self._state = output.result

            logger.info(
                "Reducer processed %s in %.3fs (correlation_id=%s)",
                event_type_name,
                reducer_duration,
                callback_correlation_id,
                extra={
                    "intents_count": len(output.intents),
                    "processing_time_ms": output.processing_time_ms,
                },
            )

            # Execute intents
            if output.intents:
                await self._execute_intents(output.intents, callback_correlation_id)

        except ValidationError as validation_error:
            logger.debug(
                "Message validation failed, skipping (correlation_id=%s)",
                callback_correlation_id,
                extra={
                    "validation_error_count": validation_error.error_count(),
                },
            )

        except json.JSONDecodeError as json_error:
            logger.warning(
                "Failed to decode JSON from message: %s (correlation_id=%s)",
                sanitize_error_message(json_error),
                callback_correlation_id,
                extra={
                    "error_type": type(json_error).__name__,
                    "error_position": getattr(json_error, "pos", None),
                },
            )

        except Exception as msg_error:
            # Use warning instead of exception to avoid credential exposure
            logger.warning(
                "Failed to process contract message: %s (correlation_id=%s)",
                sanitize_error_message(msg_error),
                callback_correlation_id,
                extra={
                    "error_type": type(msg_error).__name__,
                },
            )

        finally:
            callback_duration = time.time() - callback_start_time
            logger.debug(
                "Contract message callback completed in %.3fs (correlation_id=%s)",
                callback_duration,
                callback_correlation_id,
                extra={
                    "callback_duration_seconds": callback_duration,
                },
            )

    async def _execute_intents(
        self,
        intents: tuple[ModelIntent, ...],
        correlation_id: UUID,
    ) -> None:
        """Dispatch intents to effect handlers by payload.intent_type.

        Each intent has a payload with an intent_type field (e.g.,
        "postgres.upsert_contract"). This method looks up the appropriate
        handler and executes it.

        Error Handling:
            Errors are logged but not raised. If a handler fails, we continue
            processing remaining intents.

        Args:
            intents: Tuple of ModelIntent objects from the reducer.
            correlation_id: Correlation ID for distributed tracing.
        """
        for intent in intents:
            try:
                # Extract intent_type from payload
                payload = intent.payload
                intent_type = getattr(payload, "intent_type", None)

                if intent_type is None:
                    logger.warning(
                        "Intent payload missing intent_type field (correlation_id=%s)",
                        correlation_id,
                        extra={"intent_target": intent.target},
                    )
                    continue

                if intent_type not in self._effect_handlers:
                    logger.warning(
                        "No handler for intent_type: %s (correlation_id=%s)",
                        intent_type,
                        correlation_id,
                        extra={
                            "available_handlers": list(self._effect_handlers.keys()),
                            "intent_target": intent.target,
                        },
                    )
                    continue

                handler = self._effect_handlers[intent_type]
                handler_start_time = time.time()

                # Extract correlation_id from payload if available
                payload_correlation_id = getattr(
                    payload, "correlation_id", correlation_id
                )

                result = await handler.handle(payload, payload_correlation_id)
                handler_duration = time.time() - handler_start_time

                # Check if handler returned a result with success field
                success = getattr(result, "success", True) if result else True

                if success:
                    logger.debug(
                        "Intent %s executed successfully in %.3fs (correlation_id=%s)",
                        intent_type,
                        handler_duration,
                        correlation_id,
                    )
                else:
                    error_msg = getattr(result, "error", "Unknown error")
                    logger.warning(
                        "Intent %s failed: %s (correlation_id=%s)",
                        intent_type,
                        error_msg,
                        correlation_id,
                        extra={
                            "handler_duration_seconds": handler_duration,
                            "intent_target": intent.target,
                        },
                    )

            except Exception as e:
                logger.warning(
                    "Error executing intent: %s (correlation_id=%s)",
                    sanitize_error_message(e),
                    correlation_id,
                    extra={
                        "error_type": type(e).__name__,
                        "intent_type": getattr(
                            intent.payload, "intent_type", "unknown"
                        ),
                    },
                )

    async def _tick_loop(self) -> None:
        """Periodic tick for staleness computation.

        Runs continuously until shutdown is signaled. Each tick:
        1. Creates a ModelRuntimeTick event
        2. Calls reducer.reduce(state, tick, metadata)
        3. Executes resulting intents (typically postgres.mark_stale)

        The tick interval is configurable (default 60s, minimum 5s).
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._tick_interval_seconds)

                if self._shutdown_event.is_set():
                    break

                # Increment sequence for each tick
                self._tick_sequence += 1
                tick_correlation_id = uuid4()
                now = datetime.now(UTC)

                tick = ModelRuntimeTick(
                    tick_id=uuid4(),
                    now=now,
                    sequence_number=self._tick_sequence,
                    scheduled_at=now,
                    correlation_id=tick_correlation_id,
                    scheduler_id=ROUTER_SCHEDULER_ID,
                    tick_interval_ms=self._tick_interval_seconds * 1000,
                )

                # Internal tick metadata - not from Kafka
                metadata: dict[str, JsonType] = {
                    "topic": "__internal_tick__",
                    "partition": 0,
                    "offset": self._tick_sequence,
                }

                tick_start_time = time.time()
                output = self._reducer.reduce(self._state, tick, metadata)
                tick_duration = time.time() - tick_start_time

                # Update state
                self._state = output.result

                logger.debug(
                    "Tick processed in %.3fs, emitted %d intents (correlation_id=%s)",
                    tick_duration,
                    len(output.intents),
                    tick_correlation_id,
                    extra={
                        "tick_sequence": self._tick_sequence,
                        "processing_time_ms": output.processing_time_ms,
                    },
                )

                # Execute intents (typically postgres.mark_stale)
                if output.intents:
                    await self._execute_intents(output.intents, tick_correlation_id)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.warning(
                    "Error in tick loop: %s",
                    sanitize_error_message(e),
                    extra={
                        "error_type": type(e).__name__,
                        "tick_sequence": self._tick_sequence,
                    },
                )
