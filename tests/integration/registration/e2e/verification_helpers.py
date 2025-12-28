# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Verification helpers for E2E registration tests.

Provides async helpers to verify registration data persistence across
infrastructure services (Consul, PostgreSQL, Kafka) and validate state
transitions in the registration workflow.

Architecture:
    Uses the declarative orchestrator pattern with:
    - ModelONEXContainer for dependency injection
    - ProjectionReaderRegistration for reading registration state
    - Decision event models for workflow verification

Related Tickets:
    - OMN-892: E2E Registration Tests
    - OMN-915: Mocked E2E Registration Tests (A0-A6)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_core.enums import EnumNodeKind

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)
from omnibase_infra.models.registration import (
    ModelNodeBecameActive,
    ModelNodeHeartbeatEvent,
    ModelNodeIntrospectionEvent,
    ModelNodeLivenessExpired,
    ModelNodeRegistrationAccepted,
    ModelNodeRegistrationAckReceived,
    ModelNodeRegistrationAckTimedOut,
    ModelNodeRegistrationInitiated,
    ModelNodeRegistrationRejected,
)

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from pydantic import BaseModel

    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
    from omnibase_infra.handlers import ConsulHandler
    from omnibase_infra.projectors import ProjectionReaderRegistration

logger = logging.getLogger(__name__)


# =============================================================================
# Consul Verification Helpers
# =============================================================================


async def verify_consul_registration(
    consul_handler: ConsulHandler,
    service_id: str,
    timeout_seconds: float = 5.0,
) -> dict[str, object] | None:
    """Verify a service is registered in Consul.

    Queries Consul for a service registration by service_id. Returns
    the registration dict if found, None otherwise. Retries until
    timeout to handle async registration propagation.

    Args:
        consul_handler: Initialized ConsulHandler instance.
        service_id: The service ID to verify.
        timeout_seconds: Maximum time to wait for registration.

    Returns:
        Service registration dict if found, None otherwise.

    Note:
        This helper queries Consul's KV store where service metadata
        is stored, not the agent service catalog directly.
    """
    # Build envelope for KV get operation
    envelope: dict[str, object] = {
        "operation": "consul.kv_get",
        "payload": {
            "key": f"onex/services/{service_id}",
        },
    }

    start_time = asyncio.get_running_loop().time()
    while asyncio.get_running_loop().time() - start_time < timeout_seconds:
        try:
            result = await consul_handler.execute(envelope)
            if result.result and result.result.payload:
                payload_data = result.result.payload.data
                # Check if we got a found response (not NotFound)
                if hasattr(payload_data, "value") and payload_data.value is not None:
                    return {"service_id": service_id, "value": payload_data.value}
        except (TimeoutError, ConnectionError, OSError) as e:
            # Network/connection errors - log and retry
            logger.debug(
                "Consul KV lookup failed (retrying): %s",
                type(e).__name__,
                extra={"service_id": service_id},
            )
        except Exception as e:
            # Unexpected errors - log with more detail but still retry
            logger.warning(
                "Unexpected error during Consul lookup (retrying): %s: %s",
                type(e).__name__,
                str(e),
                extra={"service_id": service_id},
            )

        await asyncio.sleep(0.2)

    return None


async def wait_for_consul_registration(
    consul_handler: ConsulHandler,
    service_id: str,
    timeout_seconds: float = 10.0,
    poll_interval: float = 0.5,
) -> dict[str, object]:
    """Wait for a service to appear in Consul.

    Polls Consul until the service is registered or timeout is reached.

    Args:
        consul_handler: Initialized ConsulHandler instance.
        service_id: The service ID to wait for.
        timeout_seconds: Maximum time to wait.
        poll_interval: Time between poll attempts.

    Returns:
        Service registration dict when found.

    Raises:
        TimeoutError: If service not found within timeout.
    """
    start_time = asyncio.get_running_loop().time()
    last_error: Exception | None = None

    while asyncio.get_running_loop().time() - start_time < timeout_seconds:
        try:
            result = await verify_consul_registration(
                consul_handler, service_id, timeout_seconds=poll_interval
            )
            if result is not None:
                return result
        except (TimeoutError, ConnectionError, OSError) as e:
            # Expected network/connection errors - log at debug and retry
            last_error = e
            logger.debug(
                "Consul wait poll failed: %s",
                type(e).__name__,
                extra={"service_id": service_id},
            )
        except Exception as e:
            # Unexpected errors - log with more detail
            last_error = e
            logger.warning(
                "Unexpected error during Consul poll: %s: %s",
                type(e).__name__,
                str(e),
                extra={"service_id": service_id},
            )

        await asyncio.sleep(poll_interval)

    error_msg = f"Service '{service_id}' not found in Consul within {timeout_seconds}s"
    if last_error:
        error_msg += f" (last error: {last_error})"
    raise TimeoutError(error_msg)


# =============================================================================
# PostgreSQL Verification Helpers (via ProjectionReader)
# =============================================================================


async def verify_postgres_registration(
    projection_reader: ProjectionReaderRegistration,
    node_id: UUID,
    domain: str = "registration",
) -> ModelRegistrationProjection | None:
    """Verify a node registration exists in PostgreSQL via projection reader.

    Uses the ProjectionReaderRegistration to query for a node's current
    registration state. This is the recommended way to verify persistence
    without raw SQL queries.

    Args:
        projection_reader: Initialized ProjectionReaderRegistration instance.
        node_id: UUID of the node to verify.
        domain: Domain namespace (default: "registration").

    Returns:
        ModelRegistrationProjection if found, None otherwise.
    """
    try:
        return await projection_reader.get_entity_state(node_id, domain)
    except (TimeoutError, ConnectionError, OSError) as e:
        # Expected network/database connection errors
        logger.debug(
            "Projection reader lookup failed: %s",
            type(e).__name__,
            extra={"node_id": str(node_id), "domain": domain},
        )
        return None
    except KeyError:
        # Entity not found - expected case, no need to log
        return None
    except Exception as e:
        # Unexpected errors - log with more detail
        logger.warning(
            "Unexpected error during projection lookup: %s: %s",
            type(e).__name__,
            str(e),
            extra={"node_id": str(node_id), "domain": domain},
        )
        return None


async def wait_for_postgres_registration(
    projection_reader: ProjectionReaderRegistration,
    node_id: UUID,
    expected_state: EnumRegistrationState | None = None,
    timeout_seconds: float = 10.0,
    poll_interval: float = 0.5,
) -> ModelRegistrationProjection:
    """Wait for registration to appear in PostgreSQL with expected state.

    Polls the projection reader until the registration appears and optionally
    matches the expected state.

    Args:
        projection_reader: Initialized ProjectionReaderRegistration instance.
        node_id: UUID of the node to wait for.
        expected_state: Optional state to wait for. If None, any state is accepted.
        timeout_seconds: Maximum time to wait.
        poll_interval: Time between poll attempts.

    Returns:
        ModelRegistrationProjection when found with matching state.

    Raises:
        TimeoutError: If registration not found or state not matched within timeout.
    """
    start_time = asyncio.get_running_loop().time()
    last_projection: ModelRegistrationProjection | None = None

    while asyncio.get_running_loop().time() - start_time < timeout_seconds:
        projection = await verify_postgres_registration(projection_reader, node_id)
        if projection is not None:
            last_projection = projection
            if expected_state is None or projection.current_state == expected_state:
                return projection

        await asyncio.sleep(poll_interval)

    if last_projection is not None and expected_state is not None:
        raise TimeoutError(
            f"Registration for node '{node_id}' found but state "
            f"'{last_projection.current_state}' != expected '{expected_state}' "
            f"within {timeout_seconds}s"
        )

    raise TimeoutError(
        f"Registration for node '{node_id}' not found in PostgreSQL "
        f"within {timeout_seconds}s"
    )


# =============================================================================
# Kafka Event Verification Helpers
# =============================================================================


async def wait_for_kafka_event(
    event_bus: KafkaEventBus,
    topic: str,
    correlation_id: UUID,
    timeout_seconds: float = 10.0,
) -> ModelEventEnvelope | None:
    """Wait for an event with matching correlation_id on topic.

    Subscribes temporarily to the topic and waits for an event with
    the specified correlation_id.

    Args:
        event_bus: Initialized KafkaEventBus instance.
        topic: Topic to listen on.
        correlation_id: Correlation ID to match.
        timeout_seconds: Maximum time to wait.

    Returns:
        ModelEventEnvelope if found, None on timeout.

    Note:
        This creates a temporary subscription that is cleaned up after
        the event is found or timeout is reached.
    """
    from omnibase_infra.event_bus.models import ModelEventMessage

    result: ModelEventEnvelope | None = None
    event_found = asyncio.Event()

    async def handler(message: ModelEventMessage) -> None:
        nonlocal result
        if message.headers and message.headers.correlation_id == correlation_id:
            # Try to parse as envelope
            if message.value:
                try:
                    import json

                    from omnibase_core.models.events.model_event_envelope import (
                        ModelEventEnvelope,
                    )

                    data = json.loads(message.value.decode("utf-8"))
                    result = ModelEventEnvelope.model_validate(data)
                    event_found.set()
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    # Malformed message data - skip silently (expected in test scenarios)
                    logger.debug("Failed to decode Kafka message: %s", type(e).__name__)
                except ValueError as e:
                    # Pydantic validation failed - message doesn't match expected schema
                    logger.debug("Failed to validate event envelope: %s", e)

    group_id = f"e2e-test-{correlation_id.hex[:8]}"
    unsubscribe = await event_bus.subscribe(topic, group_id, handler)

    try:
        await asyncio.wait_for(event_found.wait(), timeout=timeout_seconds)
        return result
    except TimeoutError:
        return None
    finally:
        await unsubscribe()


async def collect_registration_events(
    event_bus: KafkaEventBus,
    node_id: UUID,
    event_types: list[type[BaseModel]],
    timeout_seconds: float = 10.0,
) -> list[BaseModel]:
    """Collect specific registration event types for a node.

    Subscribes to registration event topics and collects events matching
    the specified types for the given node.

    Args:
        event_bus: Initialized KafkaEventBus instance.
        node_id: Node UUID to filter events for.
        event_types: List of event model types to collect.
        timeout_seconds: Maximum time to collect events.

    Returns:
        List of matched events (as Pydantic models).

    Note:
        This is a best-effort collection - events may be missed if they
        were published before subscription started.
    """
    import json

    from omnibase_infra.event_bus.models import ModelEventMessage

    collected: list[BaseModel] = []
    type_map = {t.__name__: t for t in event_types}

    async def handler(message: ModelEventMessage) -> None:
        if message.value:
            try:
                data = json.loads(message.value.decode("utf-8"))
                # Check if payload matches any expected type
                payload = data.get("payload", data)
                event_node_id = payload.get("node_id") or payload.get("entity_id")
                # Safely parse UUID - malformed UUIDs are skipped
                try:
                    parsed_node_id = UUID(event_node_id) if event_node_id else None
                except (ValueError, TypeError, AttributeError):
                    # Invalid UUID format or non-string type - skip this event
                    return
                if parsed_node_id == node_id:
                    # Extract event type for exact matching
                    event_type_name = data.get("event_type")
                    if event_type_name is None:
                        # Skip events without explicit event_type field
                        return

                    # Strict type check: event_type MUST be a string
                    # This prevents false positives from non-string types
                    # being coerced via str() (e.g., int, dict, list)
                    if not isinstance(event_type_name, str):
                        # Skip events with non-string event_type
                        return

                    # Validate non-empty after stripping whitespace
                    event_type_str = event_type_name.strip()
                    if not event_type_str:
                        # Skip events with empty event_type
                        return

                    # Extract the final component of namespaced event types
                    # e.g., "dev.registration.ModelNodeRegistrationInitiated"
                    #       -> "ModelNodeRegistrationInitiated"
                    # This uses rsplit to handle both namespaced and simple types
                    event_type_class_name = event_type_str.rsplit(".", 1)[-1]

                    # Validate extracted class name is non-empty and follows
                    # ONEX Model naming convention (prevents false positives)
                    if (
                        not event_type_class_name
                        or not event_type_class_name.startswith("Model")
                    ):
                        # Skip events that don't follow expected naming pattern
                        return

                    for type_name, model_class in type_map.items():
                        # Exact match only: event type class name must exactly
                        # equal the expected model class name (case-sensitive)
                        if event_type_class_name == type_name:
                            try:
                                event = model_class.model_validate(payload)
                                collected.append(event)
                                # Break after first match to prevent duplicate collection
                                break
                            except ValueError:
                                # Pydantic validation failed - skip this event
                                pass
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Malformed JSON or encoding - skip silently
                pass
            except (KeyError, TypeError, AttributeError):
                # Missing expected fields or unexpected structure
                # (e.g., payload is not a dict, missing required keys)
                pass

    # Subscribe to registration event topics
    topics = [
        "dev.registration.events.v1",
        "registration.decisions.v1",
    ]
    unsubscribers: list[Callable[[], Awaitable[None]]] = []
    group_id = f"e2e-collector-{node_id.hex[:8]}"

    for topic in topics:
        try:
            unsub = await event_bus.subscribe(topic, group_id, handler)
            unsubscribers.append(unsub)
        except (TimeoutError, ConnectionError, OSError) as e:
            # Network errors during subscription - skip this topic
            logger.debug("Failed to subscribe to topic %s: %s", topic, type(e).__name__)
        except Exception as e:
            # Unexpected errors - log but continue with other topics
            logger.warning(
                "Unexpected error subscribing to topic %s: %s: %s",
                topic,
                type(e).__name__,
                str(e),
            )

    await asyncio.sleep(timeout_seconds)

    # Cleanup subscriptions
    for unsub in unsubscribers:
        try:
            await unsub()
        except (TimeoutError, ConnectionError, OSError):
            # Network errors during cleanup - best effort
            pass
        except Exception as e:
            # Unexpected errors during cleanup - log but continue
            logger.debug("Error during subscription cleanup: %s", type(e).__name__)

    return collected


# =============================================================================
# Dual Registration Verification
# =============================================================================


async def verify_dual_registration(
    consul_handler: ConsulHandler,
    projection_reader: ProjectionReaderRegistration,
    node_id: UUID,
    service_id: str,
    timeout_seconds: float = 10.0,
) -> tuple[dict[str, object], ModelRegistrationProjection]:
    """Verify node is registered in BOTH Consul and PostgreSQL.

    Waits for registration to appear in both services, ensuring the
    dual registration pattern completed successfully.

    Args:
        consul_handler: Initialized ConsulHandler instance.
        projection_reader: Initialized ProjectionReaderRegistration instance.
        node_id: UUID of the node.
        service_id: Consul service ID.
        timeout_seconds: Maximum time to wait for both registrations.

    Returns:
        Tuple of (Consul registration dict, PostgreSQL projection).

    Raises:
        TimeoutError: If either registration not found within timeout.
    """
    start_time = asyncio.get_running_loop().time()
    consul_result: dict[str, object] | None = None
    postgres_result: ModelRegistrationProjection | None = None

    while asyncio.get_running_loop().time() - start_time < timeout_seconds:
        # Check Consul
        if consul_result is None:
            consul_result = await verify_consul_registration(
                consul_handler, service_id, timeout_seconds=1.0
            )

        # Check PostgreSQL
        if postgres_result is None:
            postgres_result = await verify_postgres_registration(
                projection_reader, node_id
            )

        # Both found
        if consul_result is not None and postgres_result is not None:
            return consul_result, postgres_result

        await asyncio.sleep(0.5)

    missing = []
    if consul_result is None:
        missing.append("Consul")
    if postgres_result is None:
        missing.append("PostgreSQL")

    raise TimeoutError(
        f"Dual registration incomplete - missing in: {', '.join(missing)} "
        f"(node_id={node_id}, service_id={service_id}, timeout={timeout_seconds}s)"
    )


# =============================================================================
# State Transition Verification
# =============================================================================


async def verify_state_transition(
    projection_reader: ProjectionReaderRegistration,
    node_id: UUID,
    from_state: EnumRegistrationState,
    to_state: EnumRegistrationState,
    timeout_seconds: float = 10.0,
) -> ModelRegistrationProjection:
    """Wait for a state transition to occur.

    Verifies that a node transitions from one state to another within
    the timeout period.

    Args:
        projection_reader: Initialized ProjectionReaderRegistration instance.
        node_id: UUID of the node.
        from_state: Expected starting state.
        to_state: Expected ending state.
        timeout_seconds: Maximum time to wait for transition.

    Returns:
        ModelRegistrationProjection in the target state.

    Raises:
        TimeoutError: If transition doesn't occur within timeout.
        AssertionError: If initial state doesn't match from_state.
    """
    # First, verify current state matches from_state (or wait for it)
    start_time = asyncio.get_running_loop().time()

    # Wait for from_state first (might not be there yet)
    initial_projection: ModelRegistrationProjection | None = None
    while asyncio.get_running_loop().time() - start_time < timeout_seconds / 2:
        projection = await verify_postgres_registration(projection_reader, node_id)
        if projection is not None:
            initial_projection = projection
            if projection.current_state == from_state:
                break
            if projection.current_state == to_state:
                # Already transitioned - that's fine
                return projection

        await asyncio.sleep(0.2)

    if initial_projection is not None and initial_projection.current_state == to_state:
        return initial_projection

    if initial_projection is None:
        raise TimeoutError(
            f"Node '{node_id}' not found while waiting for state transition "
            f"from '{from_state}' to '{to_state}'"
        )

    # Now wait for to_state
    remaining_time = timeout_seconds - (asyncio.get_running_loop().time() - start_time)
    if remaining_time <= 0:
        raise TimeoutError(
            f"Timeout waiting for state transition from '{from_state}' to '{to_state}' "
            f"for node '{node_id}' (current state: {initial_projection.current_state})"
        )

    return await wait_for_postgres_registration(
        projection_reader,
        node_id,
        expected_state=to_state,
        timeout_seconds=remaining_time,
        poll_interval=0.2,
    )


def assert_registration_state(
    projection: ModelRegistrationProjection,
    expected_state: EnumRegistrationState,
) -> None:
    """Assert registration is in expected state.

    Args:
        projection: The registration projection to check.
        expected_state: The expected registration state.

    Raises:
        AssertionError: If state doesn't match.
    """
    assert projection.current_state == expected_state, (
        f"Expected registration state '{expected_state}', "
        f"got '{projection.current_state}' for entity_id={projection.entity_id}"
    )


# =============================================================================
# Heartbeat Verification
# =============================================================================


def assert_heartbeat_updated(
    projection: ModelRegistrationProjection,
    min_heartbeat_time: datetime,
) -> None:
    """Assert last_heartbeat_at is after min_heartbeat_time.

    Args:
        projection: The registration projection to check.
        min_heartbeat_time: Minimum expected heartbeat timestamp.

    Raises:
        AssertionError: If heartbeat is None or before min time.
    """
    assert projection.last_heartbeat_at is not None, (
        f"Expected last_heartbeat_at to be set, got None "
        f"for entity_id={projection.entity_id}"
    )

    assert projection.last_heartbeat_at >= min_heartbeat_time, (
        f"Expected last_heartbeat_at ({projection.last_heartbeat_at}) "
        f">= min_heartbeat_time ({min_heartbeat_time}) "
        f"for entity_id={projection.entity_id}"
    )


async def wait_for_heartbeat_update(
    projection_reader: ProjectionReaderRegistration,
    node_id: UUID,
    min_heartbeat_time: datetime,
    timeout_seconds: float = 35.0,
) -> ModelRegistrationProjection:
    """Wait for heartbeat to be updated after min time.

    Polls until last_heartbeat_at is after the specified minimum time.
    Default timeout is 35s to allow for the standard 30s heartbeat interval.

    Args:
        projection_reader: Initialized ProjectionReaderRegistration instance.
        node_id: UUID of the node.
        min_heartbeat_time: Minimum heartbeat timestamp to wait for.
        timeout_seconds: Maximum time to wait (default 35s > 30s heartbeat interval).

    Returns:
        ModelRegistrationProjection with updated heartbeat.

    Raises:
        TimeoutError: If heartbeat not updated within timeout.
    """
    start_time = asyncio.get_running_loop().time()

    while asyncio.get_running_loop().time() - start_time < timeout_seconds:
        projection = await verify_postgres_registration(projection_reader, node_id)
        if (
            projection is not None
            and projection.last_heartbeat_at is not None
            and projection.last_heartbeat_at >= min_heartbeat_time
        ):
            return projection

        await asyncio.sleep(1.0)

    projection = await verify_postgres_registration(projection_reader, node_id)
    if projection is None:
        raise TimeoutError(
            f"Node '{node_id}' not found while waiting for heartbeat update"
        )

    raise TimeoutError(
        f"Heartbeat not updated after {min_heartbeat_time} within {timeout_seconds}s "
        f"for node '{node_id}'. Last heartbeat: {projection.last_heartbeat_at}"
    )


# =============================================================================
# Event Model Assertions
# =============================================================================


def assert_introspection_event_complete(event: ModelNodeIntrospectionEvent) -> None:
    """Assert introspection event has all required fields.

    Validates that the introspection event contains all essential fields
    for node registration.

    Args:
        event: The introspection event to validate.

    Raises:
        AssertionError: If any required field is missing or invalid.
    """
    assert event.node_id is not None, "node_id is required"
    assert event.node_type is not None, "node_type is required"
    # Use EnumNodeKind values for type-safe validation (excluding RUNTIME_HOST)
    valid_node_types = {
        EnumNodeKind.EFFECT.value,
        EnumNodeKind.COMPUTE.value,
        EnumNodeKind.REDUCER.value,
        EnumNodeKind.ORCHESTRATOR.value,
    }
    assert event.node_type in valid_node_types, f"Invalid node_type: {event.node_type}"
    assert event.node_version is not None, "node_version is required"
    assert event.correlation_id is not None, "correlation_id is required"
    assert event.timestamp is not None, "timestamp is required"
    assert event.timestamp.tzinfo is not None, "timestamp must be timezone-aware"
    assert event.capabilities is not None, "capabilities is required"


def assert_registration_initiated(event: ModelNodeRegistrationInitiated) -> None:
    """Assert registration initiated event is valid.

    Validates the event structure for registration initiation.

    Args:
        event: The registration initiated event to validate.

    Raises:
        AssertionError: If any required field is missing or invalid.
    """
    assert event.entity_id is not None, "entity_id is required"
    assert event.node_id is not None, "node_id is required"
    assert event.entity_id == event.node_id, (
        f"entity_id ({event.entity_id}) should equal node_id ({event.node_id}) "
        "in registration domain"
    )
    assert event.correlation_id is not None, "correlation_id is required"
    assert event.causation_id is not None, "causation_id is required"
    assert event.emitted_at is not None, "emitted_at is required"
    assert event.registration_attempt_id is not None, (
        "registration_attempt_id is required"
    )


def assert_node_became_active(event: ModelNodeBecameActive) -> None:
    """Assert node became active event is valid.

    Validates the event structure for node activation.

    Args:
        event: The node became active event to validate.

    Raises:
        AssertionError: If any required field is missing or invalid.
    """
    assert event.entity_id is not None, "entity_id is required"
    assert event.node_id is not None, "node_id is required"
    assert event.entity_id == event.node_id, (
        f"entity_id ({event.entity_id}) should equal node_id ({event.node_id}) "
        "in registration domain"
    )
    assert event.correlation_id is not None, "correlation_id is required"
    assert event.causation_id is not None, "causation_id is required"
    assert event.emitted_at is not None, "emitted_at is required"
    assert event.capabilities is not None, "capabilities is required"


def assert_registration_accepted(event: ModelNodeRegistrationAccepted) -> None:
    """Assert registration accepted event is valid.

    Validates the event structure for registration acceptance.

    Args:
        event: The registration accepted event to validate.

    Raises:
        AssertionError: If any required field is missing or invalid.
    """
    assert event.entity_id is not None, "entity_id is required"
    assert event.node_id is not None, "node_id is required"
    assert event.correlation_id is not None, "correlation_id is required"
    assert event.causation_id is not None, "causation_id is required"
    assert event.emitted_at is not None, "emitted_at is required"


def assert_registration_rejected(event: ModelNodeRegistrationRejected) -> None:
    """Assert registration rejected event is valid.

    Validates the event structure for registration rejection.

    Args:
        event: The registration rejected event to validate.

    Raises:
        AssertionError: If any required field is missing or invalid.
    """
    assert event.entity_id is not None, "entity_id is required"
    assert event.node_id is not None, "node_id is required"
    assert event.correlation_id is not None, "correlation_id is required"
    assert event.causation_id is not None, "causation_id is required"
    assert event.emitted_at is not None, "emitted_at is required"
    assert event.rejection_reason is not None and event.rejection_reason.strip(), (
        "rejection_reason is required"
    )


def assert_ack_received(event: ModelNodeRegistrationAckReceived) -> None:
    """Assert ack received event is valid.

    Validates the event structure for acknowledgment receipt.

    Args:
        event: The ack received event to validate.

    Raises:
        AssertionError: If any required field is missing or invalid.
    """
    assert event.entity_id is not None, "entity_id is required"
    assert event.node_id is not None, "node_id is required"
    assert event.correlation_id is not None, "correlation_id is required"
    assert event.causation_id is not None, "causation_id is required"
    assert event.emitted_at is not None, "emitted_at is required"


def assert_ack_timed_out(event: ModelNodeRegistrationAckTimedOut) -> None:
    """Assert ack timed out event is valid.

    Validates the event structure for acknowledgment timeout.

    Args:
        event: The ack timed out event to validate.

    Raises:
        AssertionError: If any required field is missing or invalid.
    """
    assert event.entity_id is not None, "entity_id is required"
    assert event.node_id is not None, "node_id is required"
    assert event.correlation_id is not None, "correlation_id is required"
    assert event.causation_id is not None, "causation_id is required"
    assert event.emitted_at is not None, "emitted_at is required"


def assert_liveness_expired(event: ModelNodeLivenessExpired) -> None:
    """Assert liveness expired event is valid.

    Validates the event structure for liveness expiration.

    Args:
        event: The liveness expired event to validate.

    Raises:
        AssertionError: If any required field is missing or invalid.
    """
    assert event.entity_id is not None, "entity_id is required"
    assert event.node_id is not None, "node_id is required"
    assert event.correlation_id is not None, "correlation_id is required"
    assert event.causation_id is not None, "causation_id is required"
    assert event.emitted_at is not None, "emitted_at is required"
    # last_heartbeat_at can be None if node never sent a heartbeat


def assert_heartbeat_event_valid(event: ModelNodeHeartbeatEvent) -> None:
    """Assert heartbeat event has all required fields.

    Validates that the heartbeat event contains all essential fields.

    Args:
        event: The heartbeat event to validate.

    Raises:
        AssertionError: If any required field is missing or invalid.
    """
    assert event.node_id is not None, "node_id is required"
    assert event.node_type is not None, "node_type is required"
    assert event.uptime_seconds >= 0, "uptime_seconds must be non-negative"
    assert event.timestamp is not None, "timestamp is required"
    assert event.timestamp.tzinfo is not None, "timestamp must be timezone-aware"


__all__: list[str] = [
    # Consul verification
    "verify_consul_registration",
    "wait_for_consul_registration",
    # PostgreSQL verification
    "verify_postgres_registration",
    "wait_for_postgres_registration",
    # Kafka verification
    "wait_for_kafka_event",
    "collect_registration_events",
    # Dual registration
    "verify_dual_registration",
    # State transitions
    "verify_state_transition",
    "assert_registration_state",
    # Heartbeat verification
    "assert_heartbeat_updated",
    "wait_for_heartbeat_update",
    # Event model assertions
    "assert_introspection_event_complete",
    "assert_registration_initiated",
    "assert_node_became_active",
    "assert_registration_accepted",
    "assert_registration_rejected",
    "assert_ack_received",
    "assert_ack_timed_out",
    "assert_liveness_expired",
    "assert_heartbeat_event_valid",
]
