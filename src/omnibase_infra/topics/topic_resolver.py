# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Canonical topic resolver for ONEX infrastructure.

Formal Invariant:
    Contracts declare realm-agnostic topic suffixes. The TopicResolver is the
    single canonical function that maps topic suffix -> concrete Kafka topic.

    All scattered ``resolve_topic()`` methods across the codebase (event bus
    wiring, adapters, dispatchers, etc.) MUST delegate to this class. Direct
    pass-through logic in individual components is prohibited.

Current Behavior:
    Pass-through. Topic suffixes are returned unchanged because ONEX topics are
    realm-agnostic. The environment/realm is enforced via envelope identity and
    consumer group naming, NOT via topic name prefixing.

Future Phases:
    This class is the single extension point for realm-based routing, topic
    aliasing, or tenant-scoped topic mapping. When those features are needed,
    they are added HERE and all callers automatically benefit.

Topic Suffix Format:
    onex.<kind>.<producer>.<event-name>.v<version>

    Examples:
        onex.evt.platform.node-registration.v1
        onex.cmd.platform.request-introspection.v1

See Also:
    omnibase_core.validation.validate_topic_suffix - Suffix format validation
    omnibase_infra.topics.platform_topic_suffixes - Platform-reserved suffixes
"""

from __future__ import annotations

from uuid import UUID

from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class TopicResolutionError(ProtocolConfigurationError):
    """Raised when a topic suffix cannot be resolved to a concrete topic.

    This error indicates that the provided topic suffix does not conform to the
    ONEX topic naming convention and therefore cannot be mapped to a Kafka topic.

    Extends ``ProtocolConfigurationError`` so that all TopicResolver failures
    are automatically instances of the canonical infrastructure configuration
    error type. This ensures consistent error taxonomy across the codebase
    without requiring callers to manually wrap topic errors.

    A ``ModelInfraErrorContext`` is always attached (auto-generated when not
    explicitly provided), guaranteeing that every ``TopicResolutionError``
    carries a ``correlation_id`` for distributed tracing.

    Attributes:
        infra_context: ``ModelInfraErrorContext`` carrying the correlation_id,
            transport type, and operation. Always present -- callers can rely
            on structured error context without parsing the message.
    """

    def __init__(
        self,
        message: str,
        *,
        correlation_id: UUID | None = None,
        infra_context: ModelInfraErrorContext | None = None,
    ) -> None:
        """Initialize TopicResolutionError with correlation tracking.

        If ``infra_context`` is not provided, one is auto-generated with
        transport_type=KAFKA and operation="resolve_topic". If neither
        ``infra_context`` nor ``correlation_id`` is provided, a fresh
        correlation_id is auto-generated so every error is traceable.

        Args:
            message: Human-readable error message.
            correlation_id: Optional correlation ID for distributed tracing.
                Used to build an ``infra_context`` when one is not explicitly
                provided.
            infra_context: Optional infrastructure error context with transport
                type, operation, and correlation_id. When provided, takes
                precedence over ``correlation_id``.
        """
        if infra_context is None:
            infra_context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="resolve_topic",
            )
        self.infra_context = infra_context
        super().__init__(message, context=infra_context)


class TopicResolver:
    """Canonical resolver that maps ONEX topic suffixes to concrete Kafka topics.

    This is the single source of truth for topic name resolution in ONEX. All
    components that need to resolve a topic suffix to a concrete Kafka topic
    MUST use this class rather than implementing their own resolution logic.

    The resolver validates that the provided suffix conforms to the ONEX topic
    naming convention before returning it. Invalid suffixes are rejected with
    a ``TopicResolutionError``.

    Current behavior is pass-through (realm-agnostic topics, no environment
    prefix). The environment is enforced via consumer group naming, not topic
    names.

    Example:
        >>> resolver = TopicResolver()
        >>> resolver.resolve("onex.evt.platform.node-registration.v1")
        'onex.evt.platform.node-registration.v1'

        >>> resolver.resolve("bad-topic")
        Traceback (most recent call last):
            ...
        TopicResolutionError: Invalid topic suffix 'bad-topic': ...
    """

    def resolve(
        self,
        topic_suffix: str,
        *,
        correlation_id: UUID | None = None,
    ) -> str:
        """Resolve a topic suffix to a concrete Kafka topic name.

        Validates the suffix against the ONEX topic naming convention and
        returns the resolved topic name. Currently this is a pass-through
        (the suffix IS the topic name) because ONEX topics are realm-agnostic.

        Args:
            topic_suffix: ONEX format topic suffix
                (e.g., ``'onex.evt.platform.node-registration.v1'``)
            correlation_id: Optional correlation ID for error traceability.
                When provided, included in the ``TopicResolutionError`` message
                so callers can correlate failures to specific request flows.

        Returns:
            Concrete Kafka topic name. Currently identical to the input suffix.

        Raises:
            TopicResolutionError: If the suffix does not match the required
                ONEX topic format ``onex.<kind>.<producer>.<event-name>.v<n>``.
        """
        result = validate_topic_suffix(topic_suffix)
        if not result.is_valid:
            # Always build structured infra context with a correlation_id.
            # When the caller provides a correlation_id it is propagated;
            # otherwise ModelInfraErrorContext.with_correlation() auto-generates
            # one so every error is traceable via distributed tracing.
            infra_context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="resolve_topic",
            )
            # Include correlation_id in the human-readable message only when
            # the caller explicitly provided one; auto-generated IDs are
            # available via the structured infra_context attribute.
            if correlation_id is not None:
                raise TopicResolutionError(
                    f"Invalid topic suffix '{topic_suffix}' "
                    f"(correlation_id={correlation_id}): {result.error}",
                    correlation_id=correlation_id,
                    infra_context=infra_context,
                )
            raise TopicResolutionError(
                f"Invalid topic suffix '{topic_suffix}': {result.error}",
                infra_context=infra_context,
            )
        return topic_suffix
