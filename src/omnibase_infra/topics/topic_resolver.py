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
    Pass-through by default. Topic suffixes are returned unchanged because ONEX
    topics are realm-agnostic. The environment/realm is enforced via envelope
    identity and consumer group naming, NOT via topic name prefixing.

Multi-Bus Resolution (Phase 5 - OMN-2894):
    When constructed with ``bus_descriptors`` and called with a ``trust_domain``
    parameter, the resolver finds the matching bus descriptor and prepends its
    ``namespace_prefix`` to the validated topic suffix. This enables
    trust-domain-scoped topic routing across multiple message buses.

    Without these optional parameters, behavior is identical to the original
    pass-through implementation (full backward compatibility).

Topic Suffix Format:
    onex.<kind>.<producer>.<event-name>.v<version>

    Examples:
        onex.evt.platform.node-registration.v1
        onex.cmd.platform.request-introspection.v1

See Also:
    omnibase_core.validation.validate_topic_suffix - Suffix format validation
    omnibase_infra.topics.platform_topic_suffixes - Platform-reserved suffixes
    omnibase_infra.topics.model_bus_descriptor - Bus descriptor model
"""

from __future__ import annotations

from uuid import UUID

from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.topics.model_bus_descriptor import ModelBusDescriptor


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


class BusDescriptorNotFoundError(TopicResolutionError):
    """Raised when no bus descriptor matches the requested trust domain.

    This error indicates that the ``TopicResolver`` was asked to resolve a
    topic for a specific trust domain, but no ``ModelBusDescriptor`` in the
    configured set matches that domain.

    Attributes:
        trust_domain: The trust domain that was requested but not found.
    """

    def __init__(
        self,
        trust_domain: str,
        *,
        correlation_id: UUID | None = None,
        infra_context: ModelInfraErrorContext | None = None,
    ) -> None:
        self.trust_domain = trust_domain
        message = f"No bus descriptor found for trust domain '{trust_domain}'"
        if correlation_id is not None:
            message += f" (correlation_id={correlation_id})"
        super().__init__(
            message,
            correlation_id=correlation_id,
            infra_context=infra_context,
        )


class TopicResolver:
    """Canonical resolver that maps ONEX topic suffixes to concrete Kafka topics.

    This is the single source of truth for topic name resolution in ONEX. All
    components that need to resolve a topic suffix to a concrete Kafka topic
    MUST use this class rather than implementing their own resolution logic.

    The resolver validates that the provided suffix conforms to the ONEX topic
    naming convention before returning it. Invalid suffixes are rejected with
    a ``TopicResolutionError``.

    **Default behavior** (no ``bus_descriptors``): pass-through. The validated
    suffix is returned unchanged. The environment is enforced via consumer
    group naming, not topic names.

    **Multi-bus behavior** (with ``bus_descriptors`` and ``trust_domain`` on
    ``resolve()``): the resolver looks up the bus descriptor for the given
    trust domain and prepends its ``namespace_prefix`` to the validated suffix.

    Args:
        bus_descriptors: Optional sequence of bus descriptors for multi-bus
            routing. When ``None`` (the default), the resolver operates in
            pass-through mode regardless of the ``trust_domain`` argument
            to ``resolve()``.

    Example:
        >>> # Pass-through (backward compatible)
        >>> resolver = TopicResolver()
        >>> resolver.resolve("onex.evt.platform.node-registration.v1")
        'onex.evt.platform.node-registration.v1'

        >>> # Multi-bus with namespace prefix
        >>> from omnibase_infra.topics.model_bus_descriptor import ModelBusDescriptor
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>> desc = ModelBusDescriptor(
        ...     bus_id="bus.org",
        ...     trust_domain="org.omninode",
        ...     transport_type=EnumInfraTransportType.KAFKA,
        ...     namespace_prefix="org.omninode.",
        ...     bootstrap_servers=("kafka.org:9092",),
        ... )
        >>> resolver = TopicResolver(bus_descriptors=[desc])
        >>> resolver.resolve(
        ...     "onex.evt.platform.node-registration.v1",
        ...     trust_domain="org.omninode",
        ... )
        'org.omninode.onex.evt.platform.node-registration.v1'

    .. versionchanged:: 0.10.0
        Added optional ``bus_descriptors`` constructor parameter and
        ``trust_domain`` parameter on ``resolve()`` for Phase 5 multi-bus
        topic resolution (OMN-2894).
    """

    def __init__(
        self,
        bus_descriptors: list[ModelBusDescriptor] | None = None,
    ) -> None:
        # Index descriptors by trust_domain for O(1) lookup.
        self._descriptors_by_domain: dict[str, ModelBusDescriptor] = {}
        if bus_descriptors is not None:
            for desc in bus_descriptors:
                self._descriptors_by_domain[desc.trust_domain] = desc

    @property
    def bus_descriptors(self) -> list[ModelBusDescriptor]:
        """Return a list of all configured bus descriptors.

        Returns an empty list when no descriptors are configured (pass-through
        mode).
        """
        return list(self._descriptors_by_domain.values())

    def resolve(
        self,
        topic_suffix: str,
        *,
        correlation_id: UUID | None = None,
        trust_domain: str | None = None,
    ) -> str:
        """Resolve a topic suffix to a concrete Kafka topic name.

        Validates the suffix against the ONEX topic naming convention and
        returns the resolved topic name. When ``trust_domain`` is provided
        and bus descriptors are configured, the matching descriptor's
        ``namespace_prefix`` is prepended to the suffix.

        Args:
            topic_suffix: ONEX format topic suffix
                (e.g., ``'onex.evt.platform.node-registration.v1'``)
            correlation_id: Optional correlation ID for error traceability.
                When provided, included in the ``TopicResolutionError`` message
                so callers can correlate failures to specific request flows.
            trust_domain: Optional trust domain for multi-bus routing. When
                provided with configured bus descriptors, the matching
                descriptor's namespace prefix is prepended to the topic.
                When ``None`` or when no bus descriptors are configured,
                behavior is identical to pass-through.

        Returns:
            Concrete Kafka topic name. Identical to the input suffix in
            pass-through mode, or prefixed with the bus namespace when
            trust-domain routing is active.

        Raises:
            TopicResolutionError: If the suffix does not match the required
                ONEX topic format ``onex.<kind>.<producer>.<event-name>.v<n>``.
            BusDescriptorNotFoundError: If ``trust_domain`` is provided and
                bus descriptors are configured, but no descriptor matches the
                requested trust domain.
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

        # Multi-bus resolution: if trust_domain is provided and descriptors
        # are configured, look up the matching bus and prepend its prefix.
        if trust_domain is not None and self._descriptors_by_domain:
            descriptor = self._descriptors_by_domain.get(trust_domain)
            if descriptor is None:
                infra_context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="resolve_topic",
                )
                raise BusDescriptorNotFoundError(
                    trust_domain,
                    correlation_id=correlation_id,
                    infra_context=infra_context,
                )
            if descriptor.namespace_prefix:
                return f"{descriptor.namespace_prefix}{topic_suffix}"

        return topic_suffix
