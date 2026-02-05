# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Engine Service for Runtime Gateway.

This module provides a policy engine for evaluating message filtering rules
in the ONEX runtime gateway. The engine supports:

- Topic allowlist checking with wildcard patterns
- Realm boundary enforcement to prevent cross-realm messages
- Extensible policy evaluation for inbound/outbound messages

The policy engine is designed to be lightweight and fast, suitable for
high-throughput message processing pipelines.

Design Decisions:
    - Empty allowlist means "allow all" for operational simplicity
    - Wildcard patterns use simple prefix/suffix matching (not regex)
    - Rejections are logged by default for security audit trails
    - All policy decisions include a reason for debugging

Security Considerations:
    - Topic allowlists should be loaded from trusted configuration
    - Realm enforcement prevents tenant isolation violations
    - All rejections are logged with full context for audit

Example:
    >>> from omnibase_infra.gateway.services import (
    ...     ServicePolicyEngine,
    ...     EnumPolicyDecision,
    ... )
    >>> engine = ServicePolicyEngine(
    ...     allowed_topics=["events.*", "commands.user.*"],
    ...     expected_realm="tenant-123",
    ... )
    >>> decision = engine.evaluate_inbound("events.order.created", realm="tenant-123")
    >>> assert decision.decision == EnumPolicyDecision.ALLOW

Related:
    - OMN-1899: Runtime gateway policy engine implementation
    - OMN-1897: Infrastructure Docker Compose integration

"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class EnumPolicyDecision(str, Enum):
    """Policy evaluation decision.

    Represents the outcome of a policy evaluation. Used by the
    gateway to determine whether to allow or reject a message.

    Attributes:
        ALLOW: Message passes policy checks and should be processed.
        DENY: Message fails policy checks and should be rejected.

    """

    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    """Result of policy evaluation.

    Immutable dataclass containing the policy decision and an optional
    reason explaining the decision. The reason is particularly useful
    for debugging rejected messages.

    Attributes:
        decision: The policy decision (ALLOW or DENY).
        reason: Optional explanation for the decision. Always populated
            for DENY decisions, may be None for ALLOW decisions.

    Example:
        >>> decision = PolicyDecision(
        ...     decision=EnumPolicyDecision.DENY,
        ...     reason="Topic 'events.internal.secret' not in allowlist",
        ... )
        >>> if decision.decision == EnumPolicyDecision.DENY:
        ...     logger.warning(f"Message rejected: {decision.reason}")

    """

    decision: EnumPolicyDecision
    reason: str | None = None

    def __bool__(self) -> bool:
        """Allow using decision in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            the decision is ALLOW. Differs from typical dataclass behavior
            where any non-empty instance would be truthy.

        Returns:
            True if decision is ALLOW, False if DENY.

        Example:
            >>> decision = PolicyDecision(decision=EnumPolicyDecision.ALLOW)
            >>> if decision:
            ...     process_message()

        """
        return self.decision == EnumPolicyDecision.ALLOW


class ServicePolicyEngine:
    """Policy engine for message filtering and topic allowlisting.

    Evaluates whether messages should be allowed based on configurable
    policies including topic allowlists and realm boundary enforcement.

    The policy engine is designed for high-throughput scenarios with
    minimal overhead. Pattern matching uses simple glob-style wildcards
    rather than full regex for performance.

    Thread Safety:
        This class is thread-safe. All state is immutable after construction
        and evaluation methods are pure functions with no side effects
        (except optional logging).

    Attributes:
        allowed_topics: Tuple of allowed topic patterns. None or empty
            means all topics are allowed (open policy).
        expected_realm: Expected realm identifier for inbound messages.
            None means realm checking is disabled.
        log_rejections: Whether to log rejected messages (default: True).

    Example (Topic Allowlist):
        >>> engine = ServicePolicyEngine(
        ...     allowed_topics=["events.*", "commands.user.created"],
        ... )
        >>> engine.evaluate_inbound("events.order.created")
        PolicyDecision(decision=<EnumPolicyDecision.ALLOW>, reason=None)
        >>> engine.evaluate_inbound("internal.secret")
        PolicyDecision(decision=<EnumPolicyDecision.DENY>, reason="Topic 'internal.secret' not in allowlist")

    Example (Realm Enforcement):
        >>> engine = ServicePolicyEngine(
        ...     expected_realm="tenant-123",
        ... )
        >>> engine.evaluate_inbound("events.order", realm="tenant-123")
        PolicyDecision(decision=<EnumPolicyDecision.ALLOW>, reason=None)
        >>> engine.evaluate_inbound("events.order", realm="tenant-456")
        PolicyDecision(decision=<EnumPolicyDecision.DENY>, reason="Realm mismatch: expected 'tenant-123', got 'tenant-456'")

    Example (Combined Policies):
        >>> engine = ServicePolicyEngine(
        ...     allowed_topics=["events.*"],
        ...     expected_realm="tenant-123",
        ... )
        >>> # Both topic AND realm must match
        >>> engine.evaluate_inbound("events.order", realm="tenant-123")
        PolicyDecision(decision=<EnumPolicyDecision.ALLOW>, reason=None)

    """

    def __init__(
        self,
        allowed_topics: list[str] | None = None,
        expected_realm: str | None = None,
        log_rejections: bool = True,
    ) -> None:
        """Initialize policy engine.

        Args:
            allowed_topics: Topic allowlist patterns. Supports glob-style
                wildcards (e.g., "events.*" matches "events.order.created").
                None or empty list means all topics are allowed.
            expected_realm: Expected realm for incoming messages. When set,
                inbound messages must have a matching realm or be rejected.
                None disables realm checking.
            log_rejections: Whether to log rejected messages for audit/debugging.
                Enabled by default for security audit trails.

        Note:
            The allowed_topics list is converted to an immutable tuple internally
            to ensure thread safety and prevent accidental modification.

        """
        # Convert to tuple for immutability and thread safety
        self._allowed_topics: tuple[str, ...] | None = (
            tuple(allowed_topics) if allowed_topics else None
        )
        self._expected_realm: str | None = expected_realm
        self._log_rejections: bool = log_rejections

        # Log configuration for debugging (without sensitive details)
        logger.debug(
            "ServicePolicyEngine initialized",
            extra={
                "allowlist_count": len(self._allowed_topics)
                if self._allowed_topics
                else 0,
                "realm_enforcement": self._expected_realm is not None,
                "log_rejections": self._log_rejections,
            },
        )

    @property
    def allowed_topics(self) -> tuple[str, ...] | None:
        """Return the configured topic allowlist.

        Returns:
            Immutable tuple of allowed topic patterns, or None if
            all topics are allowed.

        """
        return self._allowed_topics

    @property
    def expected_realm(self) -> str | None:
        """Return the expected realm for inbound messages.

        Returns:
            The expected realm identifier, or None if realm checking
            is disabled.

        """
        return self._expected_realm

    def evaluate_inbound(
        self,
        topic: str,
        realm: str | None = None,
        correlation_id: UUID | None = None,
    ) -> PolicyDecision:
        """Evaluate inbound message policy.

        Checks whether an inbound message should be allowed based on:
        1. Topic allowlist (if configured)
        2. Realm boundary enforcement (if configured)

        Both checks must pass for the message to be allowed. Checks are
        performed in order and short-circuit on first failure.

        Args:
            topic: The topic the message is arriving on.
            realm: The realm identifier from the message metadata.
                Required if expected_realm is configured.
            correlation_id: Optional correlation ID for request tracing.
                Included in rejection logs for debugging and audit trails.

        Returns:
            PolicyDecision with decision (ALLOW/DENY) and reason.
            The reason is always populated for DENY decisions.

        Example:
            >>> engine = ServicePolicyEngine(
            ...     allowed_topics=["events.*"],
            ...     expected_realm="tenant-123",
            ... )
            >>> decision = engine.evaluate_inbound(
            ...     topic="events.order.created",
            ...     realm="tenant-123",
            ... )
            >>> if not decision:
            ...     raise PermissionError(decision.reason)

        """
        # Auto-generate correlation_id if not provided for rejection traceability
        if correlation_id is None:
            correlation_id = uuid4()

        # Check 1: Topic allowlist
        if not self.is_topic_allowed(topic):
            reason = f"Topic '{topic}' not in allowlist"
            self._log_rejection("inbound", topic, reason, correlation_id)
            return PolicyDecision(
                decision=EnumPolicyDecision.DENY,
                reason=reason,
            )

        # Check 2: Realm boundary enforcement
        if self._expected_realm is not None:
            if realm != self._expected_realm:
                reason = (
                    f"Realm mismatch: expected '{self._expected_realm}', got '{realm}'"
                )
                self._log_rejection("inbound", topic, reason, correlation_id)
                return PolicyDecision(
                    decision=EnumPolicyDecision.DENY,
                    reason=reason,
                )

        # All checks passed
        return PolicyDecision(decision=EnumPolicyDecision.ALLOW)

    def evaluate_outbound(
        self,
        topic: str,
        correlation_id: UUID | None = None,
    ) -> PolicyDecision:
        """Evaluate outbound message policy.

        Currently allows all outbound messages. This method exists as an
        extension point for future egress filtering capabilities such as:
        - Preventing sensitive data from leaving the system
        - Enforcing topic naming conventions
        - Rate limiting outbound traffic

        Args:
            topic: The topic the message is being sent to.
            correlation_id: Optional correlation ID for request tracing.
                Included in rejection logs for debugging and audit trails.

        Returns:
            PolicyDecision with ALLOW decision. Future versions may
            implement egress filtering that could return DENY.

        Note:
            Outbound policy is intentionally permissive by default.
            Egress filtering should be implemented carefully to avoid
            breaking legitimate message flows.

        """
        # Auto-generate correlation_id if not provided for future rejection traceability
        if correlation_id is None:
            correlation_id = uuid4()

        # Currently always allow outbound messages
        # Future: Add egress filtering if needed
        _ = topic  # Suppress unused argument warning
        _ = correlation_id  # Reserved for future egress filtering
        return PolicyDecision(decision=EnumPolicyDecision.ALLOW)

    def is_topic_allowed(self, topic: str) -> bool:
        """Check if topic is in allowlist.

        Evaluates whether a topic matches any pattern in the allowlist.
        Returns True if:
        - allowed_topics is None or empty (allow all policy)
        - topic matches any pattern exactly
        - topic matches any wildcard pattern

        Wildcard Patterns:
            Uses Python's fnmatch for glob-style pattern matching:
            - "*" matches any sequence of characters
            - "?" matches any single character
            - "[seq]" matches any character in seq
            - "[!seq]" matches any character not in seq

        Common Patterns:
            - "events.*" matches "events.order.created", "events.user.login"
            - "*.created" matches "events.order.created", "commands.user.created"
            - "events.order.*" matches "events.order.created", "events.order.shipped"

        Args:
            topic: The topic name to check.

        Returns:
            True if topic is allowed, False otherwise.

        Example:
            >>> engine = ServicePolicyEngine(allowed_topics=["events.*"])
            >>> engine.is_topic_allowed("events.order.created")
            True
            >>> engine.is_topic_allowed("commands.user.create")
            False

        """
        # No allowlist = allow all
        if not self._allowed_topics:
            return True

        # Check each pattern
        for pattern in self._allowed_topics:
            if self._matches_pattern(topic, pattern):
                return True

        return False

    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern.

        Uses fnmatch for glob-style pattern matching, which provides
        a good balance between expressiveness and performance.

        Supports:
            - Exact match: "events.user.created" matches "events.user.created"
            - Wildcard: "events.*" matches "events.user.created"
            - Single char: "events.?" matches "events.a" but not "events.ab"
            - Character sets: "events.[ou]rder" matches "events.order"

        Args:
            topic: The topic name to check.
            pattern: The pattern to match against.

        Returns:
            True if topic matches pattern, False otherwise.

        Note:
            Pattern matching is case-sensitive. Topics and patterns
            should use consistent casing (typically lowercase).

        """
        # Use fnmatch for glob-style pattern matching
        # This handles *, ?, and [seq] patterns efficiently
        return fnmatch.fnmatch(topic, pattern)

    def _log_rejection(
        self,
        direction: str,
        topic: str,
        reason: str,
        correlation_id: UUID,
    ) -> None:
        """Log rejected message for audit/debugging.

        Logs a warning when a message is rejected by policy. This provides
        an audit trail for security monitoring and helps with debugging
        configuration issues.

        Args:
            direction: Message direction ("inbound" or "outbound").
            topic: The topic of the rejected message.
            reason: The reason for rejection.
            correlation_id: Correlation ID for request tracing. Required
                for all rejection logs to ensure traceability.

        Note:
            Logging can be disabled via log_rejections=False in constructor
            for performance-critical scenarios, but this is not recommended
            for production as it reduces security visibility.

        """
        if not self._log_rejections:
            return

        logger.warning(
            "Policy rejection [correlation_id=%s]: %s message on topic '%s' - %s",
            correlation_id,
            direction,
            topic,
            reason,
            extra={
                "direction": direction,
                "topic": topic,
                "reason": reason,
                "policy_engine": "ServicePolicyEngine",
                "correlation_id": str(correlation_id),
            },
        )


__all__: list[str] = [
    "EnumPolicyDecision",
    "PolicyDecision",
    "ServicePolicyEngine",
]
