# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Domain Constraint Model.

Defines domain ownership rules and cross-domain consumption constraints for
message type routing in the ONEX runtime.

Design Principles:
    - Domain ownership is derived from topic name (onex.<domain>.*) AND message type
      registry entry domain field
    - Domain isolation by default: handlers can only process messages from their domain
    - Cross-domain consumption requires explicit opt-in via allowed_cross_domains
    - Clear fail-fast validation at startup time

Topic Domain Extraction:
    Domain is extracted from the first segment after the namespace prefix:
    - "onex.registration.events" -> domain = "registration"
    - "dev.user.events.v1" -> domain = "user"
    - "prod.order.commands.v2" -> domain = "order"

Related:
    - OMN-937: Central Message Type Registry implementation
    - RegistryMessageType: Uses domain constraints for validation

.. versionadded:: 0.5.0
"""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation import ModelValidationOutcome


class ModelDomainConstraint(BaseModel):
    """
    Domain constraint configuration for message type routing.

    Defines which domains a handler can consume messages from and enforces
    domain ownership rules. By default, handlers are isolated to their own
    domain and must explicitly opt-in to cross-domain message consumption.

    Attributes:
        owning_domain: The domain that owns this handler/message type.
            Must match the domain extracted from topic for same-domain processing.
            Example: "registration", "user", "order"
        allowed_cross_domains: Set of domains this handler can consume from
            in addition to its own domain. Empty by default (domain isolation).
            Example: {"user", "order"} allows consuming from user.* and order.* topics
        allow_all_domains: If True, allows consumption from any domain.
            Use sparingly - only for truly domain-agnostic handlers.
            Overrides allowed_cross_domains if True.
        require_explicit_opt_in: If True, cross-domain consumption raises error
            unless explicitly allowed. Default True for security.

    Example:
        >>> # Registration handler only processes registration messages
        >>> constraint = ModelDomainConstraint(owning_domain="registration")
        >>> constraint.can_consume_from("registration")
        True
        >>> constraint.can_consume_from("user")
        False

        >>> # Handler with cross-domain access
        >>> constraint = ModelDomainConstraint(
        ...     owning_domain="notification",
        ...     allowed_cross_domains={"user", "order"},
        ... )
        >>> constraint.can_consume_from("user")
        True
        >>> constraint.can_consume_from("billing")
        False

    Thread Safety:
        This model is immutable (frozen=True) and thread-safe for concurrent access.

    .. versionadded:: 0.5.0
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    owning_domain: str = Field(
        ...,
        description=(
            "The domain that owns this handler/message type. "
            "Must match topic domain for same-domain processing."
        ),
        min_length=1,
        max_length=100,
    )

    allowed_cross_domains: frozenset[str] = Field(
        default_factory=frozenset,
        description=(
            "Domains this handler can consume from in addition to owning_domain. "
            "Empty by default for domain isolation."
        ),
    )

    allow_all_domains: bool = Field(
        default=False,
        description=(
            "If True, allows consumption from any domain. "
            "Overrides allowed_cross_domains. Use sparingly."
        ),
    )

    require_explicit_opt_in: bool = Field(
        default=True,
        description=(
            "If True, cross-domain consumption raises error unless explicitly "
            "allowed via allowed_cross_domains or allow_all_domains."
        ),
    )

    def can_consume_from(self, topic_domain: str) -> bool:
        """
        Check if this constraint allows consuming messages from the given domain.

        Args:
            topic_domain: The domain extracted from the topic being consumed.

        Returns:
            True if consumption is allowed, False otherwise.

        Example:
            >>> constraint = ModelDomainConstraint(owning_domain="registration")
            >>> constraint.can_consume_from("registration")
            True
            >>> constraint.can_consume_from("user")
            False

        .. versionadded:: 0.5.0
        """
        # Allow all domains if flag is set
        if self.allow_all_domains:
            return True

        # Always allow own domain
        if topic_domain == self.owning_domain:
            return True

        # Check explicit cross-domain allowlist
        if topic_domain in self.allowed_cross_domains:
            return True

        return False

    def validate_consumption(
        self,
        topic_domain: str,
        message_type: str,
    ) -> ModelValidationOutcome:
        """
        Validate if consumption is allowed and return detailed error if not.

        This method provides detailed validation with error messages suitable
        for fail-fast behavior at startup time.

        Args:
            topic_domain: The domain extracted from the topic.
            message_type: The message type being consumed (for error messages).

        Returns:
            ModelValidationOutcome with is_valid=True if valid,
            or is_valid=False with error_message if invalid.

        Example:
            >>> constraint = ModelDomainConstraint(owning_domain="registration")
            >>> outcome = constraint.validate_consumption("user", "UserCreated")
            >>> outcome.is_valid
            False

        .. versionadded:: 0.5.0
        .. versionchanged:: 0.6.0
            Return type changed from tuple[bool, str | None] to ModelValidationOutcome.
        """
        if self.can_consume_from(topic_domain):
            return ModelValidationOutcome.success()

        # Build detailed error message
        if self.require_explicit_opt_in:
            allowed = [self.owning_domain]
            allowed.extend(sorted(self.allowed_cross_domains))
            return ModelValidationOutcome.failure(
                f"Domain mismatch: handler in domain '{self.owning_domain}' cannot "
                f"consume message type '{message_type}' from domain '{topic_domain}'. "
                f"Allowed domains: {allowed}. "
                f"To enable cross-domain consumption, add '{topic_domain}' to "
                f"allowed_cross_domains or set allow_all_domains=True."
            )
        else:
            return ModelValidationOutcome.failure(
                f"Domain mismatch: handler domain '{self.owning_domain}' does not "
                f"match topic domain '{topic_domain}' for message type '{message_type}'."
            )


__all__ = ["ModelDomainConstraint"]
