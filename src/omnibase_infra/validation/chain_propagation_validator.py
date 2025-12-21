# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chain Propagation Validator for Correlation and Causation Chain Validation.

Validates that messages properly maintain correlation and causation chains
during propagation through the ONEX event-driven system. This ensures
workflow traceability and supports distributed debugging.

Design Principles:
    - **Workflow Traceability**: All messages in a workflow share the same
      correlation_id for end-to-end trace visibility.
    - **Causation Chain Integrity**: Each message's causation_id must reference
      its direct parent's message_id, forming an unbroken lineage.
    - **Fail-Open Architecture**: Follows ONEX validation philosophy - validation
      failures are reported but don't block by default. Use enforce_chain_propagation()
      for strict enforcement.

Chain Rules:
    1. **Correlation Propagation**: Child messages must inherit the parent's
       correlation_id exactly. A mismatch breaks trace correlation.
    2. **Causation Chain**: Every produced message's causation_id must equal
       its direct parent's message_id. This creates parent-child relationships.
    3. **No Skipped Ancestors** (strict mode only): In strict pairwise validation
       via ``validate_chain()``, causation chains must be continuous - a message
       cannot skip its direct parent to reference a grandparent.

    Note: ``validate_workflow_chain()`` intentionally relaxes Rule 3 to allow
    ancestor skipping for workflow flexibility (fan-out patterns, aggregation,
    partial chain reconstruction). See its docstring for details.

Message ID Semantics:
    In ONEX, the ModelEventEnvelope uses:
    - envelope_id: Unique identifier for each message (serves as message_id)
    - correlation_id: Shared across all messages in a workflow
    - causation_id: Optional field referencing parent's envelope_id

    For envelopes without an explicit causation_id field, the validator
    checks the metadata for a 'causation_id' key or 'parent_message_id' key.

Causation ID Semantics:
    **Canonical Location**: When producing child messages, set causation_id in
    ``metadata.tags["causation_id"]`` as a string UUID. This is the preferred
    location for new ONEX implementations.

    **Why metadata.tags?** The ModelEventEnvelope's metadata.tags dict provides
    a flexible, schema-stable location for tracing metadata. Unlike first-class
    envelope fields (which require schema changes), tags can be extended without
    breaking compatibility.

    **Producer Responsibility**: When creating a child envelope from a parent:

    .. code-block:: python

        child_envelope = ModelEventEnvelope(
            payload=child_payload,
            correlation_id=parent_envelope.correlation_id,  # Propagate correlation
            metadata=ModelEventMetadata(
                tags={
                    "causation_id": str(parent_envelope.envelope_id),  # Canonical
                },
            ),
        )

    **Fallback Locations**: The validator also checks these locations for
    backwards compatibility with legacy systems and external integrations:

    - ``envelope.causation_id`` - Direct attribute (if envelope model has it)
    - ``metadata.tags["parent_message_id"]`` - Legacy terminology alias
    - ``metadata.headers["x-causation-id"]`` - HTTP transport convention
    - ``metadata.headers["causation-id"]`` - Alternate HTTP header name
    - ``metadata.headers["x-parent-message-id"]`` - Legacy HTTP header

    These fallbacks exist to support interoperability with older ONEX versions,
    HTTP-based message transports, and external event producers that may use
    different naming conventions.

Thread Safety:
    The ChainPropagationValidator is stateless and thread-safe. All validation
    methods are pure functions that produce fresh result objects.

Usage:
    >>> from omnibase_infra.validation.chain_propagation_validator import (
    ...     ChainPropagationValidator,
    ...     validate_message_chain,
    ...     enforce_chain_propagation,
    ... )
    >>> from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    >>> from uuid import uuid4
    >>>
    >>> # Direct validation
    >>> validator = ChainPropagationValidator()
    >>> violations = validator.validate_chain(parent_envelope, child_envelope)
    >>> if violations:
    ...     for v in violations:
    ...         print(v.format_for_logging())
    >>>
    >>> # Strict enforcement
    >>> enforce_chain_propagation(parent_envelope, child_envelope)

Related:
    - OMN-951: Enforce Correlation and Causation Chain Validation
    - docs/patterns/correlation_id_tracking.md

.. versionadded:: 0.5.0
"""

from __future__ import annotations

__all__ = [
    "ChainPropagationValidator",
    "ChainPropagationError",
    "validate_message_chain",
    "validate_linear_message_chain",
    "enforce_chain_propagation",
]

from typing import cast
from uuid import UUID

# ModelEventEnvelope is used at runtime in function parameter types, not just for type hints
from omnibase_core.models.events.model_event_envelope import (  # noqa: TC002
    ModelEventEnvelope,
)

from omnibase_infra.enums.enum_chain_violation_type import EnumChainViolationType
from omnibase_infra.errors.error_chain_propagation import ChainPropagationError
from omnibase_infra.errors.model_infra_error_context import ModelInfraErrorContext
from omnibase_infra.models.validation.model_chain_violation import ModelChainViolation

# ==============================================================================
# Helper Functions for Envelope Field Access
# ==============================================================================


def _get_message_id(envelope: ModelEventEnvelope[object]) -> UUID:
    """Get the message_id from an envelope.

    In ONEX, the envelope_id serves as the unique message identifier.

    Args:
        envelope: The event envelope.

    Returns:
        The envelope's unique identifier (envelope_id).
    """
    # envelope_id is typed as UUID in ModelEventEnvelope
    return cast(UUID, envelope.envelope_id)


def _get_correlation_id(envelope: ModelEventEnvelope[object]) -> UUID | None:
    """Get the correlation_id from an envelope.

    Args:
        envelope: The event envelope.

    Returns:
        The envelope's correlation_id, or None if not set.
    """
    # correlation_id is typed as UUID | None in ModelEventEnvelope
    correlation_id = envelope.correlation_id
    if correlation_id is None:
        return None
    return cast(UUID, correlation_id)


def _get_causation_id(envelope: ModelEventEnvelope[object]) -> UUID | None:
    """Get the causation_id from an envelope.

    Canonical Location:
        The **canonical location** for causation_id is ``metadata.tags["causation_id"]``
        stored as a string UUID. When creating child envelopes, producers SHOULD set
        causation_id in this location for consistency across the ONEX ecosystem.

        Example of canonical usage when producing a child message::

            child_envelope = ModelEventEnvelope(
                # ... other fields ...
                metadata=ModelEventMetadata(
                    tags={
                        "causation_id": str(parent_envelope.envelope_id),
                    },
                ),
            )

    Fallback Locations (Backwards Compatibility):
        This function checks multiple locations for interoperability with legacy
        systems and external message producers that may use different conventions:

        1. **Direct attribute** ``envelope.causation_id`` (UUID) - Some envelope
           implementations may expose causation_id as a first-class attribute.

        2. **Metadata tags** ``metadata.tags["causation_id"]`` (string -> UUID) -
           **Canonical location**. Preferred for new implementations.

        3. **Metadata tags** ``metadata.tags["parent_message_id"]`` (string -> UUID) -
           Legacy alias. Some systems use "parent_message_id" terminology.

        4. **Metadata headers** with keys ``x-causation-id``, ``causation-id``,
           ``x-parent-message-id`` (string -> UUID) - HTTP-style headers for
           interoperability with HTTP-based message transports.

    Best Practices:
        - **Producers**: Always set ``metadata.tags["causation_id"]`` when creating
          child envelopes. This ensures consistent behavior across validators.
        - **Consumers**: Use this function for extraction - it handles all formats.
        - **Migrations**: When migrating from legacy formats, populate both the
          canonical location and the legacy location during the transition period.

    Args:
        envelope: The event envelope.

    Returns:
        The envelope's causation_id, or None if not set in any checked location.

    See Also:
        - Module docstring "Causation ID Semantics" section for architectural context
        - ``docs/patterns/correlation_id_tracking.md`` for full tracing patterns
    """
    # Check for direct attribute
    if hasattr(envelope, "causation_id"):
        causation_id = envelope.causation_id
        if isinstance(causation_id, UUID):
            return causation_id

    # Check metadata for causation_id or parent_message_id
    # Note: metadata.headers or metadata.tags might contain this info
    if hasattr(envelope, "metadata") and envelope.metadata is not None:
        metadata = envelope.metadata

        # Check if metadata has a tags dict with causation info
        if hasattr(metadata, "tags") and metadata.tags:
            tags = metadata.tags
            for key in ("causation_id", "parent_message_id"):
                if key in tags:
                    value = tags[key]
                    if isinstance(value, UUID):
                        return value
                    if isinstance(value, str):
                        try:
                            return UUID(value)
                        except ValueError:
                            pass

        # Check headers dict as well
        if hasattr(metadata, "headers") and metadata.headers:
            headers = metadata.headers
            for key in ("x-causation-id", "causation-id", "x-parent-message-id"):
                if key in headers:
                    value = headers[key]
                    if isinstance(value, str):
                        try:
                            return UUID(value)
                        except ValueError:
                            pass

    return None


# ==============================================================================
# Chain Propagation Validator
# ==============================================================================


class ChainPropagationValidator:
    """Validates correlation and causation chain propagation.

    Enforces workflow traceability rules:
    1. All messages in a workflow share the same correlation_id
    2. Every produced message has causation_id = parent.message_id
    3. Causation chains are local (no skipping ancestors)

    Attributes:
        None - the validator is stateless.

    Thread Safety:
        ChainPropagationValidator instances are stateless and thread-safe.
        All validation methods are pure functions that produce fresh result
        objects. Multiple threads can safely call any validation method on
        the same instance concurrently.

    Example:
        >>> validator = ChainPropagationValidator()
        >>>
        >>> # Validate single parent-child relationship
        >>> violations = validator.validate_chain(parent, child)
        >>>
        >>> # Validate entire workflow chain
        >>> violations = validator.validate_workflow_chain([msg1, msg2, msg3])
    """

    def validate_correlation_propagation(
        self,
        parent_envelope: ModelEventEnvelope[object],
        child_envelope: ModelEventEnvelope[object],
    ) -> list[ModelChainViolation]:
        """Validate that child message inherits parent's correlation_id.

        All messages in a workflow must share the same correlation_id to
        enable end-to-end distributed tracing. This method checks that
        the child's correlation_id matches the parent's correlation_id.

        Args:
            parent_envelope: The parent message envelope.
            child_envelope: The child message envelope produced from parent.

        Returns:
            List containing a single CORRELATION_MISMATCH violation if the
            correlation_ids don't match, or an empty list if valid.

        Example:
            >>> validator = ChainPropagationValidator()
            >>> violations = validator.validate_correlation_propagation(parent, child)
            >>> if violations:
            ...     print("Correlation chain broken!")
        """
        violations: list[ModelChainViolation] = []

        parent_correlation = _get_correlation_id(parent_envelope)
        child_correlation = _get_correlation_id(child_envelope)

        # If parent has a correlation_id, child must have the same
        if parent_correlation is not None:
            if child_correlation is None:
                violations.append(
                    ModelChainViolation(
                        violation_type=EnumChainViolationType.CORRELATION_MISMATCH,
                        expected_value=parent_correlation,
                        actual_value=None,
                        message_id=_get_message_id(child_envelope),
                        parent_message_id=_get_message_id(parent_envelope),
                        violation_message=(
                            "Child message is missing correlation_id but parent has one. "
                            "All messages in a workflow must share the same correlation_id."
                        ),
                        severity="error",
                    )
                )
            elif child_correlation != parent_correlation:
                violations.append(
                    ModelChainViolation(
                        violation_type=EnumChainViolationType.CORRELATION_MISMATCH,
                        expected_value=parent_correlation,
                        actual_value=child_correlation,
                        message_id=_get_message_id(child_envelope),
                        parent_message_id=_get_message_id(parent_envelope),
                        violation_message=(
                            "Child message has different correlation_id than parent. "
                            "All messages in a workflow must share the same correlation_id."
                        ),
                        severity="error",
                    )
                )

        return violations

    def validate_causation_chain(
        self,
        parent_envelope: ModelEventEnvelope[object],
        child_envelope: ModelEventEnvelope[object],
    ) -> list[ModelChainViolation]:
        """Validate that child's causation_id equals parent's message_id.

        Each message's causation_id must reference its direct parent's
        message_id to form an unbroken lineage back to the workflow origin.

        Args:
            parent_envelope: The parent message envelope.
            child_envelope: The child message envelope produced from parent.

        Returns:
            List containing a CAUSATION_CHAIN_BROKEN violation if the
            causation_id doesn't match parent's message_id, or an empty
            list if valid.

        Example:
            >>> validator = ChainPropagationValidator()
            >>> violations = validator.validate_causation_chain(parent, child)
            >>> if violations:
            ...     print("Causation chain broken!")
        """
        violations: list[ModelChainViolation] = []

        parent_message_id = _get_message_id(parent_envelope)
        child_causation_id = _get_causation_id(child_envelope)

        # Child's causation_id must equal parent's message_id
        if child_causation_id is None:
            # Missing causation_id is a chain break
            violations.append(
                ModelChainViolation(
                    violation_type=EnumChainViolationType.CAUSATION_CHAIN_BROKEN,
                    expected_value=parent_message_id,
                    actual_value=None,
                    message_id=_get_message_id(child_envelope),
                    parent_message_id=parent_message_id,
                    violation_message=(
                        "Child message is missing causation_id. "
                        "Every message must reference its parent's message_id "
                        "to maintain causation chain integrity."
                    ),
                    severity="error",
                )
            )
        elif child_causation_id != parent_message_id:
            violations.append(
                ModelChainViolation(
                    violation_type=EnumChainViolationType.CAUSATION_CHAIN_BROKEN,
                    expected_value=parent_message_id,
                    actual_value=child_causation_id,
                    message_id=_get_message_id(child_envelope),
                    parent_message_id=parent_message_id,
                    violation_message=(
                        "Child message's causation_id does not match parent's message_id. "
                        "Every message must reference its direct parent's message_id."
                    ),
                    severity="error",
                )
            )

        return violations

    def validate_chain(
        self,
        parent_envelope: ModelEventEnvelope[object],
        child_envelope: ModelEventEnvelope[object],
    ) -> list[ModelChainViolation]:
        """Validate both correlation and causation chain propagation.

        Runs both correlation propagation and causation chain validation,
        returning a combined list of all detected violations.

        Args:
            parent_envelope: The parent message envelope.
            child_envelope: The child message envelope produced from parent.

        Returns:
            Combined list of all chain violations detected. Empty list
            if the chain propagation is valid.

        Example:
            >>> validator = ChainPropagationValidator()
            >>> violations = validator.validate_chain(parent, child)
            >>> for v in violations:
            ...     print(f"[{v.severity}] {v.violation_type.value}")
        """
        violations: list[ModelChainViolation] = []

        # Validate correlation propagation
        violations.extend(
            self.validate_correlation_propagation(parent_envelope, child_envelope)
        )

        # Validate causation chain
        violations.extend(
            self.validate_causation_chain(parent_envelope, child_envelope)
        )

        return violations

    def validate_workflow_chain(
        self,
        envelopes: list[ModelEventEnvelope[object]],
    ) -> list[ModelChainViolation]:
        """Validate an entire chain of messages in a workflow.

        Validates that:
        1. All messages share the same correlation_id (if first message has one)
        2. Each message's causation_id references an ancestor message within
           the provided chain (not necessarily the direct predecessor)
        3. Parent messages appear before child messages in the list order

        The envelopes list should be ordered by causation (parent before child).

        Ancestor Skipping (Intentional Design Decision):
            This method validates that causation_ids reference messages **within**
            the chain, but does NOT enforce direct parent-child ordering. A message
            may reference any ancestor in the chain (e.g., msg3 can reference msg1
            even if msg2 exists between them). This is an intentional design
            decision that provides workflow flexibility for:

            - **Partial chain reconstruction**: When only a subset of messages
              is available for validation (e.g., from logs or replay)
            - **Fan-out patterns**: When a parent spawns multiple children that
              all reference it directly rather than forming a linear chain
            - **Aggregation patterns**: When reducers aggregate from multiple
              ancestors within the same correlation context

            For strict direct parent-child validation (enforcing linear chains),
            use pairwise ``validate_chain()`` calls:

            .. code-block:: python

                # Strict linear chain validation
                for i in range(len(envelopes) - 1):
                    violations.extend(
                        validator.validate_chain(envelopes[i], envelopes[i + 1])
                    )

        Args:
            envelopes: Ordered list of message envelopes in the workflow.
                Should be ordered such that each message's causation_id
                references a message earlier in the list.

        Returns:
            List of all chain violations detected across the workflow.
            Empty list if the entire workflow chain is valid.

        Example:
            >>> validator = ChainPropagationValidator()
            >>> # Workflow with messages that may reference any ancestor
            >>> violations = validator.validate_workflow_chain([msg1, msg2, msg3])
            >>> blocking = [v for v in violations if v.is_blocking()]
            >>> if blocking:
            ...     raise ChainPropagationError(blocking)
        """
        violations: list[ModelChainViolation] = []

        if len(envelopes) < 2:
            # Single message or empty list - no chain to validate
            return violations

        # Build message_id index for quick lookup
        message_id_to_envelope: dict[UUID, ModelEventEnvelope[object]] = {}
        for env in envelopes:
            message_id_to_envelope[_get_message_id(env)] = env

        # Get the reference correlation_id from the first message
        reference_correlation_id = _get_correlation_id(envelopes[0])

        # Validate each message in the chain
        for i, envelope in enumerate(envelopes):
            message_id = _get_message_id(envelope)

            # 1. Validate correlation_id consistency
            envelope_correlation_id = _get_correlation_id(envelope)
            if reference_correlation_id is not None:
                if envelope_correlation_id is None:
                    violations.append(
                        ModelChainViolation(
                            violation_type=EnumChainViolationType.CORRELATION_MISMATCH,
                            expected_value=reference_correlation_id,
                            actual_value=None,
                            message_id=message_id,
                            parent_message_id=None,
                            violation_message=(
                                f"Message at index {i} is missing correlation_id "
                                f"but workflow uses correlation_id={reference_correlation_id}. "
                                "All messages in a workflow must share the same correlation_id."
                            ),
                            severity="error",
                        )
                    )
                elif envelope_correlation_id != reference_correlation_id:
                    violations.append(
                        ModelChainViolation(
                            violation_type=EnumChainViolationType.CORRELATION_MISMATCH,
                            expected_value=reference_correlation_id,
                            actual_value=envelope_correlation_id,
                            message_id=message_id,
                            parent_message_id=None,
                            violation_message=(
                                f"Message at index {i} has different correlation_id "
                                f"than workflow's reference. All messages must share the same "
                                "correlation_id for distributed tracing."
                            ),
                            severity="error",
                        )
                    )

            # 2. Validate causation chain (skip first message - it's the root)
            if i > 0:
                causation_id = _get_causation_id(envelope)

                if causation_id is None:
                    # Non-root message must have causation_id
                    violations.append(
                        ModelChainViolation(
                            violation_type=EnumChainViolationType.CAUSATION_CHAIN_BROKEN,
                            expected_value=None,  # Can't determine expected without causation
                            actual_value=None,
                            message_id=message_id,
                            parent_message_id=None,
                            violation_message=(
                                f"Message at index {i} is missing causation_id. "
                                "Every message (except root) must reference its parent's "
                                "message_id to maintain causation chain."
                            ),
                            severity="error",
                        )
                    )
                # Check if causation_id references a message in the chain
                elif causation_id not in message_id_to_envelope:
                    violations.append(
                        ModelChainViolation(
                            violation_type=EnumChainViolationType.CAUSATION_ANCESTOR_SKIPPED,
                            expected_value=None,
                            actual_value=causation_id,
                            message_id=message_id,
                            parent_message_id=causation_id,
                            violation_message=(
                                f"Message at index {i} has causation_id={causation_id} "
                                "which references a message not in this workflow chain. "
                                "Causation chains must form an unbroken sequence."
                            ),
                            severity="error",
                        )
                    )
                else:
                    # Check that causation_id references an earlier message
                    parent_envelope = message_id_to_envelope[causation_id]
                    parent_idx = envelopes.index(parent_envelope)

                    if parent_idx >= i:
                        # Parent appears after child in the list - order violation
                        violations.append(
                            ModelChainViolation(
                                violation_type=EnumChainViolationType.CAUSATION_CHAIN_BROKEN,
                                expected_value=None,
                                actual_value=causation_id,
                                message_id=message_id,
                                parent_message_id=causation_id,
                                violation_message=(
                                    f"Message at index {i} references parent at index {parent_idx} "
                                    "but parents must appear before children in the causation chain. "
                                    "Check message ordering."
                                ),
                                severity="warning",
                            )
                        )

        return violations

    def validate_linear_workflow_chain(
        self,
        envelopes: list[ModelEventEnvelope[object]],
    ) -> list[ModelChainViolation]:
        """Validate strict linear chain (no ancestor skipping).

        Unlike validate_workflow_chain() which allows ancestor skipping,
        this method enforces that each message's causation_id references
        the immediately preceding message (direct parent).

        Use this method when you need to verify a strict linear workflow
        where messages form a single unbroken chain:
        msg1 -> msg2 -> msg3 -> msg4

        For workflows with fan-out patterns or aggregation, use
        validate_workflow_chain() instead.

        Args:
            envelopes: Ordered list of message envelopes in the workflow.
                Each message at index i+1 must have causation_id equal to
                the envelope_id of message at index i.

        Returns:
            List of all chain violations detected. Empty list if the
            entire linear chain is valid.

        Example:
            >>> validator = ChainPropagationValidator()
            >>> # Strict linear chain - each message must reference direct parent
            >>> violations = validator.validate_linear_workflow_chain([msg1, msg2, msg3])
            >>> if violations:
            ...     print("Linear chain broken!")
        """
        violations: list[ModelChainViolation] = []

        if len(envelopes) < 2:
            return violations

        for i in range(len(envelopes) - 1):
            violations.extend(self.validate_chain(envelopes[i], envelopes[i + 1]))

        return violations


# ==============================================================================
# Module-Level Singleton Validator
# ==============================================================================
#
# Performance Optimization: ChainPropagationValidator is stateless after
# initialization. Creating new instances on every validation call is wasteful.
# Instead, we use a module-level singleton.
#
# Why a singleton is safe here:
# - The validator is completely stateless (no mutable state)
# - All validation methods are pure functions that produce new results
# - Multiple threads can safely use the same validator instance

_default_validator = ChainPropagationValidator()


# ==============================================================================
# Convenience Functions
# ==============================================================================


def validate_message_chain(
    parent_envelope: ModelEventEnvelope[object],
    child_envelope: ModelEventEnvelope[object],
) -> list[ModelChainViolation]:
    """Validate chain propagation between parent and child messages.

    Convenience function that validates both correlation and causation
    chain propagation using the default singleton validator.

    Args:
        parent_envelope: The parent message envelope.
        child_envelope: The child message envelope produced from parent.

    Returns:
        List of chain violations detected. Empty list if valid.

    Example:
        >>> violations = validate_message_chain(parent, child)
        >>> if violations:
        ...     for v in violations:
        ...         print(v.format_for_logging())
    """
    return _default_validator.validate_chain(parent_envelope, child_envelope)


def validate_linear_message_chain(
    envelopes: list[ModelEventEnvelope[object]],
) -> list[ModelChainViolation]:
    """Validate strict linear chain using default validator.

    Convenience function for validate_linear_workflow_chain() that uses
    the module-level singleton validator. Validates that each message
    in the chain references its immediate predecessor.

    Unlike validate_workflow_chain() which allows ancestor skipping,
    this function enforces strict linear ordering where each message's
    causation_id must equal the envelope_id of the immediately preceding
    message.

    Args:
        envelopes: Ordered list of message envelopes in the workflow.
            Each message at index i+1 must have causation_id equal to
            the envelope_id of message at index i.

    Returns:
        List of all chain violations detected. Empty list if the
        entire linear chain is valid.

    Example:
        >>> violations = validate_linear_message_chain([msg1, msg2, msg3])
        >>> if violations:
        ...     for v in violations:
        ...         print(v.format_for_logging())
    """
    return _default_validator.validate_linear_workflow_chain(envelopes)


def enforce_chain_propagation(
    parent_envelope: ModelEventEnvelope[object],
    child_envelope: ModelEventEnvelope[object],
) -> None:
    """Validate chain propagation and raise error if violations found.

    Strict enforcement function that validates both correlation and causation
    chain propagation, raising ChainPropagationError if any violations are
    detected.

    Args:
        parent_envelope: The parent message envelope.
        child_envelope: The child message envelope produced from parent.

    Raises:
        ChainPropagationError: If any chain violations are detected.
            Contains the list of violations for inspection.

    Example:
        >>> try:
        ...     enforce_chain_propagation(parent, child)
        ...     print("Chain propagation valid")
        ... except ChainPropagationError as e:
        ...     print(f"Invalid: {len(e.violations)} violations")
        ...     for v in e.violations:
        ...         print(f"  - {v.violation_type.value}: {v.violation_message}")
    """
    violations = _default_validator.validate_chain(parent_envelope, child_envelope)

    if violations:
        # Use parent's correlation_id for error tracking
        context = ModelInfraErrorContext(
            operation="enforce_chain_propagation",
            correlation_id=_get_correlation_id(parent_envelope),
        )
        raise ChainPropagationError(
            message="Chain propagation validation failed",
            violations=violations,
            context=context,
        )
