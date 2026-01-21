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
    checks the metadata for a 'causation_id' key.

Causation ID Semantics:
    **Canonical Location**: When producing child messages, set causation_id in
    ``metadata.tags["causation_id"]`` as a string UUID. For HTTP transports,
    use ``metadata.headers["x-causation-id"]``.

    **Why metadata.tags?** The ModelEventEnvelope's metadata.tags dict provides
    a flexible, schema-stable location for tracing metadata.

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

Thread Safety:
    The ChainPropagationValidator is stateless and thread-safe. All validation
    methods are pure functions that produce fresh result objects.

Typing Note (ModelEventEnvelope[object]):
    Functions in this module use ``ModelEventEnvelope[object]`` instead of ``Any``
    per CLAUDE.md guidance: "Use ``object`` for generic payloads".

    This is intentional:
    - CLAUDE.md mandates "NEVER use ``Any``" for type annotations
    - Chain validation operates on envelope metadata (correlation_id, causation_id,
      envelope_id), not payload content - the payload type is irrelevant
    - Using ``object`` signals "any object payload" while maintaining type safety
      (unlike ``Any`` which completely disables type checking)
    - Validators that need to inspect payload content should use ``isinstance()``
      type guards for runtime safety

Performance Considerations:
    The validator does not cache results. This is an intentional design decision:

    - **Production Use**: Workflows are validated once per dispatch, making caching
      unnecessary overhead. Each message passes through validation exactly once.
    - **Stateless Design**: Caching would introduce state, complicating thread safety
      and increasing memory footprint without meaningful performance benefit.
    - **Testing/Debugging**: For repeated validation of the same message sets during
      debugging, callers can implement their own memoization if needed.

    If profiling reveals validation as a bottleneck (unlikely given O(n) complexity),
    consider batching validations rather than adding caching.

Usage:
    >>> from omnibase_infra.validation.validator_chain_propagation import (
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

import logging
from typing import cast
from uuid import UUID

# ModelEventEnvelope is used at runtime in function parameter types, not just for type hints
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumChainViolationType, EnumValidationSeverity
from omnibase_infra.errors.error_chain_propagation import ChainPropagationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.validation.model_chain_violation import ModelChainViolation

logger = logging.getLogger(__name__)

__all__ = [
    "CAUSATION_ID_HEADER_KEYS",
    # Causation ID lookup key constants
    "CAUSATION_ID_TAG_KEYS",
    "ChainPropagationError",
    "ChainPropagationValidator",
    "enforce_chain_propagation",
    "get_causation_id",
    "get_correlation_id",
    # Helper functions for envelope field access
    "get_message_id",
    "validate_linear_message_chain",
    "validate_message_chain",
]

# ==============================================================================
# Causation ID Lookup Keys
# ==============================================================================
# These constants define the keys checked when resolving causation_id from
# envelope metadata.

CAUSATION_ID_TAG_KEYS: tuple[str, ...] = ("causation_id",)
"""Keys checked in metadata.tags for causation_id.

The canonical location for causation_id is metadata.tags["causation_id"].
"""

CAUSATION_ID_HEADER_KEYS: tuple[str, ...] = ("x-causation-id",)
"""Keys checked in metadata.headers for causation_id.

The canonical HTTP header for causation_id is "x-causation-id".
"""

# ==============================================================================
# Helper Functions for Envelope Field Access
# ==============================================================================


def get_message_id(envelope: ModelEventEnvelope[object]) -> UUID:
    """Get the message_id from an envelope.

    In ONEX, the envelope_id serves as the unique message identifier.

    Args:
        envelope: The event envelope.

    Returns:
        The envelope's unique identifier (envelope_id).
    """
    # envelope_id is typed as UUID in ModelEventEnvelope, but mypy sees it as Any
    # due to the generic type parameter. Cast is required for type safety.
    return cast("UUID", envelope.envelope_id)


def get_correlation_id(envelope: ModelEventEnvelope[object]) -> UUID | None:
    """Get the correlation_id from an envelope.

    Args:
        envelope: The event envelope.

    Returns:
        The envelope's correlation_id, or None if not set.
    """
    # correlation_id is typed as UUID | None in ModelEventEnvelope, but mypy sees it
    # as Any due to the generic type parameter. Cast is required for type safety.
    correlation_id = envelope.correlation_id
    if correlation_id is None:
        return None
    return cast("UUID", correlation_id)


def get_causation_id(envelope: ModelEventEnvelope[object]) -> UUID | None:
    """Get the causation_id from an envelope.

    Canonical Locations:
        The **canonical location** for causation_id is ``metadata.tags["causation_id"]``
        stored as a string UUID. When creating child envelopes, producers MUST set
        causation_id in this location for consistency across the ONEX ecosystem.

        For HTTP transports, use ``metadata.headers["x-causation-id"]``.

        Example of canonical usage when producing a child message::

            child_envelope = ModelEventEnvelope(
                # ... other fields ...
                metadata=ModelEventMetadata(
                    tags={
                        "causation_id": str(parent_envelope.envelope_id),
                    },
                ),
            )

    Lookup Order:
        1. **Direct attribute** ``envelope.causation_id`` (UUID) - If envelope
           exposes causation_id as a first-class attribute.

        2. **Metadata tags** ``metadata.tags["causation_id"]`` (string -> UUID) -
           Canonical location.

        3. **Metadata headers** ``metadata.headers["x-causation-id"]`` (string -> UUID) -
           Canonical HTTP header.

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

    # Check metadata for causation_id
    if hasattr(envelope, "metadata") and envelope.metadata is not None:
        metadata = envelope.metadata

        # Check metadata.tags for causation_id (canonical location)
        if hasattr(metadata, "tags") and metadata.tags:
            tags = metadata.tags
            for key in CAUSATION_ID_TAG_KEYS:
                if key in tags:
                    value = tags[key]
                    if isinstance(value, UUID):
                        return value
                    if isinstance(value, str):
                        try:
                            return UUID(value)
                        except ValueError:
                            logger.debug(
                                "Malformed UUID in tags key '%s': %s",
                                key,
                                value,
                            )

        # Check metadata.headers for x-causation-id (HTTP transport)
        if hasattr(metadata, "headers") and metadata.headers:
            headers = metadata.headers
            for key in CAUSATION_ID_HEADER_KEYS:
                if key in headers:
                    value = headers[key]
                    if isinstance(value, str):
                        try:
                            return UUID(value)
                        except ValueError:
                            logger.debug(
                                "Malformed UUID in headers key '%s': %s",
                                key,
                                value,
                            )

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

        parent_correlation = get_correlation_id(parent_envelope)
        child_correlation = get_correlation_id(child_envelope)

        # If parent has a correlation_id, child must have the same
        if parent_correlation is not None:
            if child_correlation is None:
                violations.append(
                    ModelChainViolation(
                        violation_type=EnumChainViolationType.CORRELATION_MISMATCH,
                        expected_value=parent_correlation,
                        actual_value=None,
                        message_id=get_message_id(child_envelope),
                        parent_message_id=get_message_id(parent_envelope),
                        violation_message=(
                            f"Child message is missing correlation_id but parent has "
                            f"correlation_id={parent_correlation}. "
                            "All messages in a workflow must share the same correlation_id."
                        ),
                        severity=EnumValidationSeverity.ERROR,
                    )
                )
            elif child_correlation != parent_correlation:
                violations.append(
                    ModelChainViolation(
                        violation_type=EnumChainViolationType.CORRELATION_MISMATCH,
                        expected_value=parent_correlation,
                        actual_value=child_correlation,
                        message_id=get_message_id(child_envelope),
                        parent_message_id=get_message_id(parent_envelope),
                        violation_message=(
                            f"Correlation ID mismatch: expected={parent_correlation}, "
                            f"actual={child_correlation}. "
                            "All messages in a workflow must share the same correlation_id."
                        ),
                        severity=EnumValidationSeverity.ERROR,
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

        parent_message_id = get_message_id(parent_envelope)
        child_causation_id = get_causation_id(child_envelope)

        # Child's causation_id must equal parent's message_id
        if child_causation_id is None:
            # Missing causation_id is a chain break
            violations.append(
                ModelChainViolation(
                    violation_type=EnumChainViolationType.CAUSATION_CHAIN_BROKEN,
                    expected_value=parent_message_id,
                    actual_value=None,
                    message_id=get_message_id(child_envelope),
                    parent_message_id=parent_message_id,
                    violation_message=(
                        f"Child message is missing causation_id. "
                        f"Expected causation_id={parent_message_id} (parent's message_id). "
                        "Every message must reference its parent's message_id "
                        "to maintain causation chain integrity."
                    ),
                    severity=EnumValidationSeverity.ERROR,
                )
            )
        elif child_causation_id != parent_message_id:
            violations.append(
                ModelChainViolation(
                    violation_type=EnumChainViolationType.CAUSATION_CHAIN_BROKEN,
                    expected_value=parent_message_id,
                    actual_value=child_causation_id,
                    message_id=get_message_id(child_envelope),
                    parent_message_id=parent_message_id,
                    violation_message=(
                        f"Causation ID mismatch: expected={parent_message_id}, "
                        f"actual={child_causation_id}. "
                        "Every message must reference its direct parent's message_id."
                    ),
                    severity=EnumValidationSeverity.ERROR,
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

        # Build message_id -> envelope and message_id -> index mappings for O(1) lookup
        # This avoids O(n) list.index() calls inside the O(n) validation loop
        message_id_to_envelope: dict[UUID, ModelEventEnvelope[object]] = {}
        message_id_to_index: dict[UUID, int] = {}
        for idx, env in enumerate(envelopes):
            msg_id = get_message_id(env)
            message_id_to_envelope[msg_id] = env
            message_id_to_index[msg_id] = idx

        # Get the reference correlation_id from the first message
        reference_correlation_id = get_correlation_id(envelopes[0])

        # Validate each message in the chain
        for i, envelope in enumerate(envelopes):
            message_id = get_message_id(envelope)

            # 1. Validate correlation_id consistency
            envelope_correlation_id = get_correlation_id(envelope)
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
                            severity=EnumValidationSeverity.ERROR,
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
                                f"Message at index {i} has correlation_id mismatch: "
                                f"expected={reference_correlation_id}, actual={envelope_correlation_id}. "
                                "All messages must share the same correlation_id for distributed tracing."
                            ),
                            severity=EnumValidationSeverity.ERROR,
                        )
                    )

            # 2. Validate causation chain (skip first message - it's the root)
            if i > 0:
                causation_id = get_causation_id(envelope)

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
                                f"Message at index {i} (message_id={message_id}) is missing "
                                "causation_id. Every message (except root) must reference its "
                                "parent's message_id to maintain causation chain."
                            ),
                            severity=EnumValidationSeverity.ERROR,
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
                                f"Message at index {i} (message_id={message_id}) has "
                                f"causation_id={causation_id} which references a message "
                                "not in this workflow chain. "
                                "Causation chains must form an unbroken sequence."
                            ),
                            severity=EnumValidationSeverity.ERROR,
                        )
                    )
                else:
                    # Check that causation_id references an earlier message
                    # Use O(1) dict lookup instead of O(n) list.index()
                    parent_idx = message_id_to_index[causation_id]

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
                                    f"Message at index {i} (message_id={message_id}) references "
                                    f"parent at index {parent_idx} (causation_id={causation_id}) "
                                    "but parents must appear before children in the causation chain. "
                                    "Check message ordering."
                                ),
                                severity=EnumValidationSeverity.WARNING,
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
            correlation_id=get_correlation_id(parent_envelope),
        )
        raise ChainPropagationError(
            message="Chain propagation validation failed",
            violations=violations,
            context=context,
        )
