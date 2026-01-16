# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol interface for policy plugins in ONEX Infrastructure.

This module defines the ProtocolPolicy interface for implementing pure decision
logic plugins that can be registered with the RegistryPolicy. Policy plugins are
the cornerstone of ONEX's extensible decision-making architecture.

**CRITICAL: Policy plugins are PURE decision logic only.**

Policy plugins MUST NOT:
    - Perform I/O operations (file, network, database)
    - Have side effects (state mutation outside return values)
    - Make external service calls
    - Log at runtime (logging during registration/validation is acceptable)
    - Depend on mutable global state

Policy plugins MUST:
    - Be deterministic given the same input
    - Return decisions via the return value only
    - Be synchronous by default (async only with explicit flag and justification)
    - Be idempotent - same input always produces same output
    - Be thread-safe for concurrent evaluation

Policy Types and Use Cases:

**Orchestrator Policies** (policy_type="orchestrator"):
    Control workflow execution and coordination logic:
    - Step ordering: Determine sequence of workflow steps
    - Retry selection: Choose which retries to attempt and when
    - Backoff calculation: Compute delay between retry attempts
    - Routing decisions: Select which handlers/nodes to invoke
    - Timeout budgeting: Allocate time across workflow steps
    - Circuit breaker logic: Decide when to open/close circuits
    - Load balancing: Distribute work across available resources
    - Priority scheduling: Order tasks by importance/urgency

**Reducer Policies** (policy_type="reducer"):
    Control state aggregation and transformation logic:
    - State merge strategy: How to combine multiple state updates
    - Conflict resolution: Handle concurrent state modifications
    - Idempotency rules: Ensure operations can be safely repeated
    - Projection logic: Transform state for different views
    - Redaction/filtering: Remove sensitive data from state
    - Validation rules: Verify state consistency
    - Compaction strategies: Reduce state size over time
    - Event sourcing: Apply events to rebuild state

Example Usage:
    ```python
    from omnibase_infra.runtime.protocol_policy import (
        PolicyContext,
        PolicyResult,
        ProtocolPolicy,
    )
    from omnibase_infra.enums import EnumPolicyType

    class ExponentialBackoffPolicy:
        '''Policy for calculating exponential backoff delays.'''

        @property
        def policy_id(self) -> str:
            return "exponential_backoff_v1"

        @property
        def policy_type(self) -> EnumPolicyType:
            return EnumPolicyType.ORCHESTRATOR  # Recommended for type safety

        def evaluate(self, context: PolicyContext) -> PolicyResult:
            '''Calculate backoff delay based on retry attempt.'''
            attempt = int(context.get("attempt", 0))
            base_delay = float(context.get("base_delay_seconds", 1.0))
            max_delay = float(context.get("max_delay_seconds", 60.0))

            delay = min(base_delay * (2 ** attempt), max_delay)
            return PolicyResult(
                delay_seconds=delay,
                should_retry=attempt < 10,
            )

        def decide(self, context: PolicyContext) -> PolicyResult:
            '''Alias for evaluate() - delegates to evaluate().'''
            return self.evaluate(context)

    # Type checking works via Protocol
    policy: ProtocolPolicy = ExponentialBackoffPolicy()
    context = PolicyContext(attempt=3, base_delay_seconds=1.0)
    result = policy.evaluate(context)
    ```

Integration with RegistryPolicy:
    ```python
    from omnibase_infra.runtime.registry_policy import RegistryPolicy
    from omnibase_infra.enums import EnumPolicyType

    registry = RegistryPolicy()
    registry.register_policy(
        policy_id="exponential_backoff_v1",
        policy_class=ExponentialBackoffPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )

    # Retrieve and use policy
    policy_cls = registry.get("exponential_backoff_v1")
    policy = policy_cls()
    decision = policy.evaluate(context)
    ```

See Also:
    - RegistryPolicy: Registry for managing policy plugins (SINGLE SOURCE OF TRUTH)
    - ProtocolHandler: Protocol for I/O handlers (contrast with pure policies)
    - ModelPolicyContext: Structured context model for policy evaluation
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.runtime.models.model_policy_context import ModelPolicyContext
from omnibase_infra.runtime.models.model_policy_result import ModelPolicyResult

__all__ = [
    "ModelPolicyContext",
    "ModelPolicyResult",
    "PolicyContext",
    "PolicyResult",
    "ProtocolPolicy",
]

# Type aliases for cleaner protocol signatures
# These provide semantic naming for policy evaluation types
PolicyContext = ModelPolicyContext
"""Type alias providing semantic naming for policy context.

This alias provides a cleaner type name for protocol signatures:
    def evaluate(self, context: PolicyContext) -> PolicyResult

Uses ModelPolicyContext Pydantic model, which provides:
    - Structured context with common fields (correlation_id, attempt, etc.)
    - extra="allow" for policy-specific parameters
    - Dict-like access via get() for compatibility
"""

PolicyResult = ModelPolicyResult
"""Type alias providing semantic naming for policy results.

This alias provides a cleaner type name for protocol signatures:
    def evaluate(self, context: PolicyContext) -> PolicyResult

Uses ModelPolicyResult Pydantic model, which provides:
    - Structured result with common fields (success, reason, etc.)
    - extra="allow" for policy-specific return values
    - Dict-like access via get() for compatibility
"""


@runtime_checkable
class ProtocolPolicy(Protocol):
    """Protocol interface for policy plugins implementing pure decision logic.

    Policy plugins provide pluggable decision-making capabilities without side
    effects. They receive a context dictionary and return a decision dictionary.

    This protocol enables:
        - Dependency injection of decision logic
        - Test isolation through mock policies
        - Runtime policy swapping without code changes
        - Composition of multiple policies

    Thread Safety:
        Implementations MUST be thread-safe for concurrent calls.

        **Guarantees implementers MUST provide:**
            - The evaluate() and decide() methods are safe for concurrent calls
            - No mutable instance state that could cause race conditions
            - Pure functions with no side effects

        **Locking recommendations:**
            - Policies SHOULD be stateless (no locking needed)
            - If state is required (e.g., caching), use threading.Lock
            - Avoid asyncio.Lock since policies are synchronous by design

        **What callers can assume:**
            - Multiple threads can call evaluate() concurrently
            - Same input always produces same output (determinism)
            - No side effects from policy evaluation

    Determinism:
        Implementations MUST be deterministic. Given identical context input,
        the policy MUST always return the same decision output.

    Performance:
        Implementations SHOULD be fast. Policy evaluation happens on the critical
        path of workflow execution. Target sub-millisecond evaluation times.

    Attributes:
        policy_id: Unique identifier for this policy (e.g., "exponential_backoff_v1")
        policy_type: Policy category - either "orchestrator" or "reducer"

    Example:
        ```python
        class CircuitBreakerPolicy:
            '''Decides whether to open/close circuit based on failure rate.'''

            @property
            def policy_id(self) -> str:
                return "circuit_breaker_v1"

            @property
            def policy_type(self) -> str:
                return "orchestrator"

            def evaluate(self, context: PolicyContext) -> PolicyResult:
                failure_count = int(context.get("failure_count", 0))
                threshold = int(context.get("threshold", 5))
                return PolicyResult(
                    circuit_open=failure_count >= threshold,
                    failure_count=failure_count,
                )
        ```
    """

    @property
    def policy_id(self) -> str:
        """Unique identifier for this policy.

        The policy_id is used for:
            - Registry lookup and retrieval
            - Logging and observability
            - Configuration references
            - Versioning (recommended to include version suffix)

        Naming Conventions:
            - Use snake_case (e.g., "exponential_backoff")
            - Include version suffix (e.g., "_v1", "_v2")
            - Be descriptive but concise
            - Avoid special characters except underscore

        Returns:
            Unique string identifier for this policy instance.

        Example:
            ```python
            @property
            def policy_id(self) -> str:
                return "weighted_retry_selection_v1"
            ```
        """
        ...

    @property
    def policy_type(self) -> Literal["orchestrator", "reducer"] | EnumPolicyType:
        """Policy category indicating usage context.

        The policy_type determines where this policy can be used:
            - "orchestrator" or EnumPolicyType.ORCHESTRATOR: Workflow coordination and execution control
            - "reducer" or EnumPolicyType.REDUCER: State aggregation and transformation

        This categorization enables:
            - Type-safe policy lookup by category
            - Validation that policies are used appropriately
            - Clear documentation of policy purpose

        Returns:
            Either a string literal ("orchestrator" or "reducer") or EnumPolicyType value.

        Example:
            ```python
            from omnibase_infra.enums import EnumPolicyType

            # Option 1: Using EnumPolicyType (Recommended)
            @property
            def policy_type(self) -> EnumPolicyType:
                return EnumPolicyType.ORCHESTRATOR

            # Option 2: Using string literal (for backward compatibility)
            @property
            def policy_type(self) -> Literal["orchestrator", "reducer"]:
                return "orchestrator"
            ```

            Note: Choose ONE of the above options, not both.
        """
        ...

    def evaluate(self, context: PolicyContext) -> PolicyResult:
        """Evaluate the policy with the given context and return a decision.

        This is the primary method for policy execution. It receives contextual
        information and returns a decision model that the caller can act upon.

        **CRITICAL: This method MUST be pure.**

        Pure means:
            - No I/O operations (files, network, database)
            - No side effects (logging, state mutation)
            - No external service calls
            - Deterministic output for identical input
            - Thread-safe for concurrent calls

        Args:
            context: ModelPolicyContext containing evaluation context.
                Supports arbitrary fields via extra="allow". Common fields:
                - For orchestrator policies: attempt, error_type, elapsed_ms
                - For reducer policies: current_state, event, timestamp
                Access fields via attribute access or dict-like get():
                    context.attempt or context.get("attempt", 0)

        Returns:
            ModelPolicyResult containing the policy decision.
                Supports arbitrary fields via extra="allow". Common patterns:
                - For retry policies: should_retry, delay_seconds
                - For routing policies: target_handler, priority
                - For merge policies: merged_state, conflicts

        Example:
            ```python
            def evaluate(self, context: PolicyContext) -> PolicyResult:
                '''Decide retry behavior based on error type.'''
                error_type = str(context.get("error_type", "unknown"))
                attempt = int(context.get("attempt", 0))

                retryable_errors = {"timeout", "connection_error", "rate_limit"}
                should_retry = error_type in retryable_errors and attempt < 3

                return PolicyResult(
                    should_retry=should_retry,
                    reason=f"error_type={error_type}, attempt={attempt}",
                )
            ```
        """
        ...

    def decide(self, context: PolicyContext) -> PolicyResult:
        """Alias for evaluate() - provided for semantic clarity.

        Some use cases read more naturally with "decide" rather than "evaluate".
        This method is semantically identical to evaluate().

        **NOTE: This method is optional.** If your policy only implements evaluate(),
        the RegistryPolicy will work correctly. The decide() method is provided as
        a convenience for policies that prefer this semantic naming.

        Args:
            context: ModelPolicyContext containing evaluation context.

        Returns:
            ModelPolicyResult containing the policy decision.

        Example:
            ```python
            # These are equivalent:
            result = policy.evaluate(context)
            result = policy.decide(context)
            ```

        Note:
            Default implementations should delegate to evaluate():
            ```python
            def decide(self, context: PolicyContext) -> PolicyResult:
                return self.evaluate(context)
            ```
        """
        ...
