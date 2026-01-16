# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Registry - SINGLE SOURCE OF TRUTH for policy plugin registration.

This module provides the RegistryPolicy class for registering and resolving
pure decision policy plugins in the ONEX infrastructure layer.

The registry is responsible for:
- Registering policy plugins by (policy_id, policy_type, version) tuple
- Resolving policy classes for specific policy configurations
- Thread-safe registration operations
- Listing all registered policies
- Enforcing synchronous-by-default policy execution

Design Principles:
- Single source of truth: All policy registrations go through this registry
- Sync enforcement: Async policies must be explicitly flagged
- Type-safe: Full typing for policy registrations (no Any types)
- Thread-safe: Registration operations protected by lock
- Testable: Easy to mock and test policy configurations

Policy Categories (by policy type):
- Orchestrator policies: Workflow coordination, retry strategies, routing
- Reducer policies: State aggregation, conflict resolution, projections

CRITICAL: Policy plugins are PURE decision logic only.

Policy plugins MUST NOT:
    - Perform I/O operations (file, network, database)
    - Have side effects (state mutation outside return values)
    - Make external service calls
    - Log at runtime
    - Depend on mutable global state

Example Usage:
    ```python
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.runtime.registry_policy import RegistryPolicy, ModelPolicyRegistration
    from omnibase_infra.runtime.util_container_wiring import wire_infrastructure_services
    from omnibase_infra.enums import EnumPolicyType

    # Container-based DI (preferred)
    container = ModelONEXContainer()
    await wire_infrastructure_services(container)
    registry = await container.service_registry.resolve_service(RegistryPolicy)

    # Register a synchronous policy using the model (PREFERRED API)
    registration = ModelPolicyRegistration(
        policy_id="exponential_backoff",
        policy_class=ExponentialBackoffPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )
    registry.register(registration)

    # Register using convenience method (preserves original API)
    # Note: For new code, prefer register(ModelPolicyRegistration(...)) instead
    registry.register_policy(
        policy_id="async_merge",
        policy_class=AsyncMergePolicy,
        policy_type=EnumPolicyType.REDUCER,
        version="1.0.0",
        allow_async=True,  # MUST be explicit for async policies
    )

    # Retrieve a policy
    policy_cls = registry.get("exponential_backoff")
    policy = policy_cls()
    result = policy.evaluate(context)

    # List all policies
    policies = registry.list_keys()  # [(id, type, version), ...]
    ```

Integration Points:
- RuntimeHostProcess uses this registry to discover and instantiate policies
- Policies are loaded based on contract definitions
- Supports hot-reload patterns for development
"""

from __future__ import annotations

import functools
import threading
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

from omnibase_core.errors import ModelOnexError
from omnibase_core.models.primitives import ModelSemVer
from pydantic import ValidationError

from omnibase_infra.enums import EnumInfraTransportType, EnumPolicyType
from omnibase_infra.errors import PolicyRegistryError, ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.runtime.mixin_policy_validation import MixinPolicyValidation
from omnibase_infra.runtime.mixin_semver_cache import MixinSemverCache
from omnibase_infra.runtime.models import ModelPolicyKey, ModelPolicyRegistration
from omnibase_infra.runtime.util_version import normalize_version
from omnibase_infra.types import PolicyTypeInput

if TYPE_CHECKING:
    from omnibase_infra.runtime.protocol_policy import ProtocolPolicy


# =============================================================================
# Policy Registry
# =============================================================================


class RegistryPolicy(MixinPolicyValidation, MixinSemverCache):
    """SINGLE SOURCE OF TRUTH for policy plugin registration in omnibase_infra.

    Thread-safe registry for policy plugins. Manages pure decision logic plugins
    that can be used by orchestrator and reducer nodes.

    The registry maintains a mapping from ModelPolicyKey instances to policy classes
    that implement the ProtocolPolicy protocol. ModelPolicyKey provides strong typing
    and replaces the legacy tuple[str, str, str] pattern.

    Container Integration:
        RegistryPolicy is designed to be managed by ModelONEXContainer from omnibase_core.
        Use container_wiring.wire_infrastructure_services() to register RegistryPolicy
        in the container, then resolve it via:

        ```python
        from omnibase_core.container import ModelONEXContainer
        from omnibase_infra.runtime.registry_policy import RegistryPolicy

        # Resolve from container (preferred) - async in omnibase_core v0.5.6+
        registry = await container.service_registry.resolve_service(RegistryPolicy)

        # Or use helper function (also async)
        from omnibase_infra.runtime.util_container_wiring import get_policy_registry_from_container
        registry = await get_policy_registry_from_container(container)
        ```

    Thread Safety:
        All registration operations are protected by a threading.Lock to ensure
        thread-safe access in concurrent environments.

    Sync Enforcement:
        By default, policies must be synchronous. If a policy has async methods
        (evaluate, decide, reduce), registration will fail unless
        allow_async=True is explicitly specified.

    Scale and Performance Characteristics:

        Expected Registry Scale:
            - Typical ONEX system: 20-50 unique policies across 2-5 versions each
            - Medium deployment: 50-100 policies across 3-8 versions each
            - Large deployment: 100-200 policies across 5-10 versions each
            - Stress tested: 500+ total registrations (100 policies x 5 versions)

            Policy categories (by policy_type):
                - Orchestrator policies: Workflow coordination, retry strategies, routing
                - Reducer policies: State aggregation, conflict resolution, projections

            Typical distribution: 60% orchestrator policies, 40% reducer policies

        Performance Characteristics:

            Primary Operations:
                - register(): O(1) - Direct dictionary insert with secondary index update
                - get(policy_id): O(1) best case, O(k) average, O(k*log k) filtered worst case
                    where k = number of matching versions after filtering
                    - Uses secondary index (_policy_id_index) for O(1) policy_id lookup
                    - Fast path (no filters, single version): O(1) direct lookup
                    - Multi-version (no filters): O(k) to find max version via comparison
                    - Filtered path (policy_type + multi-version): O(k*log k) for filter + semver sort
                    - Deferred error generation: Expensive _list_internal() only on error
                    - Cached semver parsing: LRU cache (128 entries) avoids re-parsing

                - is_registered(): O(k) where k = versions for policy_id
                - list_keys(): O(n*log n) where n = total registrations (full scan + sort)
                - list_versions(): O(k) where k = versions for policy_id
                - unregister(): O(k) where k = versions for policy_id

            Benchmark Results (500 policy registrations):
                - 1000 sequential get() calls: < 100ms (< 0.1ms per lookup)
                - 1000 concurrent get() calls (10 threads): < 500ms
                - 100 failed lookups (missing policy_id): < 500ms (early exit optimization)
                - Fast path speedup vs filtered path: > 1.1x

            Lock Contention:
                - Read operations (get, is_registered): Hold lock during lookup only
                - Write operations (register, unregister): Hold lock for full operation
                - Critical sections minimized to reduce contention
                - Expected concurrent throughput: > 2000 reads/sec under 10-thread load

        Memory Footprint:

            Per Policy Registration:
                - ModelPolicyKey: ~200 bytes (3 strings: policy_id, policy_type, version)
                - Policy class reference: 8 bytes (Python object pointer)
                - Secondary index entry: ~50 bytes (list entry + key reference)
                - Total per registration: ~260 bytes

            Estimated Registry Memory:
                - 50 registrations: ~13 KB
                - 100 registrations: ~26 KB
                - 500 registrations: ~130 KB
                - 1000 registrations: ~260 KB

            Cache Overhead:
                - Semver LRU cache: 128 entries x ~100 bytes = ~12.8 KB
                - Total with cache: Registry memory + 12.8 KB

            Note: Memory footprint is negligible compared to typical ONEX process memory
            (100-500 MB). Registry memory is not a bottleneck in production systems.

        Secondary Indexes (Performance Optimization):

            Current Indexes:
                - _policy_id_index: Maps policy_id -> list[ModelPolicyKey]
                    - Purpose: O(1) lookup by policy_id (avoids O(n) scan of all registrations)
                    - Updated on: register(), unregister()
                    - Memory: ~50 bytes per policy_id + 8 bytes per version
                    - Hit rate: 100% for all get() operations

            When to Add Additional Indexes:

                Consider _policy_type_index if:
                    - Frequent list_keys(policy_type=...) calls (currently O(n))
                    - Deployment has > 500 total registrations
                    - Profiling shows list_keys filtering as bottleneck

                Consider _version_index if:
                    - Frequent cross-policy version queries
                    - Complex version-based policy routing logic
                    - Deployment has > 10 versions per policy on average

                Trade-off Analysis:
                    - Each index adds ~50-100 bytes per entry
                    - Benefits: O(n) -> O(1) for filtered queries
                    - Costs: Write amplification (update multiple indexes per register/unregister)
                    - Recommendation: Profile first, optimize only if proven bottleneck

        Monitoring Recommendations:

            Key Metrics to Track:
                1. Registry size: len(registry) - Track growth over time
                2. Lookup latency: Time for get() operations (p50, p95, p99)
                3. Lookup errors: PolicyRegistryError frequency (indicates config issues)
                4. Cache hit rate: LRU cache effectiveness (_parse_semver cache)
                5. Lock contention: Concurrent access patterns and throughput

            Performance Thresholds (alert if exceeded):
                - Average get() latency: > 1ms (indicates potential lock contention)
                - P99 get() latency: > 10ms (indicates blocking on write operations)
                - Registry size: > 1000 registrations (may need index optimization)
                - Cache miss rate: > 10% (indicates cache size insufficient)
                - Concurrent throughput: < 1000 reads/sec (indicates lock bottleneck)

            Recommended Instrumentation:
                ```python
                import time
                from omnibase_core.metrics import histogram, counter

                # In production RegistryPolicy wrapper:
                start = time.perf_counter()
                policy_cls = registry.get(policy_id)
                histogram("policy_registry.get_latency_ms", (time.perf_counter() - start) * 1000)
                counter("policy_registry.get_total")

                # Track registry growth:
                histogram("policy_registry.size", len(registry))
                ```

            Health Check Integration:
                - Include len(registry) in health check response
                - Alert if registry empty (indicates bootstrap failure)
                - Alert if registry size changes unexpectedly (> 20% delta)

    Trust Model and Security Considerations:

        RegistryPolicy performs LIMITED validation on registered policy classes.
        This section documents what guarantees exist and what the caller is
        responsible for.

        VALIDATED (by RegistryPolicy):
            - Async method detection: Policies with async methods (reduce, decide,
              evaluate) must explicitly set allow_async=True. This prevents
              accidental async policy registration that could cause runtime issues.
            - Policy type validation: policy_type must be a valid EnumPolicyType
              value ("orchestrator" or "reducer"). Invalid types raise PolicyRegistryError.
            - Version format validation: version must be valid semver format
              (e.g., "1.0.0", "1.2.3-beta"). Invalid formats raise ProtocolConfigurationError.
            - Non-empty policy_id: Validated via ModelPolicyRegistration Pydantic model.
            - Thread-safe registration: Registration operations are protected by lock.

        NOT VALIDATED (caller's responsibility):
            - Policy class correctness: The registry does not verify that a policy
              class correctly implements ProtocolPolicy methods. A class missing
              required methods will only fail at runtime when invoked.
            - Policy class safety: No static analysis, sandboxing, or security
              scanning is performed. Malicious code in a policy class will execute
              with the same privileges as the host process.
            - Policy behavior: The registry cannot validate that policy decision
              logic is correct, deterministic, or free of bugs.
            - Policy dependencies: Import-time side effects, malicious dependencies,
              or resource-intensive imports are not prevented.
            - Runtime behavior: Policies that hang, exhaust memory, raise unexpected
              exceptions, or violate timeouts are not sandboxed.
            - Idempotency: The registry does not verify that policies are idempotent
              or safe to retry.

        Trust Assumptions:
            1. Policy classes come from TRUSTED sources only:
               - Internal codebase modules
               - Vetted first-party packages
               - Audited third-party packages
            2. Policy classes do not execute arbitrary code on registration:
               - No __init_subclass__ side effects
               - No metaclass execution during class reference
               - No import-time network calls or file I/O
            3. Policy instances are created and used within the same trust boundary:
               - No cross-tenant policy sharing
               - No user-provided policy classes at runtime
            4. Policy implementers follow the purity contract:
               - No I/O operations (file, network, database)
               - No side effects (state mutation outside return values)
               - No external service calls
               - No runtime logging (use structured outputs only)

        For High-Security Environments:
            If deploying RegistryPolicy in environments with stricter security
            requirements, consider implementing additional safeguards:

            - Code Review: Mandatory review for all policy implementations before
              registration approval.

            - Static Analysis: Run linters and security scanners on policy modules
              before allowing registration:
              ```python
              # Example: Pre-registration validation hook
              def validate_policy_module(module_path: str) -> bool:
                  # Run bandit, semgrep, or custom security checks
                  result = run_security_scan(module_path)
                  return result.passed
              ```

            - Allowlist Pattern: Maintain an explicit allowlist of approved policy_ids
              and reject registration attempts for unlisted policies:
              ```python
              APPROVED_POLICIES = {"exponential_backoff", "rate_limiter", "retry_strategy"}

              def register_with_allowlist(registration: ModelPolicyRegistration) -> None:
                  if registration.policy_id not in APPROVED_POLICIES:
                      raise PolicyRegistryError(
                          f"Policy '{registration.policy_id}' not in approved list",
                          policy_id=registration.policy_id,
                      )
                  registry.register(registration)
              ```

            - Sandboxing: Execute policy code in isolated environments (not built
              into RegistryPolicy, requires external infrastructure):
              - Process isolation (subprocess with resource limits)
              - Container isolation (Docker with security profiles)
              - WASM isolation (for extreme security requirements)

            - Runtime Monitoring: Instrument policy execution with timeouts and
              resource monitoring:
              ```python
              async def execute_policy_with_limits(
                  policy: ProtocolPolicy,
                  context: dict,
                  timeout_seconds: float = 1.0,
              ) -> PolicyResult:
                  try:
                      return await asyncio.wait_for(
                          policy.evaluate(context),
                          timeout=timeout_seconds,
                      )
                  except asyncio.TimeoutError:
                      raise PolicyRegistryError(
                          f"Policy '{policy.policy_id}' exceeded timeout",
                          policy_id=policy.policy_id,
                      )
              ```

        Safe Usage Patterns:

            DO:
                - Register policies from known, reviewed source modules
                - Use container-based DI for better lifecycle management
                - Document policy dependencies and requirements
                - Test policies in isolation before registration
                - Monitor policy execution metrics (latency, error rates)

            DON'T:
                - Register policy classes provided by untrusted users
                - Allow dynamic policy class construction from user input
                - Skip code review for new policy implementations
                - Assume policies are safe because they're in the registry
                - Share registries across trust boundaries

        See Also:
            - docs/patterns/policy_registry_trust_model.md for detailed security guide
            - ProtocolPolicy for interface requirements
            - ModelPolicyRegistration for registration model validation

    Attributes:
        _registry: Internal dictionary mapping ModelPolicyKey instances to policy classes
        _lock: Threading lock for thread-safe registration operations
        _policy_id_index: Secondary index for O(1) policy_id lookup

    Inherited from MixinSemverCache:
        SEMVER_CACHE_SIZE: Class variable for configuring LRU cache size (default: 128)

    Class-Level Configuration:
        The semver parsing cache size can be configured for large deployments:

        ```python
        # Option 1: Set class attribute before first use
        RegistryPolicy.SEMVER_CACHE_SIZE = 256

        # Option 2: Use configure method (recommended)
        RegistryPolicy.configure_semver_cache(maxsize=256)

        # Must be done BEFORE any registry operations
        registry = RegistryPolicy()
        ```

        For testing, use _reset_semver_cache() to clear and reconfigure.

    Example:
        >>> from omnibase_infra.runtime.models import ModelPolicyRegistration
        >>> registry = RegistryPolicy()
        >>> registration = ModelPolicyRegistration(
        ...     policy_id="retry_backoff",
        ...     policy_class=RetryBackoffPolicy,
        ...     policy_type=EnumPolicyType.ORCHESTRATOR,
        ... )
        >>> registry.register(registration)
        >>> policy_cls = registry.get("retry_backoff")
        >>> print(registry.list_keys())
        [('retry_backoff', 'orchestrator', '1.0.0')]
    """

    def __init__(self) -> None:
        """Initialize an empty policy registry with thread lock."""
        # Key: ModelPolicyKey -> policy_class (strong typing replaces tuple pattern)
        self._registry: dict[ModelPolicyKey, type[ProtocolPolicy]] = {}
        self._lock: threading.Lock = threading.Lock()

        # Performance optimization: Secondary indexes for O(1) lookups
        # Maps policy_id -> list of ModelPolicyKey instances
        self._policy_id_index: dict[str, list[ModelPolicyKey]] = {}

    # Note: _validate_protocol_implementation and _validate_sync_enforcement
    # are inherited from MixinPolicyValidation with the correct signatures.
    # Do not override them here.

    def _normalize_policy_type(
        self,
        policy_type: PolicyTypeInput,
    ) -> str:
        """Normalize policy type to string value and validate against EnumPolicyType.

        This method provides centralized policy type validation logic used by all
        registration and query methods. It accepts both EnumPolicyType enum values
        and string literals, normalizing them to their string representation while
        ensuring they match valid EnumPolicyType values.

        Validation Process:
            1. If policy_type is EnumPolicyType instance, extract .value
            2. If policy_type is string, validate against EnumPolicyType values
            3. Raise PolicyRegistryError if string doesn't match any enum value
            4. Return normalized string value

        This centralized validation ensures consistent policy type handling across
        all registry operations (register, get, list_keys, is_registered, unregister).

        Args:
            policy_type: Policy type as EnumPolicyType enum or string literal.
                        Valid values: "orchestrator", "reducer"

        Returns:
            Normalized string value for the policy type (e.g., "orchestrator", "reducer")

        Raises:
            PolicyRegistryError: If policy_type is a string that doesn't match any
                                EnumPolicyType value. Error includes the invalid value
                                and list of valid options.

        Example:
            >>> from omnibase_infra.enums import EnumPolicyType
            >>> registry = RegistryPolicy()
            >>> # Enum to string
            >>> registry._normalize_policy_type(EnumPolicyType.ORCHESTRATOR)
            'orchestrator'
            >>> # Valid string passthrough
            >>> registry._normalize_policy_type("reducer")
            'reducer'
            >>> # Invalid string raises error
            >>> registry._normalize_policy_type("invalid")
            PolicyRegistryError: Invalid policy_type: 'invalid'.
                                 Must be one of: ['orchestrator', 'reducer']
        """
        if isinstance(policy_type, EnumPolicyType):
            return policy_type.value

        # Validate string against enum values
        valid_types = {e.value for e in EnumPolicyType}
        if policy_type not in valid_types:
            raise PolicyRegistryError(
                f"Invalid policy_type: {policy_type!r}. "
                f"Must be one of: {sorted(valid_types)}",
                policy_id=None,
                policy_type=policy_type,
            )

        return policy_type

    @staticmethod
    def _normalize_version(version: str) -> str:
        """Normalize version string for consistent lookups.

        Delegates to the shared normalize_version utility which is the
        SINGLE SOURCE OF TRUTH for version normalization in omnibase_infra.

        This method wraps the shared utility to convert ValueError to
        ProtocolConfigurationError for RegistryPolicy's error contract.

        Normalization rules:
            1. Strip leading/trailing whitespace
            2. Strip leading 'v' or 'V' prefix
            3. Expand partial versions (1 -> 1.0.0, 1.0 -> 1.0.0)
            4. Parse with ModelSemVer.parse() for validation
            5. Preserve prerelease suffix if present

        Args:
            version: The version string to normalize

        Returns:
            Normalized version string in "x.y.z" or "x.y.z-prerelease" format

        Raises:
            ProtocolConfigurationError: If the version format is invalid

        Example:
            >>> RegistryPolicy._normalize_version("1.0")
            '1.0.0'
            >>> RegistryPolicy._normalize_version("v2.1")
            '2.1.0'
        """
        try:
            return normalize_version(version)
        except ValueError as e:
            raise ProtocolConfigurationError(
                str(e),
                version=version,
            ) from e

    def register(
        self,
        registration: ModelPolicyRegistration,
    ) -> None:
        """Register a policy plugin using a registration model.

        Associates a (policy_id, policy_type, version) tuple with a policy class.
        If the combination is already registered, the existing registration is
        overwritten.

        This is the PREFERRED API for registering policies. For new code, use this
        method with ModelPolicyRegistration instead of register_policy().

        Args:
            registration: ModelPolicyRegistration containing all registration parameters:
                - policy_id: Unique identifier for the policy
                - policy_class: The policy class to register (must implement ProtocolPolicy)
                - policy_type: Whether this is orchestrator or reducer policy
                - version: Semantic version string (default: "1.0.0")
                - allow_async: If True, allows async interface

        Raises:
            PolicyRegistryError: If policy has async methods and
                               allow_async=False, or if policy_type is invalid

        Example:
            >>> from omnibase_infra.runtime.models import ModelPolicyRegistration
            >>> registry = RegistryPolicy()
            >>> registration = ModelPolicyRegistration(
            ...     policy_id="retry_backoff",
            ...     policy_class=RetryBackoffPolicy,
            ...     policy_type=EnumPolicyType.ORCHESTRATOR,
            ...     version="1.0.0",
            ... )
            >>> registry.register(registration)
        """
        # Extract fields from model
        policy_id = registration.policy_id
        policy_class = registration.policy_class
        policy_type = registration.policy_type
        version = registration.version
        allow_async = registration.allow_async

        # Validate protocol implementation (evaluate() method exists and is callable)
        # Pass policy_type for complete parameter validation and error context
        self._validate_protocol_implementation(policy_id, policy_class, policy_type)

        # Validate sync enforcement (pass policy_type for better error context)
        self._validate_sync_enforcement(
            policy_id, policy_class, allow_async, policy_type
        )

        # Normalize policy type
        normalized_type = self._normalize_policy_type(policy_type)

        # Normalize version string before storing to prevent lookup mismatches
        # This ensures "1.0", "1.0.0", and "v1.0.0" all resolve to "1.0.0"
        # Note: ModelPolicyRegistration already normalizes, but we normalize again
        # here to guarantee consistency with lookup operations
        normalized_version = self._normalize_version(version)

        # Validate version format (ensures semantic versioning compliance)
        # This calls _parse_semver which will raise ProtocolConfigurationError if invalid
        self._parse_semver(normalized_version)

        # Register the policy using ModelPolicyKey with normalized version
        key = ModelPolicyKey(
            policy_id=policy_id,
            policy_type=normalized_type,
            version=normalized_version,
        )
        with self._lock:
            self._registry[key] = policy_class
            # Update secondary index for performance optimization
            if policy_id not in self._policy_id_index:
                self._policy_id_index[policy_id] = []
            if key not in self._policy_id_index[policy_id]:
                self._policy_id_index[policy_id].append(key)

    def register_policy(
        self,
        policy_id: str,
        policy_class: type[ProtocolPolicy],
        policy_type: PolicyTypeInput,
        version: str = "1.0.0",
        allow_async: bool = False,
    ) -> None:
        """Convenience method to register a policy with individual parameters.

        Wraps parameters in ModelPolicyRegistration and calls register().
        Partial version strings (e.g., "1", "1.0") are auto-normalized to
        "x.y.z" format by ModelPolicyRegistration.

        Note:
            For new code, prefer using register(ModelPolicyRegistration(...))
            directly. This is a convenience method for simple registrations.

        Args:
            policy_id: Unique identifier for the policy (e.g., 'exponential_backoff')
            policy_class: The policy class to register. Must implement ProtocolPolicy.
            policy_type: Whether this is orchestrator or reducer policy.
                        Can be EnumPolicyType or string literal.
            version: Semantic version string (default: "1.0.0"). Partial versions
                    like "1" or "1.0" are auto-normalized to "1.0.0" or "1.0.0".
            allow_async: If True, allows async interface. MUST be explicitly
                                flagged for policies with async methods.

        Raises:
            PolicyRegistryError: If policy has async methods and
                               allow_async=False, or if policy_type is invalid
            ProtocolConfigurationError: If version format is invalid

        Example:
            >>> registry = RegistryPolicy()
            >>> registry.register_policy(
            ...     policy_id="retry_backoff",
            ...     policy_class=RetryBackoffPolicy,
            ...     policy_type=EnumPolicyType.ORCHESTRATOR,
            ...     version="1.0.0",
            ... )
        """
        # Version normalization is handled by ModelPolicyRegistration validator
        # which normalizes partial versions and v-prefixed versions automatically
        try:
            registration = ModelPolicyRegistration(
                policy_id=policy_id,
                policy_class=policy_class,
                policy_type=policy_type,
                version=version,
                allow_async=allow_async,
            )
        except ValidationError as e:
            # Convert all validation errors to ProtocolConfigurationError for consistency
            # This ensures uniform error handling across all validation failures
            for error in e.errors():
                field_loc = error.get("loc", ())
                field_name = field_loc[0] if field_loc else "unknown"
                error_msg = error.get("msg", str(e))

                if field_name == "version":
                    raise ProtocolConfigurationError(
                        f"Invalid version format: {error_msg}",
                        version=version,
                    ) from e
                if field_name == "policy_id":
                    raise ProtocolConfigurationError(
                        f"Invalid policy_id: {error_msg}",
                        policy_id=policy_id,
                    ) from e
                if field_name == "policy_type":
                    raise ProtocolConfigurationError(
                        f"Invalid policy_type: {error_msg}",
                        policy_type=str(policy_type),
                    ) from e
                if field_name == "policy_class":
                    raise ProtocolConfigurationError(
                        f"Invalid policy_class: {error_msg}",
                    ) from e
            # Fallback for any unhandled validation errors
            raise ProtocolConfigurationError(
                f"Validation error in policy registration: {e}",
            ) from e

        self.register(registration)

    def get(
        self,
        policy_id: str,
        policy_type: PolicyTypeInput | None = None,
        version: str | None = None,
    ) -> type[ProtocolPolicy]:
        """Get policy class by ID, type, and optional version.

        Resolves the policy class registered for the given policy configuration.
        If policy_type is not specified, returns the first matching policy_id.
        If version is not specified, returns the latest version (by semantic version).

        Performance Characteristics:
            - Best case: O(1) - Direct lookup with policy_id only (single version, no filters)
            - Average case: O(k) where k = number of matching versions (multi-version, no filters)
            - Worst case: O(k*log(k)) when policy_type filter applied with multiple versions
              (requires both filtering candidates and sorting by semver to find latest)
            - Uses secondary index for O(1) policy_id lookup instead of O(n) scan
            - Defers expensive error message generation until actually needed
            - Fast path optimization when no filters applied (common case)

        Args:
            policy_id: Policy identifier.
            policy_type: Optional policy type filter (orchestrator or reducer).
            version: Optional version filter. If None, returns latest version.

        Returns:
            Policy class registered for the configuration.

        Raises:
            PolicyRegistryError: If no matching policy is found.

        Example:
            >>> registry = RegistryPolicy()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> policy_cls = registry.get("retry")
            >>> policy_cls = registry.get("retry", policy_type="orchestrator")
            >>> policy_cls = registry.get("retry", version="1.0.0")
        """
        # Normalize policy_type if provided (outside lock for minimal critical section)
        # Use empty string as sentinel for "no filter" to reduce union types
        normalized_type: str = (
            self._normalize_policy_type(policy_type) if policy_type is not None else ""
        )

        # Normalize version for consistent lookup (e.g., "1.0" matches "1.0.0")
        normalized_version: str | None = None
        if version is not None:
            normalized_version = self._normalize_version(version)

        with self._lock:
            # Performance optimization: Use secondary index for O(1) lookup by policy_id
            # This avoids iterating through all registry entries (O(n) -> O(1))
            candidate_keys = self._policy_id_index.get(policy_id, [])

            # Early exit if policy_id not found - avoid building matches list
            if not candidate_keys:
                # Defer expensive _list_internal() call until actually raising error
                filters = [f"policy_id={policy_id!r}"]
                if policy_type is not None:
                    filters.append(f"policy_type={policy_type!r}")
                if version is not None:
                    filters.append(f"version={version!r}")

                # Inline list generation for error message (avoids separate method)
                registered = [
                    k.to_tuple()
                    for k in sorted(
                        self._registry.keys(),
                        key=lambda k: (k.policy_id, k.policy_type, k.version),
                    )
                ]
                raise PolicyRegistryError(
                    f"No policy registered matching: {', '.join(filters)}. "
                    f"Registered policies: {registered}",
                    policy_id=policy_id,
                    policy_type=str(policy_type) if policy_type else None,
                )

            # Find matching entries from candidates (optimized to reduce allocations)
            # Fast path: no filtering needed (common case - just get latest version)
            if not normalized_type and normalized_version is None:
                # Fast path optimization: avoid tuple allocation and batch dict lookups
                # Only build the matches list if we have multiple versions
                if len(candidate_keys) == 1:
                    # Single version - direct return without any allocations
                    return self._registry[candidate_keys[0]]
                else:
                    # Multiple versions - need to find latest
                    # Use direct key comparison instead of building tuples
                    latest_key = max(
                        candidate_keys,
                        key=lambda k: self._parse_semver(k.version),
                    )
                    return self._registry[latest_key]
            else:
                # Filtered path: apply type and version filters
                matches = []
                for key in candidate_keys:
                    if normalized_type and key.policy_type != normalized_type:
                        continue
                    if (
                        normalized_version is not None
                        and key.version != normalized_version
                    ):
                        continue
                    matches.append((key, self._registry[key]))

                if not matches:
                    # Filters eliminated all candidates - build error message
                    filters = [f"policy_id={policy_id!r}"]
                    if policy_type is not None:
                        filters.append(f"policy_type={policy_type!r}")
                    if version is not None:
                        filters.append(f"version={version!r}")

                    # Inline list generation for error message (avoids separate method)
                    registered = [
                        k.to_tuple()
                        for k in sorted(
                            self._registry.keys(),
                            key=lambda k: (k.policy_id, k.policy_type, k.version),
                        )
                    ]
                    raise PolicyRegistryError(
                        f"No policy registered matching: {', '.join(filters)}. "
                        f"Registered policies: {registered}",
                        policy_id=policy_id,
                        policy_type=str(policy_type) if policy_type else None,
                    )

                # If version not specified and multiple matches, return latest
                # (using cached semantic version comparison)
                if normalized_version is None and len(matches) > 1:
                    # Sort in-place to avoid allocating a new list
                    matches.sort(
                        key=lambda x: self._parse_semver(x[0].version), reverse=True
                    )

                return matches[0][1]

    # ==========================================================================
    # Semver Cache Configuration Methods
    # ==========================================================================

    @classmethod
    def configure_semver_cache(cls, maxsize: int) -> None:
        """Configure semver cache size. Must be called before first parse.

        This method allows configuring the LRU cache size for semver parsing
        in large deployments with many policy versions. For most deployments,
        the default of 128 entries is sufficient.

        When to Increase Cache Size:
            - Very large deployments with > 100 unique policy versions
            - High-frequency lookups across many version combinations
            - Observed cache eviction causing performance regression

        Args:
            maxsize: Maximum cache entries (default: 128).
                     Recommended range: 64-512 for most deployments.
                     Each entry uses ~100 bytes.

        Raises:
            ProtocolConfigurationError: If cache already initialized (first parse already occurred)

        Example:
            >>> # Configure before any registry operations
            >>> RegistryPolicy.configure_semver_cache(maxsize=256)
            >>> registry = RegistryPolicy()

        Note:
            For testing purposes, use _reset_semver_cache() to clear the cache
            and allow reconfiguration.
        """
        with cls._semver_cache_lock:
            if cls._semver_cache is not None:
                context = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="configure_semver_cache",
                )
                raise ProtocolConfigurationError(
                    "Cannot reconfigure semver cache after first use. "
                    "Set RegistryPolicy.SEMVER_CACHE_SIZE before creating any "
                    "registry instances, or use _reset_semver_cache() for testing.",
                    context=context,
                )
            cls.SEMVER_CACHE_SIZE = maxsize

    @classmethod
    def _reset_semver_cache(cls) -> None:
        """Reset semver cache. For testing only.

        Clears the cached semver parser, allowing reconfiguration of cache size.
        This should only be used in test fixtures to ensure test isolation.

        Thread Safety:
            This method is thread-safe and uses the class-level lock. The reset
            operation is atomic - either the cache is fully reset or not at all.

            In-flight Operations:
                If other threads have already obtained a reference to the cache
                via _get_semver_parser(), they will continue using the old cache
                until they complete. This is safe because the old cache remains
                a valid callable until garbage collected. New operations after
                reset will get the new cache instance when created.

            Memory Reclamation:
                The old cache's internal LRU entries are explicitly cleared via
                cache_clear() before the reference is released. This ensures
                prompt memory reclamation rather than waiting for garbage
                collection.

            Concurrent Reset:
                Multiple concurrent reset calls are safe. Each reset will clear
                the current cache (if any) and set the reference to None. The
                lock ensures only one reset executes at a time.

        Example:
            >>> # In test fixture
            >>> RegistryPolicy._reset_semver_cache()
            >>> RegistryPolicy.SEMVER_CACHE_SIZE = 64
            >>> # Now cache will be initialized with size 64 on next use
        """
        with cls._semver_cache_lock:
            # Clear the inner LRU-cached function (has the actual cache)
            inner_cache = cls._semver_cache_inner
            if inner_cache is not None:
                # Clear internal LRU cache entries before releasing reference.
                # This ensures prompt memory reclamation rather than waiting
                # for garbage collection of the orphaned function object.
                # Note: cache_clear() is added by @lru_cache decorator but not
                # reflected in Callable type annotation. This is a known mypy
                # limitation with lru_cache wrappers.
                inner_cache.cache_clear()  # type: ignore[attr-defined]
            cls._semver_cache = None
            cls._semver_cache_inner = None

    @classmethod
    def _get_semver_parser(cls) -> Callable[[str], ModelSemVer]:
        """Get or create the semver parser with configured cache size.

        This method implements lazy initialization of the LRU-cached semver parser.
        The cache size is determined by SEMVER_CACHE_SIZE at initialization time.

        Thread Safety:
            Uses double-checked locking pattern for thread-safe lazy initialization.
            The fast path stores the cache reference in a local variable to prevent
            TOCTOU (time-of-check-time-of-use) race conditions where another thread
            could call _reset_semver_cache() between the None check and the return.

        Cache Key Normalization:
            Version strings are normalized BEFORE being used as cache keys to ensure
            that equivalent versions (e.g., "1.0" and "1.0.0") share the same cache
            entry. This prevents cache fragmentation and improves hit rates.

        Returns:
            Cached semver parsing function that returns ModelSemVer instances.

        Performance:
            - First call: Creates LRU-cached function (one-time cost)
            - Subsequent calls: Returns cached function reference (O(1))
            - Cache hit rate improved by normalizing keys before lookup
        """
        # Fast path: cache already initialized
        # CRITICAL: Store in local variable to prevent TOCTOU race condition.
        # Without this, another thread could call _reset_semver_cache() between
        # the None check and the return, causing this method to return None.
        cache = cls._semver_cache
        if cache is not None:
            return cache

        # Slow path: initialize with lock
        with cls._semver_cache_lock:
            # Double-check after acquiring lock
            if cls._semver_cache is not None:
                return cls._semver_cache

            # Create LRU-cached parser with configured size
            # The cache key is the NORMALIZED version string to prevent
            # fragmentation (e.g., "1.0" and "1.0.0" share the same entry)
            @functools.lru_cache(maxsize=cls.SEMVER_CACHE_SIZE)
            def _parse_semver_cached(normalized_version: str) -> ModelSemVer:
                """Parse normalized semantic version string into ModelSemVer.

                This function receives ALREADY NORMALIZED version strings.
                The normalization is done by the wrapper function before
                caching to ensure equivalent versions share cache entries.

                Args:
                    normalized_version: Pre-normalized version in "x.y.z" or
                        "x.y.z-prerelease" format

                Returns:
                    ModelSemVer instance for comparison

                Raises:
                    ProtocolConfigurationError: If version format is invalid
                """
                # ModelOnexError is imported at module level
                try:
                    return ModelSemVer.parse(normalized_version)
                except ModelOnexError as e:
                    raise ProtocolConfigurationError(
                        str(e),
                        version=normalized_version,
                    ) from e
                except ValueError as e:
                    raise ProtocolConfigurationError(
                        str(e),
                        version=normalized_version,
                    ) from e

            def _parse_semver_impl(version: str) -> ModelSemVer:
                """Parse semantic version string into ModelSemVer.

                Implementation moved here to support configurable cache size.
                See _parse_semver docstring for full documentation.

                IMPORTANT: This wrapper normalizes version strings BEFORE
                passing to the LRU-cached parsing function. This ensures that
                equivalent versions (e.g., "1.0" and "1.0.0", "v1.0.0" and "1.0.0")
                share the same cache entry, improving cache hit rates.

                All validation (empty strings, prerelease suffix, format) is
                delegated to _normalize_version to eliminate code duplication.
                """
                # Delegate all validation to _normalize_version (single source of truth)
                # This eliminates duplicated validation logic (empty check, prerelease suffix)
                normalized = RegistryPolicy._normalize_version(version)

                # Now call the cached function with the NORMALIZED version
                # This ensures "1.0", "1.0.0", "v1.0.0" all use the same cache entry
                return _parse_semver_cached(normalized)

            # Store both the outer wrapper and inner cached function
            # The wrapper is what callers use (_semver_cache)
            # The inner function is needed for cache_clear() access (_semver_cache_inner)
            cls._semver_cache = _parse_semver_impl
            cls._semver_cache_inner = _parse_semver_cached
            return cls._semver_cache

    @classmethod
    def _parse_semver(cls, version: str) -> ModelSemVer:
        """Parse semantic version string into ModelSemVer for comparison.

        This method implements SEMANTIC VERSION SORTING, not lexicographic sorting.
        This is critical for correct "latest version" selection.

        Why This Matters (PR #36 feedback):
            Lexicographic sorting (string comparison):
                "1.10.0" < "1.9.0" WRONG (because '1' < '9' in strings)
                "10.0.0" < "2.0.0" WRONG (because '1' < '2' in strings)

            Semantic version sorting (integer comparison):
                1.10.0 > 1.9.0 CORRECT (because 10 > 9 as integers)
                10.0.0 > 2.0.0 CORRECT (because 10 > 2 as integers)

        Implementation:
            - Returns ModelSemVer instance with integer major, minor, patch
            - ModelSemVer implements comparison operators for correct ordering
            - Prerelease is parsed but NOT used in comparisons (major.minor.patch only)
            - "1.0.0-alpha" and "1.0.0" compare as EQUAL (same major.minor.patch)

        Supported Formats:
            - Full: "1.2.3", "1.2.3-beta"
            - Partial: "1" -> (1, 0, 0), "1.2" -> (1, 2, 0)
            - Prerelease: "1.0.0-alpha", "2.1.0-rc.1"

        Validation:
            - Rejects empty strings
            - Rejects non-numeric components
            - Rejects negative numbers
            - Rejects >3 version parts (e.g., "1.2.3.4")

        Performance:
            This method uses an LRU cache with configurable size (default: 128)
            to avoid re-parsing the same version strings repeatedly, improving
            performance for lookups that compare multiple versions.

            Cache Size Configuration:
                For large deployments, configure before first use:
                    RegistryPolicy.configure_semver_cache(maxsize=256)

            Cache Size Rationale (default 128):
                - Typical registry: 10-50 unique policy versions
                - Peak scenarios: 50-100 versions across multiple policy types
                - Each cache entry: ~200 bytes (string key + ModelSemVer instance)
                - Total memory: ~25.6KB worst case (negligible overhead)
                - Hit rate: >95% for repeated get() calls with version comparisons
                - Eviction: Rare in practice, LRU ensures least-used versions purged

        Args:
            version: Semantic version string (e.g., "1.2.3" or "1.0.0-beta")

        Returns:
            ModelSemVer instance for comparison.
            Components are INTEGERS (not strings) for correct semantic sorting.
            Prerelease is parsed and stored but ignored in version comparisons.

        Raises:
            ProtocolConfigurationError: If version format is invalid

        Examples:
            >>> RegistryPolicy._parse_semver("1.9.0")
            ModelSemVer(major=1, minor=9, patch=0, prerelease='')
            >>> RegistryPolicy._parse_semver("1.10.0")
            ModelSemVer(major=1, minor=10, patch=0, prerelease='')
            >>> RegistryPolicy._parse_semver("1.10.0") > RegistryPolicy._parse_semver("1.9.0")
            True
            >>> RegistryPolicy._parse_semver("10.0.0") > RegistryPolicy._parse_semver("2.0.0")
            True
            >>> RegistryPolicy._parse_semver("1.0.0-alpha")
            ModelSemVer(major=1, minor=0, patch=0, prerelease='alpha')
            >>> # Prerelease is parsed but NOT used in comparisons:
            >>> RegistryPolicy._parse_semver("1.0.0-alpha") == RegistryPolicy._parse_semver("1.0.0")
            True  # Same major.minor.patch, prerelease ignored
        """
        parser = cls._get_semver_parser()
        return parser(version)

    @classmethod
    def _get_semver_cache_info(cls) -> functools._CacheInfo | None:
        """Get cache statistics for the semver parser. For testing only.

        Returns the cache_info() from the inner LRU-cached function.
        This allows tests to verify cache behavior without accessing
        internal implementation details.

        Returns:
            functools._CacheInfo with hits, misses, maxsize, currsize,
            or None if cache not yet initialized.

        Example:
            >>> RegistryPolicy._reset_semver_cache()
            >>> RegistryPolicy._parse_semver("1.0.0")
            >>> info = RegistryPolicy._get_semver_cache_info()
            >>> info.misses  # First call is a miss
            1
            >>> RegistryPolicy._parse_semver("1.0.0")
            >>> info = RegistryPolicy._get_semver_cache_info()
            >>> info.hits  # Second call is a hit
            1
        """
        if cls._semver_cache_inner is None:
            return None
        # cache_info() is added by @lru_cache decorator
        # The return type is functools._CacheInfo
        result: functools._CacheInfo = cls._semver_cache_inner.cache_info()  # type: ignore[attr-defined]
        return result

    def _list_internal(self) -> list[tuple[str, str, str]]:
        """Internal list method (assumes lock is held).

        Returns:
            List of (policy_id, policy_type, version) tuples.
        """
        return [
            k.to_tuple()
            for k in sorted(
                self._registry.keys(),
                key=lambda k: (k.policy_id, k.policy_type, k.version),
            )
        ]

    def list_keys(
        self,
        policy_type: PolicyTypeInput | None = None,
    ) -> list[tuple[str, str, str]]:
        """List registered policy keys as (id, type, version) tuples.

        Args:
            policy_type: Optional filter to list only policies of a specific type.

        Returns:
            List of (policy_id, policy_type, version) tuples, sorted alphabetically.

        Example:
            >>> registry = RegistryPolicy()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> registry.register("merge", MergePolicy, EnumPolicyType.REDUCER)
            >>> print(registry.list_keys())
            [('merge', 'reducer', '1.0.0'), ('retry', 'orchestrator', '1.0.0')]
            >>> print(registry.list_keys(policy_type="orchestrator"))
            [('retry', 'orchestrator', '1.0.0')]
        """
        # Normalize policy_type if provided
        # Use empty string as sentinel for "no filter" to reduce union types
        normalized_type: str = (
            self._normalize_policy_type(policy_type) if policy_type is not None else ""
        )

        with self._lock:
            results: list[tuple[str, str, str]] = []
            for key in sorted(
                self._registry.keys(),
                key=lambda k: (k.policy_id, k.policy_type, k.version),
            ):
                if normalized_type and key.policy_type != normalized_type:
                    continue
                results.append(key.to_tuple())
            return results

    def list_policy_types(self) -> list[str]:
        """List registered policy types.

        Returns:
            List of unique policy type strings that have registered policies.

        Example:
            >>> registry = RegistryPolicy()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> print(registry.list_policy_types())
            ['orchestrator']
        """
        with self._lock:
            types = {key.policy_type for key in self._registry}
            return sorted(types)

    def list_versions(self, policy_id: str) -> list[str]:
        """List registered versions for a policy ID.

        Args:
            policy_id: The policy ID to list versions for.

        Returns:
            List of version strings registered for the policy ID, sorted.

        Example:
            >>> registry = RegistryPolicy()
            >>> registry.register("retry", RetryPolicyV1, "orchestrator", "1.0.0")
            >>> registry.register("retry", RetryPolicyV2, "orchestrator", "2.0.0")
            >>> print(registry.list_versions("retry"))
            ['1.0.0', '2.0.0']
        """
        with self._lock:
            # Performance optimization: Use secondary index
            candidate_keys = self._policy_id_index.get(policy_id, [])
            versions = {key.version for key in candidate_keys}
            return sorted(versions)

    def is_registered(
        self,
        policy_id: str,
        policy_type: PolicyTypeInput | None = None,
        version: str | None = None,
    ) -> bool:
        """Check if a policy is registered.

        Args:
            policy_id: Policy identifier.
            policy_type: Optional policy type filter.
            version: Optional version filter.

        Returns:
            True if a matching policy is registered, False otherwise.

        Example:
            >>> registry = RegistryPolicy()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> registry.is_registered("retry")
            True
            >>> registry.is_registered("unknown")
            False
        """
        # Normalize policy_type if provided
        # Use empty string as sentinel for "no filter" to reduce union types
        normalized_type: str = ""
        if policy_type is not None:
            try:
                normalized_type = self._normalize_policy_type(policy_type)
            except PolicyRegistryError:
                return False

        # Normalize version for consistent lookup (e.g., "1.0" matches "1.0.0")
        normalized_version: str | None = None
        if version is not None:
            normalized_version = self._normalize_version(version)

        with self._lock:
            # Performance optimization: Use secondary index
            candidate_keys = self._policy_id_index.get(policy_id, [])
            for key in candidate_keys:
                if normalized_type and key.policy_type != normalized_type:
                    continue
                if normalized_version is not None and key.version != normalized_version:
                    continue
                return True
            return False

    def unregister(
        self,
        policy_id: str,
        policy_type: PolicyTypeInput | None = None,
        version: str | None = None,
    ) -> int:
        """Unregister policy plugins.

        Removes policy registrations matching the given criteria.
        This is useful for testing and hot-reload scenarios.

        Args:
            policy_id: Policy identifier to unregister.
            policy_type: Optional policy type filter.
            version: Optional version filter.

        Returns:
            Number of policies unregistered.

        Example:
            >>> registry = RegistryPolicy()
            >>> registry.register("retry", RetryPolicyV1, "orchestrator", "1.0.0")
            >>> registry.register("retry", RetryPolicyV2, "orchestrator", "2.0.0")
            >>> registry.unregister("retry")  # Removes all versions
            2
            >>> registry.unregister("retry", version="1.0.0")  # Remove specific version
            1
        """
        # Normalize policy_type if provided
        # Use empty string as sentinel for "no filter" to reduce union types
        normalized_type: str = ""
        if policy_type is not None:
            try:
                normalized_type = self._normalize_policy_type(policy_type)
            except PolicyRegistryError:
                return 0

        # Normalize version for consistent lookup (e.g., "1.0" matches "1.0.0")
        normalized_version: str | None = None
        if version is not None:
            normalized_version = self._normalize_version(version)

        # Thread safety: Lock held during full unregister operation (write operation)
        with self._lock:
            # Performance optimization: Use secondary index
            candidate_keys = self._policy_id_index.get(policy_id, [])
            keys_to_remove: list[ModelPolicyKey] = []

            for key in candidate_keys:
                if normalized_type and key.policy_type != normalized_type:
                    continue
                if normalized_version is not None and key.version != normalized_version:
                    continue
                keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._registry[key]
                # Update secondary index
                self._policy_id_index[policy_id].remove(key)

            # Clean up empty index entries
            if (
                policy_id in self._policy_id_index
                and not self._policy_id_index[policy_id]
            ):
                del self._policy_id_index[policy_id]

            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all policy registrations.

        Removes all registered policies from the registry.

        Warning:
            This method is intended for **testing purposes only**.
            Calling it in production code will emit a warning.
            It breaks the immutability guarantee after startup.

        Example:
            >>> registry = RegistryPolicy()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> registry.clear()
            >>> registry.list_keys()
            []
        """
        warnings.warn(
            "RegistryPolicy.clear() is intended for testing only. "
            "Do not use in production code.",
            UserWarning,
            stacklevel=2,
        )
        with self._lock:
            self._registry.clear()
            self._policy_id_index.clear()

    def __len__(self) -> int:
        """Return the number of registered policies.

        Returns:
            Number of registered policy (id, type, version) combinations.

        Example:
            >>> registry = RegistryPolicy()
            >>> len(registry)
            0
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> len(registry)
            1
        """
        with self._lock:
            return len(self._registry)

    def __contains__(self, policy_id: str) -> bool:
        """Check if policy ID is registered using 'in' operator.

        Args:
            policy_id: Policy identifier.

        Returns:
            True if policy ID is registered (any type/version), False otherwise.

        Example:
            >>> registry = RegistryPolicy()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> "retry" in registry
            True
            >>> "unknown" in registry
            False
        """
        return self.is_registered(policy_id)


# =============================================================================
# Module Exports
# =============================================================================

__all__: list[str] = [
    "ModelPolicyKey",
    # Models
    "ModelPolicyRegistration",
    # Registry class
    "RegistryPolicy",
]
