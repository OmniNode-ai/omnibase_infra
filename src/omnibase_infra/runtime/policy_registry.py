# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Registry - SINGLE SOURCE OF TRUTH for policy plugin registration.

This module provides the PolicyRegistry class for registering and resolving
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
    from omnibase_infra.runtime.policy_registry import (
        PolicyRegistry,
        ModelPolicyRegistration,
        get_policy_registry,
    )
    from omnibase_infra.enums import EnumPolicyType

    registry = get_policy_registry()

    # Register a synchronous policy using the model (preferred)
    registration = ModelPolicyRegistration(
        policy_id="exponential_backoff",
        policy_class=ExponentialBackoffPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )
    registry.register(registration)

    # Register using convenience method (preserves original API)
    registry.register_policy(
        policy_id="async_merge",
        policy_class=AsyncMergePolicy,
        policy_type=EnumPolicyType.REDUCER,
        version="1.0.0",
        deterministic_async=True,  # MUST be explicit for async policies
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

import asyncio
import functools
import threading
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import PolicyRegistryError, ProtocolConfigurationError
from omnibase_infra.runtime.models import ModelPolicyKey, ModelPolicyRegistration

if TYPE_CHECKING:
    from omnibase_infra.runtime.protocol_policy import ProtocolPolicy


# =============================================================================
# Policy Registry
# =============================================================================


class PolicyRegistry:
    """SINGLE SOURCE OF TRUTH for policy plugin registration in omnibase_infra.

    Thread-safe registry for policy plugins. Manages pure decision logic plugins
    that can be used by orchestrator and reducer nodes.

    The registry maintains a mapping from ModelPolicyKey instances to policy classes
    that implement the ProtocolPolicy protocol. ModelPolicyKey provides strong typing
    and replaces the legacy tuple[str, str, str] pattern.

    This class implements the ProtocolRegistryBase[tuple[str, str, str], type[ProtocolPolicy]]
    interface from omnibase_spi 0.4.0. The external API maintains tuple compatibility
    while internal operations use ModelPolicyKey for strong typing.

    Thread Safety:
        All registration operations are protected by a threading.Lock to ensure
        thread-safe access in concurrent environments.

    Sync Enforcement:
        By default, policies must be synchronous. If a policy has async methods
        (evaluate, decide, reduce), registration will fail unless
        deterministic_async=True is explicitly specified.

    Attributes:
        _registry: Internal dictionary mapping ModelPolicyKey instances to policy classes
        _lock: Threading lock for thread-safe registration operations

    Example:
        >>> from omnibase_infra.runtime.models import ModelPolicyRegistration
        >>> registry = PolicyRegistry()
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

    # Methods to check for async validation
    _ASYNC_CHECK_METHODS: tuple[str, ...] = ("reduce", "decide", "evaluate")

    def __init__(self) -> None:
        """Initialize an empty policy registry with thread lock."""
        # Key: ModelPolicyKey -> policy_class (strong typing replaces tuple pattern)
        self._registry: dict[ModelPolicyKey, type[ProtocolPolicy]] = {}
        self._lock: threading.Lock = threading.Lock()

        # Performance optimization: Secondary indexes for O(1) lookups
        # Maps policy_id -> list of ModelPolicyKey instances
        self._policy_id_index: dict[str, list[ModelPolicyKey]] = {}

    def _validate_sync_enforcement(
        self,
        policy_id: str,
        policy_class: type[ProtocolPolicy],
        deterministic_async: bool,
    ) -> None:
        """Validate that policy methods are synchronous unless explicitly async.

        This validation enforces the synchronous-by-default policy execution model.
        Policy plugins are expected to be pure decision logic without I/O or async
        operations. If a policy needs async methods (e.g., for deterministic async
        computation), it must be explicitly flagged with deterministic_async=True
        during registration.

        Validation Process:
            1. Inspect policy class for methods: reduce(), decide(), evaluate()
            2. Check if any of these methods are async (coroutine functions)
            3. If async methods found and deterministic_async=False, raise error
            4. If async methods found and deterministic_async=True, allow registration

        This validation helps prevent accidental async policy registration and ensures
        that async policies are consciously marked as such for proper runtime handling.

        Args:
            policy_id: Unique identifier for the policy being validated
            policy_class: The policy class to validate for async methods
            deterministic_async: If True, allows async interface; if False, enforces sync

        Raises:
            PolicyRegistryError: If policy has async methods (reduce, decide, evaluate)
                                and deterministic_async=False. Error includes the policy_id
                                and the name of the async method that caused validation failure.

        Example:
            >>> # This will fail - async policy without explicit flag
            >>> class AsyncPolicy:
            ...     async def evaluate(self, context):
            ...         return True
            >>> registry._validate_sync_enforcement("async_pol", AsyncPolicy, False)
            PolicyRegistryError: Policy 'async_pol' has async evaluate() but
                                 deterministic_async=True not specified.

            >>> # This will succeed - async explicitly flagged
            >>> registry._validate_sync_enforcement("async_pol", AsyncPolicy, True)
        """
        for method_name in self._ASYNC_CHECK_METHODS:
            if hasattr(policy_class, method_name):
                method = getattr(policy_class, method_name)
                if asyncio.iscoroutinefunction(method):
                    if not deterministic_async:
                        raise PolicyRegistryError(
                            f"Policy '{policy_id}' has async {method_name}() but "
                            f"deterministic_async=True not specified. "
                            f"Policy plugins must be synchronous by default.",
                            policy_id=policy_id,
                            policy_type=None,
                            async_method=method_name,
                        )

    def _normalize_policy_type(
        self,
        policy_type: str | EnumPolicyType,
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
            >>> registry = PolicyRegistry()
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

    def register(
        self,
        registration: ModelPolicyRegistration,
    ) -> None:
        """Register a policy plugin using a registration model.

        Associates a (policy_id, policy_type, version) tuple with a policy class.
        If the combination is already registered, the existing registration is
        overwritten.

        Args:
            registration: ModelPolicyRegistration containing all registration parameters:
                - policy_id: Unique identifier for the policy
                - policy_class: The policy class to register (must implement ProtocolPolicy)
                - policy_type: Whether this is orchestrator or reducer policy
                - version: Semantic version string (default: "1.0.0")
                - deterministic_async: If True, allows async interface

        Raises:
            PolicyRegistryError: If policy has async methods and
                               deterministic_async=False, or if policy_type is invalid

        Example:
            >>> from omnibase_infra.runtime.models import ModelPolicyRegistration
            >>> registry = PolicyRegistry()
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
        deterministic_async = registration.deterministic_async

        # Validate sync enforcement
        self._validate_sync_enforcement(policy_id, policy_class, deterministic_async)

        # Normalize policy type
        normalized_type = self._normalize_policy_type(policy_type)

        # Validate version format (ensures semantic versioning compliance)
        # This calls _parse_semver which will raise ProtocolConfigurationError if invalid
        try:
            self._parse_semver(version)
        except ProtocolConfigurationError:
            # Re-raise as-is to maintain error context
            raise

        # Register the policy using ModelPolicyKey
        key = ModelPolicyKey(
            policy_id=policy_id,
            policy_type=normalized_type,
            version=version,
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
        policy_type: str | EnumPolicyType,
        version: str = "1.0.0",
        deterministic_async: bool = False,
    ) -> None:
        """Convenience method to register a policy with individual parameters.

        Wraps parameters in ModelPolicyRegistration and calls register().
        This method preserves the original API for backwards compatibility.

        Args:
            policy_id: Unique identifier for the policy (e.g., 'exponential_backoff')
            policy_class: The policy class to register. Must implement ProtocolPolicy.
            policy_type: Whether this is orchestrator or reducer policy.
                        Can be EnumPolicyType or string literal.
            version: Semantic version string (default: "1.0.0")
            deterministic_async: If True, allows async interface. MUST be explicitly
                                flagged for policies with async methods.

        Raises:
            PolicyRegistryError: If policy has async methods and
                               deterministic_async=False, or if policy_type is invalid

        Example:
            >>> registry = PolicyRegistry()
            >>> registry.register_policy(
            ...     policy_id="retry_backoff",
            ...     policy_class=RetryBackoffPolicy,
            ...     policy_type=EnumPolicyType.ORCHESTRATOR,
            ...     version="1.0.0",
            ... )
        """
        registration = ModelPolicyRegistration(
            policy_id=policy_id,
            policy_class=policy_class,
            policy_type=policy_type,
            version=version,
            deterministic_async=deterministic_async,
        )
        self.register(registration)

    def get(
        self,
        policy_id: str,
        policy_type: str | EnumPolicyType | None = None,
        version: str | None = None,
    ) -> type[ProtocolPolicy]:
        """Get policy class by ID, type, and optional version.

        Resolves the policy class registered for the given policy configuration.
        If policy_type is not specified, returns the first matching policy_id.
        If version is not specified, returns the latest version (lexicographically).

        Performance Characteristics:
            - Best case: O(1) - Direct lookup with policy_id only (single version)
            - Average case: O(k) where k = number of versions for policy_id
            - Worst case: O(k*log(k)) when multiple versions require sorting
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
            >>> registry = PolicyRegistry()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> policy_cls = registry.get("retry")
            >>> policy_cls = registry.get("retry", policy_type="orchestrator")
            >>> policy_cls = registry.get("retry", version="1.0.0")
        """
        # Normalize policy_type if provided (outside lock for minimal critical section)
        normalized_type: str | None = None
        if policy_type is not None:
            normalized_type = self._normalize_policy_type(policy_type)

        with self._lock:
            # Performance optimization: Use secondary index for O(1) lookup by policy_id
            # This avoids iterating through all registry entries (O(n) → O(1))
            candidate_keys = self._policy_id_index.get(policy_id, [])

            # Early exit if policy_id not found - avoid building matches list
            if not candidate_keys:
                # Defer expensive _list_internal() call until actually raising error
                filters = [f"policy_id={policy_id!r}"]
                if policy_type is not None:
                    filters.append(f"policy_type={policy_type!r}")
                if version is not None:
                    filters.append(f"version={version!r}")

                registered = self._list_internal()
                raise PolicyRegistryError(
                    f"No policy registered matching: {', '.join(filters)}. "
                    f"Registered policies: {registered}",
                    policy_id=policy_id,
                    policy_type=str(policy_type) if policy_type else None,
                )

            # Find matching entries from candidates (optimized to reduce allocations)
            # Fast path: no filtering needed (common case - just get latest version)
            if normalized_type is None and version is None:
                matches: list[tuple[ModelPolicyKey, type[ProtocolPolicy]]] = [
                    (key, self._registry[key]) for key in candidate_keys
                ]
            else:
                # Filtered path: apply type and version filters
                matches = []
                for key in candidate_keys:
                    if (
                        normalized_type is not None
                        and key.policy_type != normalized_type
                    ):
                        continue
                    if version is not None and key.version != version:
                        continue
                    matches.append((key, self._registry[key]))

            if not matches:
                # Filters eliminated all candidates - build error message
                filters = [f"policy_id={policy_id!r}"]
                if policy_type is not None:
                    filters.append(f"policy_type={policy_type!r}")
                if version is not None:
                    filters.append(f"version={version!r}")

                # Defer expensive _list_internal() call until actually raising error
                registered = self._list_internal()
                raise PolicyRegistryError(
                    f"No policy registered matching: {', '.join(filters)}. "
                    f"Registered policies: {registered}",
                    policy_id=policy_id,
                    policy_type=str(policy_type) if policy_type else None,
                )

            # If version not specified and multiple matches, return latest
            # (using cached semantic version comparison)
            if version is None and len(matches) > 1:
                # Sort in-place to avoid allocating a new list
                matches.sort(
                    key=lambda x: self._parse_semver(x[0].version), reverse=True
                )

            return matches[0][1]

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _parse_semver(version: str) -> tuple[int, int, int, str]:
        """Parse semantic version string into comparable tuple.

        Handles versions like "1.0.0", "2.1.3", "1.0.0-alpha".
        Pre-release versions sort before release versions.

        This method is cached using LRU cache (maxsize=128) to avoid re-parsing
        the same version strings repeatedly, improving performance for lookups
        that compare multiple versions.

        Cache Size Rationale:
            128 entries balances memory vs performance for typical workloads:
            - Typical registry: 10-50 unique policy versions
            - Peak scenarios: 50-100 versions across multiple policy types
            - Each cache entry: ~100 bytes (string key + tuple value)
            - Total memory: ~12.8KB worst case (negligible overhead)
            - Hit rate: >95% for repeated get() calls with version comparisons
            - Eviction: Rare in practice, LRU ensures least-used versions purged

        Args:
            version: Semantic version string (e.g., "1.2.3" or "1.0.0-beta")

        Returns:
            Tuple of (major, minor, patch, prerelease) for comparison.
            Prerelease is empty string for release versions (sorts after prereleases).

        Raises:
            ProtocolConfigurationError: If version format is invalid
        """
        # Validate non-empty version string
        if not version or not version.strip():
            raise ProtocolConfigurationError(
                "Invalid semantic version format: empty version string",
                version=version,
            )

        # Split off any prerelease suffix (e.g., "1.0.0-alpha" -> "1.0.0", "alpha")
        if "-" in version:
            version_part, prerelease = version.split("-", 1)
        else:
            version_part, prerelease = version, ""

        # Parse major.minor.patch
        parts = version_part.split(".")

        # Validate version format (must have 1-3 parts, no empty parts)
        if len(parts) < 1 or len(parts) > 3 or any(not p.strip() for p in parts):
            raise ProtocolConfigurationError(
                f"Invalid semantic version format: '{version}'. "
                f"Expected format: 'major.minor.patch' or 'major.minor.patch-prerelease'",
                version=version,
            )

        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
        except (ValueError, IndexError) as e:
            raise ProtocolConfigurationError(
                f"Invalid semantic version format: '{version}'. "
                f"Version components must be integers (e.g., '1.2.3')",
                version=version,
            ) from e

        # Validate non-negative integers
        if major < 0 or minor < 0 or patch < 0:
            raise ProtocolConfigurationError(
                f"Invalid semantic version format: '{version}'. "
                f"Version components must be non-negative integers",
                version=version,
            )

        # Empty prerelease sorts after non-empty (release > prerelease)
        # Use chr(127) for empty to sort after any prerelease string
        sort_prerelease = prerelease if prerelease else chr(127)

        return (major, minor, patch, sort_prerelease)

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
        policy_type: str | EnumPolicyType | None = None,
    ) -> list[tuple[str, str, str]]:
        """List registered policy keys as (id, type, version) tuples.

        This method implements the ProtocolRegistryBase.list_keys() interface.

        Args:
            policy_type: Optional filter to list only policies of a specific type.

        Returns:
            List of (policy_id, policy_type, version) tuples, sorted alphabetically.

        Example:
            >>> registry = PolicyRegistry()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> registry.register("merge", MergePolicy, EnumPolicyType.REDUCER)
            >>> print(registry.list_keys())
            [('merge', 'reducer', '1.0.0'), ('retry', 'orchestrator', '1.0.0')]
            >>> print(registry.list_keys(policy_type="orchestrator"))
            [('retry', 'orchestrator', '1.0.0')]
        """
        # Normalize policy_type if provided
        normalized_type: str | None = None
        if policy_type is not None:
            normalized_type = self._normalize_policy_type(policy_type)

        with self._lock:
            results: list[tuple[str, str, str]] = []
            for key in sorted(
                self._registry.keys(),
                key=lambda k: (k.policy_id, k.policy_type, k.version),
            ):
                if normalized_type is not None and key.policy_type != normalized_type:
                    continue
                results.append(key.to_tuple())
            return results

    def list_policy_types(self) -> list[str]:
        """List registered policy types.

        Returns:
            List of unique policy type strings that have registered policies.

        Example:
            >>> registry = PolicyRegistry()
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
            >>> registry = PolicyRegistry()
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
        policy_type: str | EnumPolicyType | None = None,
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
            >>> registry = PolicyRegistry()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> registry.is_registered("retry")
            True
            >>> registry.is_registered("unknown")
            False
        """
        # Normalize policy_type if provided
        normalized_type: str | None = None
        if policy_type is not None:
            try:
                normalized_type = self._normalize_policy_type(policy_type)
            except PolicyRegistryError:
                return False

        with self._lock:
            # Performance optimization: Use secondary index
            candidate_keys = self._policy_id_index.get(policy_id, [])
            for key in candidate_keys:
                if normalized_type is not None and key.policy_type != normalized_type:
                    continue
                if version is not None and key.version != version:
                    continue
                return True
            return False

    def unregister(
        self,
        policy_id: str,
        policy_type: str | EnumPolicyType | None = None,
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
            >>> registry = PolicyRegistry()
            >>> registry.register("retry", RetryPolicyV1, "orchestrator", "1.0.0")
            >>> registry.register("retry", RetryPolicyV2, "orchestrator", "2.0.0")
            >>> registry.unregister("retry")  # Removes all versions
            2
            >>> registry.unregister("retry", version="1.0.0")  # Remove specific version
            1
        """
        # Normalize policy_type if provided
        normalized_type: str | None = None
        if policy_type is not None:
            try:
                normalized_type = self._normalize_policy_type(policy_type)
            except PolicyRegistryError:
                return 0

        with self._lock:
            # Performance optimization: Use secondary index
            candidate_keys = self._policy_id_index.get(policy_id, [])
            keys_to_remove: list[ModelPolicyKey] = []

            for key in candidate_keys:
                if normalized_type is not None and key.policy_type != normalized_type:
                    continue
                if version is not None and key.version != version:
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
        This is useful for testing scenarios.

        Example:
            >>> registry = PolicyRegistry()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> registry.clear()
            >>> registry.list_keys()
            []
        """
        with self._lock:
            self._registry.clear()
            self._policy_id_index.clear()

    def __len__(self) -> int:
        """Return the number of registered policies.

        Returns:
            Number of registered policy (id, type, version) combinations.

        Example:
            >>> registry = PolicyRegistry()
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
            >>> registry = PolicyRegistry()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> "retry" in registry
            True
            >>> "unknown" in registry
            False
        """
        return self.is_registered(policy_id)


# =============================================================================
# Module-Level Singleton Registry
# =============================================================================

# Module-level singleton instance (lazy initialized)
_policy_registry: PolicyRegistry | None = None
_singleton_lock: threading.Lock = threading.Lock()


def get_policy_registry() -> PolicyRegistry:
    """Get the singleton policy registry instance.

    Returns a module-level singleton instance of PolicyRegistry.
    Creates the instance on first call (lazy initialization).

    Returns:
        PolicyRegistry: The singleton policy registry instance.

    Example:
        >>> registry = get_policy_registry()
        >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
        >>> same_registry = get_policy_registry()
        >>> same_registry is registry
        True
    """
    global _policy_registry  # noqa: PLW0603
    if _policy_registry is None:
        with _singleton_lock:
            # Double-check locking pattern
            if _policy_registry is None:
                _policy_registry = PolicyRegistry()
    return _policy_registry


# =============================================================================
# Convenience Functions
# =============================================================================


def get_policy_class(
    policy_id: str,
    policy_type: str | EnumPolicyType | None = None,
    version: str | None = None,
) -> type[ProtocolPolicy]:
    """Get policy class from the singleton registry.

    Convenience function that wraps get_policy_registry().get().

    Args:
        policy_id: Policy identifier.
        policy_type: Optional policy type filter.
        version: Optional version filter.

    Returns:
        Policy class registered for the configuration.

    Raises:
        PolicyRegistryError: If no matching policy is found.

    Example:
        >>> from omnibase_infra.runtime.policy_registry import get_policy_class
        >>> policy_cls = get_policy_class("exponential_backoff")
        >>> policy = policy_cls()
    """
    return get_policy_registry().get(policy_id, policy_type, version)


def register_policy(
    policy_id: str,
    policy_class: type[ProtocolPolicy],
    policy_type: str | EnumPolicyType,
    version: str = "1.0.0",
    deterministic_async: bool = False,
) -> None:
    """Register a policy in the singleton registry.

    Convenience function that wraps get_policy_registry().register_policy().

    Args:
        policy_id: Unique identifier for the policy.
        policy_class: The policy class to register.
        policy_type: Whether this is orchestrator or reducer policy.
        version: Semantic version string (default: "1.0.0").
        deterministic_async: If True, allows async interface.

    Raises:
        PolicyRegistryError: If policy has async methods and
                           deterministic_async=False.

    Example:
        >>> from omnibase_infra.runtime.policy_registry import register_policy
        >>> register_policy(
        ...     policy_id="retry_backoff",
        ...     policy_class=RetryBackoffPolicy,
        ...     policy_type="orchestrator",
        ... )
    """
    get_policy_registry().register_policy(
        policy_id=policy_id,
        policy_class=policy_class,
        policy_type=policy_type,
        version=version,
        deterministic_async=deterministic_async,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__: list[str] = [
    # Registry class
    "PolicyRegistry",
    # Models
    "ModelPolicyRegistration",
    "ModelPolicyKey",
    # Singleton accessor
    "get_policy_registry",
    # Convenience functions
    "get_policy_class",
    "register_policy",
]
