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
    policies = registry.list()  # [(id, type, version), ...]
    ```

Integration Points:
- RuntimeHostProcess uses this registry to discover and instantiate policies
- Policies are loaded based on contract definitions
- Supports hot-reload patterns for development
"""

from __future__ import annotations

import asyncio
import builtins
import threading
from typing import TYPE_CHECKING, Optional, Union

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import PolicyRegistryError
from omnibase_infra.runtime.models import ModelPolicyRegistration

if TYPE_CHECKING:
    from omnibase_infra.runtime.protocol_policy import ProtocolPolicy


# =============================================================================
# Policy Registry
# =============================================================================


class PolicyRegistry:
    """SINGLE SOURCE OF TRUTH for policy plugin registration in omnibase_infra.

    Thread-safe registry for policy plugins. Manages pure decision logic plugins
    that can be used by orchestrator and reducer nodes.

    The registry maintains a mapping from (policy_id, policy_type, version) tuples
    to policy classes that implement the ProtocolPolicy protocol.

    Thread Safety:
        All registration operations are protected by a threading.Lock to ensure
        thread-safe access in concurrent environments.

    Sync Enforcement:
        By default, policies must be synchronous. If a policy has async methods
        (evaluate, decide, reduce), registration will fail unless
        deterministic_async=True is explicitly specified.

    Attributes:
        _registry: Internal dictionary mapping (policy_id, policy_type, version)
                   tuples to policy classes
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
        >>> print(registry.list())
        [('retry_backoff', 'orchestrator', '1.0.0')]
    """

    # Methods to check for async validation
    _ASYNC_CHECK_METHODS: tuple[str, ...] = ("reduce", "decide", "evaluate")

    def __init__(self) -> None:
        """Initialize an empty policy registry with thread lock."""
        # Key: (policy_id, policy_type, version) -> policy_class
        self._registry: dict[tuple[str, str, str], type[ProtocolPolicy]] = {}
        self._lock: threading.Lock = threading.Lock()

    def _validate_sync_enforcement(
        self,
        policy_id: str,
        policy_class: type[ProtocolPolicy],
        deterministic_async: bool,
    ) -> None:
        """Validate that policy methods are synchronous unless explicitly async.

        Args:
            policy_id: Unique identifier for the policy
            policy_class: The policy class to validate
            deterministic_async: If True, allows async interface

        Raises:
            PolicyRegistryError: If policy has async methods and
                                deterministic_async=False
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
        """Normalize policy type to string value.

        Args:
            policy_type: Policy type as string or EnumPolicyType

        Returns:
            Normalized string value for the policy type

        Raises:
            PolicyRegistryError: If policy_type is invalid
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

        # Register the policy
        key = (policy_id, normalized_type, version)
        with self._lock:
            self._registry[key] = policy_class

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
        policy_type: Optional[Union[str, EnumPolicyType]] = None,
        version: Optional[str] = None,
    ) -> type[ProtocolPolicy]:
        """Get policy class by ID, type, and optional version.

        Resolves the policy class registered for the given policy configuration.
        If policy_type is not specified, returns the first matching policy_id.
        If version is not specified, returns the latest version (lexicographically).

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
        # Normalize policy_type if provided
        normalized_type: Optional[str] = None
        if policy_type is not None:
            normalized_type = self._normalize_policy_type(policy_type)

        with self._lock:
            # Find matching entries
            matches: list[tuple[tuple[str, str, str], type[ProtocolPolicy]]] = []
            for key, policy_cls in self._registry.items():
                key_id, key_type, key_version = key
                if key_id != policy_id:
                    continue
                if normalized_type is not None and key_type != normalized_type:
                    continue
                if version is not None and key_version != version:
                    continue
                matches.append((key, policy_cls))

            if not matches:
                # Build descriptive error message
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

            # If version not specified, return latest (using semantic version comparison)
            if version is None and len(matches) > 1:
                matches.sort(key=lambda x: self._parse_semver(x[0][2]), reverse=True)

            return matches[0][1]

    @staticmethod
    def _parse_semver(version: str) -> tuple[int, int, int, str]:
        """Parse semantic version string into comparable tuple.

        Handles versions like "1.0.0", "2.1.3", "1.0.0-alpha".
        Pre-release versions sort before release versions.

        Args:
            version: Semantic version string (e.g., "1.2.3" or "1.0.0-beta")

        Returns:
            Tuple of (major, minor, patch, prerelease) for comparison.
            Prerelease is empty string for release versions (sorts after prereleases).
        """
        # Split off any prerelease suffix (e.g., "1.0.0-alpha" -> "1.0.0", "alpha")
        if "-" in version:
            version_part, prerelease = version.split("-", 1)
        else:
            version_part, prerelease = version, ""

        # Parse major.minor.patch
        parts = version_part.split(".")
        try:
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
        except ValueError:
            # Fall back to (0, 0, 0) for unparseable versions
            major, minor, patch = 0, 0, 0

        # Empty prerelease sorts after non-empty (release > prerelease)
        # Use chr(127) for empty to sort after any prerelease string
        sort_prerelease = prerelease if prerelease else chr(127)

        return (major, minor, patch, sort_prerelease)

    def _list_internal(self) -> list[tuple[str, str, str]]:
        """Internal list method (assumes lock is held).

        Returns:
            List of (policy_id, policy_type, version) tuples.
        """
        return [(k[0], k[1], k[2]) for k in sorted(self._registry.keys())]

    def list(
        self,
        policy_type: Optional[Union[str, EnumPolicyType]] = None,
    ) -> list[tuple[str, str, str]]:
        """List registered policies as (id, type, version) tuples.

        Args:
            policy_type: Optional filter to list only policies of a specific type.

        Returns:
            List of (policy_id, policy_type, version) tuples, sorted alphabetically.

        Example:
            >>> registry = PolicyRegistry()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> registry.register("merge", MergePolicy, EnumPolicyType.REDUCER)
            >>> print(registry.list())
            [('merge', 'reducer', '1.0.0'), ('retry', 'orchestrator', '1.0.0')]
            >>> print(registry.list(policy_type="orchestrator"))
            [('retry', 'orchestrator', '1.0.0')]
        """
        # Normalize policy_type if provided
        normalized_type: Optional[str] = None
        if policy_type is not None:
            normalized_type = self._normalize_policy_type(policy_type)

        with self._lock:
            results: list[tuple[str, str, str]] = []
            for key in sorted(self._registry.keys()):
                policy_id, key_type, version = key
                if normalized_type is not None and key_type != normalized_type:
                    continue
                results.append((policy_id, key_type, version))
            return results

    def list_policy_types(self) -> builtins.list[str]:
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
            types = {key[1] for key in self._registry}
            return sorted(types)

    def list_versions(self, policy_id: str) -> builtins.list[str]:
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
            versions = {key[2] for key in self._registry if key[0] == policy_id}
            return sorted(versions)

    def is_registered(
        self,
        policy_id: str,
        policy_type: Optional[Union[str, EnumPolicyType]] = None,
        version: Optional[str] = None,
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
        normalized_type: Optional[str] = None
        if policy_type is not None:
            try:
                normalized_type = self._normalize_policy_type(policy_type)
            except PolicyRegistryError:
                return False

        with self._lock:
            for key in self._registry:
                key_id, key_type, key_version = key
                if key_id != policy_id:
                    continue
                if normalized_type is not None and key_type != normalized_type:
                    continue
                if version is not None and key_version != version:
                    continue
                return True
            return False

    def unregister(
        self,
        policy_id: str,
        policy_type: Optional[Union[str, EnumPolicyType]] = None,
        version: Optional[str] = None,
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
        normalized_type: Optional[str] = None
        if policy_type is not None:
            try:
                normalized_type = self._normalize_policy_type(policy_type)
            except PolicyRegistryError:
                return 0

        with self._lock:
            keys_to_remove: list[tuple[str, str, str]] = []
            for key in self._registry:
                key_id, key_type, key_version = key
                if key_id != policy_id:
                    continue
                if normalized_type is not None and key_type != normalized_type:
                    continue
                if version is not None and key_version != version:
                    continue
                keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._registry[key]

            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all policy registrations.

        Removes all registered policies from the registry.
        This is useful for testing scenarios.

        Example:
            >>> registry = PolicyRegistry()
            >>> registry.register("retry", RetryPolicy, EnumPolicyType.ORCHESTRATOR)
            >>> registry.clear()
            >>> registry.list()
            []
        """
        with self._lock:
            self._registry.clear()

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
_policy_registry: Optional[PolicyRegistry] = None
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
    policy_type: Optional[Union[str, EnumPolicyType]] = None,
    version: Optional[str] = None,
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
    # Model
    "ModelPolicyRegistration",
    # Singleton accessor
    "get_policy_registry",
    # Convenience functions
    "get_policy_class",
    "register_policy",
]
