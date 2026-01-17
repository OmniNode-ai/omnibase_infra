# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Validation Mixin.

This module provides validation functionality for policy registration,
extracted from RegistryPolicy to reduce class method count.

The mixin provides:
- Protocol implementation validation
- Sync enforcement validation
- Policy type normalization
- Version string normalization

This mixin is designed to be used with RegistryPolicy and follows the
ONEX naming convention: mixin_<name>.py -> Mixin<Name>.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, overload

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import PolicyRegistryError, ProtocolConfigurationError
from omnibase_infra.runtime.util_version import normalize_version

if TYPE_CHECKING:
    from omnibase_infra.runtime.protocol_policy import ProtocolPolicy


class MixinPolicyValidation:
    """Mixin providing policy validation functionality.

    This mixin extracts validation methods from RegistryPolicy to reduce
    the class method count and provide better separation of concerns.

    Methods:
        _validate_policy_id_param: Validate policy_id parameter (shared helper)
        _validate_policy_type_param: Validate policy_type parameter (shared helper)
        _validate_protocol_implementation: Validate policy implements ProtocolPolicy
        _validate_sync_enforcement: Validate sync/async policy methods
        _normalize_policy_type: Normalize and validate policy type
        _normalize_version: Normalize version string format
    """

    # Methods to check for async validation (class attribute shared with RegistryPolicy)
    _ASYNC_CHECK_METHODS: tuple[str, ...] = ("reduce", "decide", "evaluate")

    @staticmethod
    def _validate_policy_id_param(
        policy_id: str | None,
        policy_class: type | None = None,
        policy_type: str | None = None,
    ) -> None:
        """Validate policy_id parameter individually.

        This shared helper validates that policy_id is a non-empty string.
        It is used by both _validate_protocol_implementation and
        _validate_sync_enforcement to ensure consistent validation.

        Args:
            policy_id: The policy_id parameter to validate.
            policy_class: Optional policy class for error context.
            policy_type: Optional policy type for error context.

        Raises:
            PolicyRegistryError: If policy_id is None, not a string, or empty.
        """
        if policy_id is None:
            raise PolicyRegistryError(
                "policy_id is required and cannot be None",
                policy_id=None,
                policy_class=policy_class.__name__ if policy_class else None,
                policy_type=policy_type,
            )
        if not isinstance(policy_id, str):
            raise PolicyRegistryError(
                f"policy_id must be a string, got {type(policy_id).__name__}",
                policy_id=str(policy_id),
                policy_class=policy_class.__name__ if policy_class else None,
                policy_type=policy_type,
            )
        if not policy_id.strip():
            raise PolicyRegistryError(
                "policy_id is required and cannot be empty",
                policy_id=policy_id,
                policy_class=policy_class.__name__ if policy_class else None,
                policy_type=policy_type,
            )

    @overload
    @staticmethod
    def _validate_policy_type_param(
        policy_type: str | EnumPolicyType | None,
        policy_id: str | None = None,
        allow_none: Literal[True] = True,
    ) -> str | None: ...

    @overload
    @staticmethod
    def _validate_policy_type_param(
        policy_type: str | EnumPolicyType | None,
        policy_id: str | None = None,
        *,
        allow_none: Literal[False],
    ) -> str: ...

    @staticmethod
    def _validate_policy_type_param(
        policy_type: str | EnumPolicyType | None,
        policy_id: str | None = None,
        allow_none: bool = True,
    ) -> str | None:
        """Validate policy_type parameter individually.

        This shared helper validates that policy_type is a valid string or
        EnumPolicyType. It is used by _validate_sync_enforcement and
        _normalize_policy_type to ensure consistent validation.

        Args:
            policy_type: The policy_type parameter to validate.
            policy_id: Optional policy_id for error context.
            allow_none: If True, None is a valid value. If False, None raises error.

        Returns:
            Normalized string value if policy_type is provided, None otherwise.

        Raises:
            PolicyRegistryError: If policy_type is invalid (wrong type, empty string,
                or None when allow_none=False).
        """
        if policy_type is None:
            if not allow_none:
                raise PolicyRegistryError(
                    "policy_type is required and cannot be None",
                    policy_id=policy_id,
                    policy_type=None,
                )
            return None

        if isinstance(policy_type, EnumPolicyType):
            return policy_type.value

        if not isinstance(policy_type, str):
            raise PolicyRegistryError(
                f"policy_type must be a string or EnumPolicyType, "
                f"got {type(policy_type).__name__}",
                policy_id=policy_id,
                policy_type=str(policy_type),
            )
        if not policy_type.strip():
            raise PolicyRegistryError(
                "policy_type cannot be empty when provided",
                policy_id=policy_id,
                policy_type=policy_type,
            )

        return policy_type

    def _validate_protocol_implementation(
        self,
        policy_id: str,
        policy_class: type[ProtocolPolicy],
        policy_type: str | EnumPolicyType | None = None,
    ) -> None:
        """Validate that policy class implements ProtocolPolicy protocol.

        Performs runtime type validation using duck typing to ensure the policy
        class has the required methods and properties before registration.

        Validation Order:
            Validations are performed in fail-fast order with cheap checks first:

            1. **Parameter validation**: Validate policy_id and policy_type
               parameters individually using shared helpers.

            2. **Comprehensive check**: If class is missing all required attributes
               (policy_id, policy_type, evaluate), raise a comprehensive error
               listing all missing requirements.

            3. **evaluate() existence** (O(1) hasattr check):
               Verifies policy_class has evaluate() method.

            4. **evaluate() callability** (O(1) callable check):
               Verifies evaluate is actually callable.

        Args:
            policy_id: Unique identifier for the policy being validated
            policy_class: The policy class to validate
            policy_type: Optional policy type for error context and validation

        Raises:
            PolicyRegistryError: If policy_class does not implement ProtocolPolicy:
                - policy_id is None or empty
                - policy_type is invalid (when provided)
                - Missing all required attributes (comprehensive error)
                - Missing evaluate() method
                - evaluate is not callable
        """
        # Validate policy_type first to get normalized value for error context
        normalized_policy_type = self._validate_policy_type_param(
            policy_type, policy_id=policy_id, allow_none=True
        )

        # Validate policy_id parameter individually using shared helper
        self._validate_policy_id_param(
            policy_id, policy_class=policy_class, policy_type=normalized_policy_type
        )

        # Check for required attributes
        has_policy_id = hasattr(policy_class, "policy_id")
        has_policy_type = hasattr(policy_class, "policy_type")
        has_evaluate = hasattr(policy_class, "evaluate")

        # If all required attributes are missing, provide comprehensive error
        if not has_policy_id and not has_policy_type and not has_evaluate:
            raise PolicyRegistryError(
                f"Policy class {policy_class.__name__!r} does not implement "
                f"ProtocolPolicy protocol. Missing: policy_id property, "
                f"policy_type property, evaluate() method",
                policy_id=policy_id,
                policy_class=policy_class.__name__,
                policy_type=normalized_policy_type,
            )

        # Validate policy_id attribute exists on the class (individual check)
        if not has_policy_id:
            raise PolicyRegistryError(
                f"Policy class {policy_class.__name__!r} is missing policy_id "
                f"property from ProtocolPolicy protocol",
                policy_id=policy_id,
                policy_class=policy_class.__name__,
                policy_type=normalized_policy_type,
            )

        # Validate policy_type attribute exists on the class (individual check)
        if not has_policy_type:
            raise PolicyRegistryError(
                f"Policy class {policy_class.__name__!r} is missing policy_type "
                f"property from ProtocolPolicy protocol",
                policy_id=policy_id,
                policy_class=policy_class.__name__,
                policy_type=normalized_policy_type,
            )

        # Validate policy_id attribute value on the class (not just existence)
        # Note: If policy_id is a property, we can't validate the value at class level
        # (it's only available on instances). Properties are valid as they follow the
        # protocol pattern. We only validate if it's a direct class attribute.
        class_policy_id_attr = getattr(policy_class, "policy_id", None)
        is_policy_id_property = isinstance(
            getattr(type(policy_class), "policy_id", None), property
        ) or isinstance(class_policy_id_attr, property)

        if not is_policy_id_property:
            # Direct class attribute - validate the value
            if class_policy_id_attr is None:
                raise PolicyRegistryError(
                    f"Policy class {policy_class.__name__!r} has policy_id attribute "
                    f"but its value is None - must be a non-empty string",
                    policy_id=policy_id,
                    policy_class=policy_class.__name__,
                    policy_type=normalized_policy_type,
                )
            if not isinstance(class_policy_id_attr, str):
                raise PolicyRegistryError(
                    f"Policy class {policy_class.__name__!r} has policy_id attribute "
                    f"but its value must be a string, got {type(class_policy_id_attr).__name__}",
                    policy_id=policy_id,
                    policy_class=policy_class.__name__,
                    policy_type=normalized_policy_type,
                )
            if not class_policy_id_attr.strip():
                raise PolicyRegistryError(
                    f"Policy class {policy_class.__name__!r} has policy_id attribute "
                    f"but its value is empty - must be a non-empty string",
                    policy_id=policy_id,
                    policy_class=policy_class.__name__,
                    policy_type=normalized_policy_type,
                )

        # Validate policy_type attribute value on the class (not just existence)
        # Note: If policy_type is a property, we can't validate the value at class level
        # (it's only available on instances). Properties are valid as they follow the
        # protocol pattern. We only validate if it's a direct class attribute.
        class_policy_type_attr = getattr(policy_class, "policy_type", None)
        is_policy_type_property = isinstance(
            getattr(type(policy_class), "policy_type", None), property
        ) or isinstance(class_policy_type_attr, property)

        if not is_policy_type_property:
            # Direct class attribute - validate the value
            if class_policy_type_attr is None:
                raise PolicyRegistryError(
                    f"Policy class {policy_class.__name__!r} has policy_type attribute "
                    f"but its value is None - must be a valid EnumPolicyType",
                    policy_id=policy_id,
                    policy_class=policy_class.__name__,
                    policy_type=normalized_policy_type,
                )
            # Accept both EnumPolicyType enum and string values
            if isinstance(class_policy_type_attr, EnumPolicyType):
                pass  # Valid enum type
            elif isinstance(class_policy_type_attr, str):
                # Validate string against enum values
                valid_types = {e.value for e in EnumPolicyType}
                if class_policy_type_attr not in valid_types:
                    raise PolicyRegistryError(
                        f"Policy class {policy_class.__name__!r} has invalid policy_type "
                        f"value: {class_policy_type_attr!r}. Must be one of: {sorted(valid_types)}",
                        policy_id=policy_id,
                        policy_class=policy_class.__name__,
                        policy_type=normalized_policy_type,
                    )
            else:
                raise PolicyRegistryError(
                    f"Policy class {policy_class.__name__!r} has policy_type attribute "
                    f"but its value must be EnumPolicyType or string, "
                    f"got {type(class_policy_type_attr).__name__}",
                    policy_id=policy_id,
                    policy_class=policy_class.__name__,
                    policy_type=normalized_policy_type,
                )

        # Check evaluate() method exists
        if not has_evaluate:
            raise PolicyRegistryError(
                f"Policy class {policy_class.__name__!r} is missing evaluate() "
                f"method from ProtocolPolicy protocol",
                policy_id=policy_id,
                policy_class=policy_class.__name__,
                policy_type=normalized_policy_type,
            )

        # Check evaluate is callable
        evaluate_attr = getattr(policy_class, "evaluate", None)
        if not callable(evaluate_attr):
            raise PolicyRegistryError(
                f"Policy class {policy_class.__name__!r} has evaluate() method "
                f"(not callable) - must be a callable method",
                policy_id=policy_id,
                policy_class=policy_class.__name__,
                policy_type=normalized_policy_type,
            )

    def _validate_sync_enforcement(
        self,
        policy_id: str,
        policy_class: type[ProtocolPolicy],
        allow_async: bool,
        policy_type: str | EnumPolicyType | None = None,
    ) -> None:
        """Validate that policy methods are synchronous unless explicitly async.

        This validation enforces the synchronous-by-default policy execution model.
        Policy plugins are expected to be pure decision logic without I/O or async
        operations. If a policy needs async methods (e.g., for deterministic async
        computation), it must be explicitly flagged with allow_async=True
        during registration.

        Validation Process:
            1. Validate policy_id and policy_type parameters individually
            2. Inspect policy class for methods: reduce(), decide(), evaluate()
            3. Check if any of these methods are async (coroutine functions)
            4. If async methods found and allow_async=False, raise error
            5. If async methods found and allow_async=True, allow registration

        This validation helps prevent accidental async policy registration and ensures
        that async policies are consciously marked as such for proper runtime handling.

        Args:
            policy_id: Unique identifier for the policy being validated
            policy_class: The policy class to validate for async methods
            allow_async: If True, allows async interface; if False, enforces sync
            policy_type: Optional policy type for error context (orchestrator or reducer)

        Raises:
            PolicyRegistryError: If policy has async methods (reduce, decide, evaluate)
                                and allow_async=False. Error includes the policy_id,
                                policy_type, and the name of the async method that
                                caused validation failure.

        Example:
            >>> # This will fail - async policy without explicit flag
            >>> class AsyncPolicy:
            ...     async def evaluate(self, context):
            ...         return True
            >>> mixin._validate_sync_enforcement("async_pol", AsyncPolicy, False)
            PolicyRegistryError: Policy 'async_pol' has async evaluate() but
                                 allow_async=True not specified.

            >>> # This will succeed - async explicitly flagged
            >>> mixin._validate_sync_enforcement("async_pol", AsyncPolicy, True)
        """
        # Validate policy_type first to get normalized value for error context
        normalized_policy_type = self._validate_policy_type_param(
            policy_type, policy_id=policy_id, allow_none=True
        )

        # Validate policy_id individually using shared helper
        self._validate_policy_id_param(policy_id, policy_type=normalized_policy_type)

        for method_name in self._ASYNC_CHECK_METHODS:
            if hasattr(policy_class, method_name):
                method = getattr(policy_class, method_name)
                if asyncio.iscoroutinefunction(method):
                    if not allow_async:
                        raise PolicyRegistryError(
                            f"Policy '{policy_id}' has async {method_name}() but "
                            f"allow_async=True not specified. "
                            f"Policy plugins must be synchronous by default.",
                            policy_id=policy_id,
                            policy_type=normalized_policy_type,
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
            >>> mixin = MixinPolicyValidation()
            >>> # Enum to string
            >>> mixin._normalize_policy_type(EnumPolicyType.ORCHESTRATOR)
            'orchestrator'
            >>> # Valid string passthrough
            >>> mixin._normalize_policy_type("reducer")
            'reducer'
            >>> # Invalid string raises error
            >>> mixin._normalize_policy_type("invalid")
            PolicyRegistryError: Invalid policy_type: 'invalid'.
                                 Must be one of: ['orchestrator', 'reducer']
        """
        # Validate policy_type using shared helper (requires non-None)
        # With allow_none=False, the overloaded return type is str (never None)
        normalized = self._validate_policy_type_param(
            policy_type, policy_id=None, allow_none=False
        )

        # Validate string against enum values
        valid_types = {e.value for e in EnumPolicyType}
        if normalized not in valid_types:
            raise PolicyRegistryError(
                f"Invalid policy_type: {normalized!r}. "
                f"Must be one of: {sorted(valid_types)}",
                policy_id=None,
                policy_type=normalized,
            )

        return normalized

    @staticmethod
    def _normalize_version(version: str) -> str:
        """Normalize version string for consistent lookups.

        Delegates to the shared normalize_version utility which is the
        SINGLE SOURCE OF TRUTH for version normalization in omnibase_infra.

        This method wraps the shared utility to convert ValueError to
        ProtocolConfigurationError for PolicyRegistry's error contract.

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
            >>> MixinPolicyValidation._normalize_version("1.0")
            '1.0.0'
            >>> MixinPolicyValidation._normalize_version("v2.1")
            '2.1.0'
        """
        try:
            return normalize_version(version)
        except ValueError as e:
            raise ProtocolConfigurationError(
                str(e),
                version=version,
            ) from e


__all__: list[str] = ["MixinPolicyValidation"]
