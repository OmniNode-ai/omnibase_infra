# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Decorator to exempt functions/classes from Any type validation.

This decorator is recognized by the ``any_type_validator.py`` AST-based validator.
It marks functions or classes as intentionally using Any types with a documented
reason, allowing the validator to skip checking those definitions.

The decorator is a **no-op at runtime** - it simply returns the original function
or class unchanged. Its sole purpose is to serve as a marker that the AST validator
can detect and use to exempt the decorated entity from Any type violations.

Usage
-----
Apply the decorator to functions or classes that intentionally use Any types:

    @allow_any("Required for legacy API compatibility - see OMN-1234")
    def legacy_handler(data: Any) -> Any:
        return process_legacy(data)

    @allow_any("Dynamic plugin dispatch requires Any for flexibility")
    class PluginDispatcher:
        def dispatch(self, payload: Any) -> Any:
            return self._route(payload)

The reason argument is optional but strongly recommended for code review
and maintainability:

    @allow_any()
    def another_handler(data: Any) -> None:
        ...

See Also
--------
- ``omnibase_infra.validation.any_type_validator``: The AST validator that
  recognizes this decorator
- CLAUDE.md: Documentation of Any type policy and enforcement levels
- ``docs/decisions/adr-any-type-pydantic-workaround.md``: ADR for Pydantic Any usage
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

# Type variable for functions and classes
# Using 'object' as the bound per ONEX strong typing policy (no 'Any')
F = TypeVar("F", bound=Callable[..., object])


def allow_any(reason: str = "") -> Callable[[F], F]:
    """Mark a function or class as exempt from Any type validation.

    This decorator is recognized by the AST-based Any type validator and
    causes it to skip checking the decorated function or class for Any
    type violations.

    The decorator is a no-op at runtime - it returns the original function
    or class unchanged.

    Args:
        reason: Documentation of why Any type usage is necessary. While
            optional, providing a reason is strongly recommended for code
            review and future maintainability. The reason is not used at
            runtime but serves as documentation for reviewers.

    Returns:
        A decorator function that returns its input unchanged.

    Examples:
        Basic usage with reason (recommended):

        >>> @allow_any("Required for JSON API responses - see OMN-1234")
        ... def process_json(data: Any) -> Any:
        ...     return transform(data)

        Usage without reason (discouraged but valid):

        >>> @allow_any()
        ... def legacy_function(data: Any) -> None:
        ...     handle(data)

        Class-level exemption:

        >>> @allow_any("Plugin system requires dynamic typing")
        ... class DynamicPlugin:
        ...     def execute(self, payload: Any) -> Any:
        ...         return self._run(payload)

    Note:
        The ``reason`` parameter is purely for documentation. It is not
        stored, logged, or validated at runtime. The validator extracts
        the decorator presence from the AST without evaluating the
        decorator's arguments.
    """
    # The reason parameter is intentionally unused at runtime.
    # It exists solely for documentation purposes in the source code.
    # The AST validator only checks for the decorator's presence,
    # not its arguments.
    _ = reason  # Explicitly mark as intentionally unused

    def decorator(func: F) -> F:
        """Return the function unchanged (no-op decorator)."""
        return func

    return decorator


__all__: list[str] = ["allow_any"]
