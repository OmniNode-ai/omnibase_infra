# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Routing Coverage Validator for ONEX Message Types.

This module provides validation functionality to ensure all message types
(Events, Commands, Intents, Projections) defined in the codebase are properly
registered in the routing configuration. It supports startup fail-fast validation
and CI gate integration.

The validator performs two types of discovery:
1. **Message Type Discovery**: Scans source code for classes following ONEX
   message type naming conventions (Event, Command, Intent, Projection suffixes).
2. **Route Registration Discovery**: Inspects the runtime registry or performs
   static analysis to find registered message routes.

Usage:
    # At application startup
    from omnibase_infra.validation import validate_routing_coverage_on_startup

    validate_routing_coverage_on_startup(
        source_directory=Path("src/omnibase_infra"),
        fail_on_unmapped=True,
    )

    # In CI pipelines
    from omnibase_infra.validation import check_routing_coverage_ci

    passed, violations = check_routing_coverage_ci(Path("src/omnibase_infra"))
    if not passed:
        for v in violations:
            print(v.format_for_ci())
        sys.exit(1)

Integration with ONEX Architecture:
    - Supports ONEX 4-node architecture message categories
    - Integrates with ProtocolBindingRegistry for runtime route inspection
    - Returns ModelExecutionShapeViolationResult for consistency with other validators
"""

from __future__ import annotations

import ast
import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.enums.enum_execution_shape_violation import (
    EnumExecutionShapeViolation,
)
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry

logger = logging.getLogger(__name__)

# =============================================================================
# Custom Exception
# =============================================================================


class RoutingCoverageError(RuntimeHostError):
    """Raised when message types are not registered in routing.

    This exception is thrown during application startup if fail-fast mode
    is enabled and there are unmapped message types. It provides a clear
    list of all types that need routing registration.

    Attributes:
        unmapped_types: Set of message type class names that lack routing.
        coverage_percent: Percentage of types that are properly registered.

    Example:
        >>> raise RoutingCoverageError(
        ...     unmapped_types={"OrderCreatedEvent", "PaymentCommand"},
        ...     total_types=10,
        ...     registered_types=8,
        ... )
        RoutingCoverageError: 2 unmapped message types (80.0% coverage):
        OrderCreatedEvent, PaymentCommand
    """

    def __init__(
        self,
        unmapped_types: set[str],
        total_types: int = 0,
        registered_types: int = 0,
    ) -> None:
        """Initialize RoutingCoverageError.

        Args:
            unmapped_types: Set of message type class names without routing.
            total_types: Total number of message types discovered.
            registered_types: Number of types with registered routes.
        """
        self.unmapped_types = unmapped_types
        self.total_types = total_types
        self.registered_types = registered_types

        coverage_percent = (
            (registered_types / total_types * 100) if total_types > 0 else 0.0
        )
        self.coverage_percent = coverage_percent

        types_list = ", ".join(sorted(unmapped_types))
        message = (
            f"{len(unmapped_types)} unmapped message types "
            f"({coverage_percent:.1f}% coverage): {types_list}"
        )
        super().__init__(message=message)


# =============================================================================
# Message Category Detection
# =============================================================================

# Suffix patterns for message type detection
_MESSAGE_SUFFIX_PATTERNS: dict[str, EnumMessageCategory] = {
    "Event": EnumMessageCategory.EVENT,
    "Command": EnumMessageCategory.COMMAND,
    "Intent": EnumMessageCategory.INTENT,
    "Projection": EnumMessageCategory.PROJECTION,
}

# Decorator patterns that indicate a message type
_MESSAGE_DECORATOR_PATTERNS: frozenset[str] = frozenset(
    {
        "message_type",
        "event_type",
        "command_type",
        "intent_type",
        "projection_type",
    }
)

# Base class patterns indicating message inheritance
_MESSAGE_BASE_PATTERNS: frozenset[str] = frozenset(
    {
        "BaseEvent",
        "BaseCommand",
        "BaseIntent",
        "BaseProjection",
        "ModelEvent",
        "ModelCommand",
        "ModelIntent",
        "ModelProjection",
        "EventBase",
        "CommandBase",
        "IntentBase",
        "ProjectionBase",
    }
)


def _get_category_from_suffix(class_name: str) -> EnumMessageCategory | None:
    """Determine message category from class name suffix.

    Args:
        class_name: The class name to analyze.

    Returns:
        EnumMessageCategory if suffix matches, None otherwise.
    """
    for suffix, category in _MESSAGE_SUFFIX_PATTERNS.items():
        if class_name.endswith(suffix):
            return category
    return None


def _get_category_from_base(base_name: str) -> EnumMessageCategory | None:
    """Determine message category from base class name.

    Args:
        base_name: The base class name to analyze.

    Returns:
        EnumMessageCategory if base matches, None otherwise.
    """
    base_lower = base_name.lower()
    if "event" in base_lower:
        return EnumMessageCategory.EVENT
    if "command" in base_lower:
        return EnumMessageCategory.COMMAND
    if "intent" in base_lower:
        return EnumMessageCategory.INTENT
    if "projection" in base_lower:
        return EnumMessageCategory.PROJECTION
    return None


def _has_message_decorator(node: ast.ClassDef) -> tuple[bool, EnumMessageCategory | None]:
    """Check if class has a message type decorator.

    Args:
        node: AST ClassDef node to analyze.

    Returns:
        Tuple of (has_decorator, category_if_found).
    """
    for decorator in node.decorator_list:
        decorator_name = ""
        if isinstance(decorator, ast.Name):
            decorator_name = decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                decorator_name = decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                decorator_name = decorator.func.attr

        if decorator_name in _MESSAGE_DECORATOR_PATTERNS:
            # Try to infer category from decorator name
            if "event" in decorator_name.lower():
                return True, EnumMessageCategory.EVENT
            if "command" in decorator_name.lower():
                return True, EnumMessageCategory.COMMAND
            if "intent" in decorator_name.lower():
                return True, EnumMessageCategory.INTENT
            if "projection" in decorator_name.lower():
                return True, EnumMessageCategory.PROJECTION
            # Generic message_type decorator
            return True, None

    return False, None


def _get_base_classes(node: ast.ClassDef) -> list[str]:
    """Extract base class names from ClassDef node.

    Args:
        node: AST ClassDef node to analyze.

    Returns:
        List of base class names as strings.
    """
    bases: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(base.attr)
    return bases


# =============================================================================
# Discovery Functions
# =============================================================================


def discover_message_types(
    source_directory: Path,
    exclude_patterns: list[str] | None = None,
) -> dict[str, EnumMessageCategory]:
    """Discover all message types defined in source code.

    Scans Python files in the source directory for classes that match
    ONEX message type patterns:
    - Classes ending in 'Event', 'Command', 'Intent', 'Projection'
    - Classes with @message_type decorator (or variants)
    - Classes inheriting from base message types

    Args:
        source_directory: Root directory to scan for message types.
        exclude_patterns: Optional list of glob patterns to exclude.

    Returns:
        Dictionary mapping message class names to their categories.

    Example:
        >>> types = discover_message_types(Path("src/omnibase_infra"))
        >>> print(types)
        {
            'ModelNodeHeartbeatEvent': EnumMessageCategory.EVENT,
            'ModelNodeIntrospectionEvent': EnumMessageCategory.EVENT,
            ...
        }
    """
    if exclude_patterns is None:
        exclude_patterns = [
            "**/test_*.py",
            "**/*_test.py",
            "**/tests/**",
            "**/__pycache__/**",
        ]

    discovered_types: dict[str, EnumMessageCategory] = {}

    # Collect all Python files
    python_files: list[Path] = []
    for pattern in ["**/*.py"]:
        python_files.extend(source_directory.glob(pattern))

    # Filter out excluded files
    filtered_files: list[Path] = []
    for file_path in python_files:
        excluded = False
        for exclude in exclude_patterns:
            if file_path.match(exclude):
                excluded = True
                break
        if not excluded:
            filtered_files.append(file_path)

    # Parse each file and discover message types
    for file_path in filtered_files:
        try:
            file_content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(file_content, filename=str(file_path))
        except SyntaxError as e:
            # Log and skip files with syntax errors
            logger.debug(
                "Skipping file with syntax error during message type discovery: %s "
                "(line %s: %s)",
                file_path,
                e.lineno,
                e.msg,
            )
            continue
        except UnicodeDecodeError as e:
            # Log and skip files with encoding errors
            logger.debug(
                "Skipping file with encoding error during message type discovery: %s "
                "(%s)",
                file_path,
                e.reason,
            )
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            class_name = node.name
            category: EnumMessageCategory | None = None

            # Strategy 1: Check suffix pattern (most common)
            category = _get_category_from_suffix(class_name)

            # Strategy 2: Check for message type decorators
            if category is None:
                has_decorator, decorator_category = _has_message_decorator(node)
                if has_decorator:
                    category = decorator_category

            # Strategy 3: Check base class inheritance
            if category is None:
                bases = _get_base_classes(node)
                for base_name in bases:
                    if base_name in _MESSAGE_BASE_PATTERNS:
                        category = _get_category_from_base(base_name)
                        if category is not None:
                            break

            # If we found a category, record it
            if category is not None:
                discovered_types[class_name] = category

    return discovered_types


def discover_registered_routes(
    registry: ProtocolBindingRegistry | None = None,
    source_directory: Path | None = None,
) -> set[str]:
    """Discover routing registrations from registry or source code.

    This function supports two discovery strategies with different semantics:

    **Strategy 1: Runtime Registry Inspection**
        When `registry` is provided, inspects the ProtocolBindingRegistry
        to find registered protocol handlers. This returns handler categories
        (e.g., "http", "db", "kafka") representing the types of handlers
        registered, NOT individual message type names.

        This is useful for verifying that handler infrastructure is configured,
        but does NOT provide message-level routing coverage.

    **Strategy 2: Static Source Analysis**
        When `source_directory` is provided, scans source files for message
        registration patterns using regex-based static analysis. This returns
        message type class names (e.g., "OrderCreatedEvent", "CreateUserCommand").

        This is the preferred approach for routing coverage validation because
        it discovers which specific message types have handler registrations.

    **Important**: The two strategies return DIFFERENT types of identifiers:
        - Registry: Handler categories (protocol types)
        - Source analysis: Message type class names

    When both parameters are provided, results are combined. However, this
    may produce inconsistent sets that shouldn't be directly compared.
    For routing coverage validation, use source_directory alone.

    Args:
        registry: Optional runtime registry instance to inspect. When used,
            returns handler categories (protocol types like "http", "kafka").
            Useful for infrastructure health checks, not coverage validation.
        source_directory: Optional source directory for static analysis.
            Returns message type class names found in registration patterns.
            Preferred for routing coverage validation.

    Returns:
        Set of discovered route identifiers. The semantics depend on which
        parameters were provided:
        - Registry only: Handler categories (http, db, kafka)
        - Source directory only: Message type class names (OrderEvent, etc.)
        - Both: Combined set (not recommended for coverage validation)

    Example:
        >>> # Runtime inspection - returns handler categories (infrastructure check)
        >>> routes = discover_registered_routes(registry=get_handler_registry())
        >>> # Returns: {"http", "kafka", "db"}
        >>> # Use case: Verify handler infrastructure is configured

        >>> # Static analysis - returns message type names (coverage validation)
        >>> routes = discover_registered_routes(
        ...     source_directory=Path("src/omnibase_infra")
        ... )
        >>> # Returns: {"OrderCreatedEvent", "UserRegisteredEvent", ...}
        >>> # Use case: Validate message routing coverage

        >>> # For coverage validation, compare with discover_message_types():
        >>> message_types = discover_message_types(Path("src/omnibase_infra"))
        >>> registered = discover_registered_routes(source_directory=Path("src"))
        >>> unmapped = set(message_types.keys()) - registered
    """
    registered_types: set[str] = set()

    # Strategy 1: Inspect runtime registry
    if registry is not None:
        # Get all registered protocol types
        protocol_types = registry.list_protocols()
        # Note: Protocol types are handler categories (http, db, kafka),
        # not individual message types. For full message routing coverage,
        # we need to inspect message-to-handler bindings separately.
        # This is a placeholder for future integration with message routing.
        registered_types.update(protocol_types)

    # Strategy 2: Static analysis of registration calls
    if source_directory is not None:
        registered_types.update(_discover_routes_static(source_directory))

    return registered_types


def _discover_routes_static(source_directory: Path) -> set[str]:
    """Discover registered routes through regex-based static analysis.

    Searches for patterns like:
    - registry.register(MessageType, handler)
    - @route(MessageType)
    - handler_map[MessageType] = ...
    - bind(MessageType, ...)
    - subscribe(topic, MessageType)

    Note:
        This function uses regex patterns, not AST parsing, for discovery.
        This approach is faster but has limitations:

        **May produce false positives:**
        - Code in comments or docstrings matching the patterns
        - String literals that happen to match (e.g., error messages)
        - Test fixtures or mocks that aren't real registrations

        **May miss registrations:**
        - Dynamically constructed registration calls
        - Non-standard registration patterns
        - Registrations via factory functions or metaclasses
        - Registrations in configuration files (YAML, JSON)

        For precise discovery, consider AST-based analysis or runtime
        registry inspection.

    Args:
        source_directory: Root directory to scan.

    Returns:
        Set of message type names found in registration patterns.
    """
    registered_types: set[str] = set()

    # Registration patterns to search for.
    # Each pattern captures the message type name as group 1.
    #
    # PATTERN ORDERING: Patterns are ordered by specificity/reliability:
    # 1. Decorator patterns - most explicit routing declarations
    # 2. Method call patterns - programmatic registration
    # 3. Dictionary patterns - mapping-based registration
    #
    # Note: Order doesn't affect matching (all patterns run independently),
    # but documents the relative reliability of each pattern type.
    registration_patterns = [
        # Decorator patterns (most explicit routing declarations)
        # Matches: @route(OrderEvent), @handle("OrderEvent"), @handler(OrderEvent)
        # False positive risk: Low - decorators are typically for routing
        r"@(?:route|handle|handler)\s*\(\s*['\"]?(\w+)['\"]?\s*\)",
        # Method call patterns (programmatic registration)
        # Matches: registry.register(OrderEvent, handler)
        # False positive risk: Medium - .register() is a common method name
        r"\.register\s*\(\s*['\"]?(\w+)['\"]?\s*,",
        # Matches: event_bus.bind(OrderEvent, handler)
        # False positive risk: Medium - .bind() could be used for other purposes
        r"\.bind\s*\(\s*['\"]?(\w+)['\"]?\s*,",
        # Matches: consumer.subscribe(topic, OrderEvent)
        # False positive risk: Medium - second argument detection is heuristic
        r"\.subscribe\s*\([^,]+,\s*['\"]?(\w+)['\"]?\s*\)",
        # Dictionary patterns (mapping-based registration)
        # Matches: handler_map[OrderEvent] = ...
        # False positive risk: Low - specific to handler_map convention
        r"handler_map\s*\[\s*['\"]?(\w+)['\"]?\s*\]",
    ]

    compiled_patterns = [re.compile(p) for p in registration_patterns]

    # Scan all Python files, excluding test files to reduce false positives
    for file_path in source_directory.glob("**/*.py"):
        path_str = str(file_path)
        # Skip __pycache__ directories
        if "__pycache__" in path_str:
            continue
        # Skip test files - they often contain mock registrations that
        # would produce false positives (e.g., `registry.register(MockEvent, ...)`)
        if file_path.name.startswith("test_") or file_path.name.endswith("_test.py"):
            continue
        if file_path.name == "conftest.py":
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
            for pattern in compiled_patterns:
                matches = pattern.findall(content)
                registered_types.update(matches)
        except UnicodeDecodeError as e:
            logger.debug(
                "Skipping file with encoding error during route discovery: %s (%s)",
                file_path,
                e.reason,
            )
            continue
        except OSError as e:
            logger.debug(
                "Skipping file with OS error during route discovery: %s (%s)",
                file_path,
                e.strerror,
            )
            continue

    return registered_types


# =============================================================================
# Routing Coverage Validator
# =============================================================================


class RoutingCoverageValidator:
    """Validator for ensuring all message types have registered routes.

    This validator checks that every message type discovered in the source code
    has a corresponding route registration. It supports both runtime inspection
    and static analysis modes.

    Attributes:
        source_directory: Root directory to scan for message types.
        registry: Optional runtime registry for route inspection.

    Thread Safety:
        This validator uses lazy initialization with double-checked locking.
        The first call to validate_coverage() or get_unmapped_types() triggers
        discovery, which is protected by a threading.Lock. Subsequent calls
        use cached results without locking overhead.

        Multiple threads can safely call validation methods concurrently.
        The cached discovery results are immutable after initialization.

    Performance:
        **Lazy Initialization**: Discovery is deferred until first validation
        call. This allows creating validators early without blocking on I/O.

        **Caching**: After first discovery, results are cached. Repeated calls
        to validate_coverage() reuse cached data without re-scanning.

        **Refresh**: If source files change, call refresh() to clear cache
        and re-discover on next validation call.

        **When to create new instances vs reuse**:
        - Reuse: Within a single application run for consistent results
        - New instance: After source code changes, or for different directories
        - CI pipelines: Typically create fresh instance per run

        For startup validation, use validate_routing_coverage_on_startup()
        which creates a fresh validator and performs immediate validation.

    Example:
        >>> validator = RoutingCoverageValidator(Path("src/omnibase_infra"))
        >>> violations = validator.validate_coverage()
        >>> if violations:
        ...     for v in violations:
        ...         print(v.format_for_ci())

        >>> # Get coverage statistics
        >>> report = validator.get_coverage_report()
        >>> print(f"Coverage: {report['coverage_percent']:.1f}%")

        >>> # After source changes, refresh the cache
        >>> validator.refresh()
        >>> violations = validator.validate_coverage()  # Re-discovers
    """

    def __init__(
        self,
        source_directory: Path,
        registry: ProtocolBindingRegistry | None = None,
    ) -> None:
        """Initialize the routing coverage validator.

        Args:
            source_directory: Root directory to scan for message types.
            registry: Optional runtime registry for route inspection.
        """
        self.source_directory = source_directory
        self.registry = registry
        self._lock = threading.Lock()
        self._discovered_types: dict[str, EnumMessageCategory] | None = None
        self._registered_routes: set[str] | None = None

    def _ensure_discovery(self) -> None:
        """Ensure discovery has been performed (lazy initialization)."""
        if self._discovered_types is None:
            with self._lock:
                if self._discovered_types is None:
                    self._discovered_types = discover_message_types(
                        self.source_directory
                    )
                    self._registered_routes = discover_registered_routes(
                        registry=self.registry,
                        source_directory=self.source_directory,
                    )

    def validate_coverage(self) -> list[ModelExecutionShapeViolationResult]:
        """Validate all message types are registered.

        Returns:
            List of violations for unmapped types. Empty if all types are mapped.

        Example:
            >>> validator = RoutingCoverageValidator(Path("src"))
            >>> violations = validator.validate_coverage()
            >>> for v in violations:
            ...     print(f"{v.file_path}:{v.line_number}: {v.message}")
        """
        self._ensure_discovery()
        assert self._discovered_types is not None
        assert self._registered_routes is not None

        violations: list[ModelExecutionShapeViolationResult] = []
        unmapped = self.get_unmapped_types()

        for type_name in sorted(unmapped):
            category = self._discovered_types.get(type_name)
            category_name = category.value if category else "unknown"

            # Routing coverage is a configuration issue, not specific to any handler type.
            # handler_type is None because this violation is about missing routing
            # registration, not about a specific handler's behavior.
            # Use UNMAPPED_MESSAGE_ROUTE for semantic correctness - this is a routing
            # configuration issue, not a topic/category mismatch.
            violation = ModelExecutionShapeViolationResult(
                violation_type=EnumExecutionShapeViolation.UNMAPPED_MESSAGE_ROUTE,
                handler_type=None,  # Routing coverage is not handler-specific
                file_path=str(self.source_directory),
                line_number=1,
                message=(
                    f"Message type '{type_name}' ({category_name}) is not registered "
                    f"in routing configuration"
                ),
                severity="error",
            )
            violations.append(violation)

        return violations

    def get_unmapped_types(self) -> set[str]:
        """Get message types that are not registered.

        Returns:
            Set of message type class names without routing registration.
        """
        self._ensure_discovery()
        assert self._discovered_types is not None
        assert self._registered_routes is not None

        discovered_names = set(self._discovered_types.keys())
        return discovered_names - self._registered_routes

    def get_coverage_report(self) -> dict[str, int | float | list[str]]:
        """Get coverage statistics.

        Returns:
            Dictionary containing:
                - total_types: Total number of discovered message types
                - registered_types: Number of types with registered routes
                - unmapped_types: List of type names without routes
                - coverage_percent: Percentage of types with routes
        """
        self._ensure_discovery()
        assert self._discovered_types is not None
        assert self._registered_routes is not None

        total = len(self._discovered_types)
        unmapped = self.get_unmapped_types()
        registered = total - len(unmapped)
        coverage = (registered / total * 100) if total > 0 else 100.0

        return {
            "total_types": total,
            "registered_types": registered,
            "unmapped_types": sorted(unmapped),
            "coverage_percent": coverage,
        }

    def fail_fast_on_unmapped(self) -> None:
        """Raise exception if any message types are unmapped.

        This method is intended for use during application startup to ensure
        all message types have routing configured before the application
        accepts requests.

        Raises:
            RoutingCoverageError: If any message types are unmapped.

        Example:
            >>> validator = RoutingCoverageValidator(Path("src"))
            >>> try:
            ...     validator.fail_fast_on_unmapped()
            ... except RoutingCoverageError as e:
            ...     print(f"Missing routes for: {e.unmapped_types}")
            ...     sys.exit(1)
        """
        self._ensure_discovery()
        assert self._discovered_types is not None

        unmapped = self.get_unmapped_types()
        if unmapped:
            total = len(self._discovered_types)
            registered = total - len(unmapped)
            raise RoutingCoverageError(
                unmapped_types=unmapped,
                total_types=total,
                registered_types=registered,
            )

    def refresh(self) -> None:
        """Clear cached discovery results and re-scan.

        Call this method if source files have changed and you need
        to re-discover message types and routes.
        """
        with self._lock:
            self._discovered_types = None
            self._registered_routes = None


# =============================================================================
# Integration Functions
# =============================================================================


def validate_routing_coverage_on_startup(
    source_directory: Path,
    fail_on_unmapped: bool = True,
    registry: ProtocolBindingRegistry | None = None,
) -> bool:
    """Validate routing coverage at application startup.

    This function is designed to be called during application initialization
    to ensure all message types have routing configured. If fail_on_unmapped
    is True, the application will fail fast with a clear error message.

    Args:
        source_directory: Root directory to scan for message types.
        fail_on_unmapped: If True, raise exception on unmapped types.
        registry: Optional runtime registry for route inspection.

    Returns:
        True if all types are mapped, False otherwise.

    Raises:
        RoutingCoverageError: If fail_on_unmapped and types are unmapped.

    Example:
        >>> # In your application startup code
        >>> from omnibase_infra.validation import validate_routing_coverage_on_startup
        >>>
        >>> def main():
        ...     # Validate routing before accepting requests
        ...     validate_routing_coverage_on_startup(
        ...         source_directory=Path("src/myapp"),
        ...         fail_on_unmapped=True,
        ...     )
        ...
        ...     # Continue with application startup
        ...     app.run()
    """
    validator = RoutingCoverageValidator(
        source_directory=source_directory,
        registry=registry,
    )

    if fail_on_unmapped:
        validator.fail_fast_on_unmapped()
        return True

    unmapped = validator.get_unmapped_types()
    return len(unmapped) == 0


def check_routing_coverage_ci(
    source_directory: Path,
    registry: ProtocolBindingRegistry | None = None,
) -> tuple[bool, list[ModelExecutionShapeViolationResult]]:
    """CI gate for routing coverage.

    This function is designed for CI/CD pipeline integration. It returns
    both a pass/fail status and a list of violations that can be formatted
    for CI output (e.g., GitHub Actions annotations).

    Args:
        source_directory: Root directory to scan for message types.
        registry: Optional runtime registry for route inspection.

    Returns:
        Tuple of (passed, violations) where:
            - passed: True if all types have routes
            - violations: List of ModelExecutionShapeViolationResult for CI output

    Example:
        >>> # In your CI script
        >>> from omnibase_infra.validation import check_routing_coverage_ci
        >>>
        >>> passed, violations = check_routing_coverage_ci(Path("src"))
        >>>
        >>> # Output in GitHub Actions format
        >>> for v in violations:
        ...     print(v.format_for_ci())
        >>>
        >>> sys.exit(0 if passed else 1)
    """
    validator = RoutingCoverageValidator(
        source_directory=source_directory,
        registry=registry,
    )

    violations = validator.validate_coverage()
    passed = len(violations) == 0

    return passed, violations


# =============================================================================
# Design Note: Why No Module-Level Singleton
# =============================================================================
#
# Unlike other validators (ExecutionShapeValidator, RuntimeShapeValidator,
# TopicCategoryValidator), the RoutingCoverageValidator is NOT cached as a
# module-level singleton. This is an intentional design decision:
#
# 1. **Parameterized Construction**: Each RoutingCoverageValidator requires a
#    source_directory parameter. Different callers may validate different
#    directories (e.g., "src/handlers" vs "src/events").
#
# 2. **Stateful Discovery**: The validator caches discovered message types
#    and routes internally. This state is tied to the specific source_directory.
#    A singleton would only work for one directory.
#
# 3. **Built-in Caching**: The validator already implements lazy initialization
#    with thread-safe locking (_ensure_discovery). Callers who need repeated
#    validation of the same directory should keep a reference to their validator
#    instance rather than relying on a module-level singleton.
#
# 4. **Refresh Capability**: The refresh() method allows re-scanning after
#    source file changes. This per-instance state management wouldn't work
#    well with a shared singleton.
#
# Performance Recommendation for Callers:
#     # For repeated validation of the same directory, reuse the instance:
#     validator = RoutingCoverageValidator(Path("src/handlers"))
#     violations = validator.validate_coverage()  # Discovery happens here
#     report = validator.get_coverage_report()    # Uses cached discovery
#
#     # For one-time CI validation, the integration functions are sufficient:
#     passed, violations = check_routing_coverage_ci(Path("src/handlers"))

# =============================================================================
# Module Exports
# =============================================================================


__all__: list[str] = [
    # Exception
    "RoutingCoverageError",
    # Discovery functions
    "discover_message_types",
    "discover_registered_routes",
    # Validator class
    "RoutingCoverageValidator",
    # Integration functions
    "validate_routing_coverage_on_startup",
    "check_routing_coverage_ci",
]
