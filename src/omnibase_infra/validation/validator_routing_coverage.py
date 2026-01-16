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
    - Integrates with RegistryProtocolBinding for runtime route inspection
    - Returns ModelExecutionShapeViolationResult for consistency with other validators
"""

from __future__ import annotations

import ast
import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.enums import (
    EnumExecutionShapeViolation,
    EnumMessageCategory,
    EnumNodeOutputType,
    EnumValidationSeverity,
)
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.validation.model_category_match_result import (
    ModelCategoryMatchResult,
)
from omnibase_infra.models.validation.model_coverage_metrics import (
    ModelCoverageMetrics,
)
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)
from omnibase_infra.types import MessageOutputCategory
from omnibase_infra.validation.infra_validators import should_skip_path

if TYPE_CHECKING:
    from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding

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
# Note: PROJECTION uses EnumNodeOutputType as it's a node output, not a message category
_MESSAGE_SUFFIX_PATTERNS: dict[str, MessageOutputCategory] = {
    "Event": EnumMessageCategory.EVENT,
    "Command": EnumMessageCategory.COMMAND,
    "Intent": EnumMessageCategory.INTENT,
    "Projection": EnumNodeOutputType.PROJECTION,
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


def _get_category_from_suffix(
    class_name: str,
) -> MessageOutputCategory | None:
    """Determine message category or node output type from class name suffix.

    Args:
        class_name: The class name to analyze.

    Returns:
        EnumMessageCategory or EnumNodeOutputType if suffix matches, None otherwise.
    """
    for suffix, category in _MESSAGE_SUFFIX_PATTERNS.items():
        if class_name.endswith(suffix):
            return category
    return None


def _get_category_from_base(
    base_name: str,
) -> MessageOutputCategory | None:
    """Determine message category or node output type from base class name.

    Args:
        base_name: The base class name to analyze.

    Returns:
        EnumMessageCategory or EnumNodeOutputType if base matches, None otherwise.
    """
    base_lower = base_name.lower()
    if "event" in base_lower:
        return EnumMessageCategory.EVENT
    if "command" in base_lower:
        return EnumMessageCategory.COMMAND
    if "intent" in base_lower:
        return EnumMessageCategory.INTENT
    if "projection" in base_lower:
        return EnumNodeOutputType.PROJECTION
    return None


def _has_message_decorator(
    node: ast.ClassDef,
) -> ModelCategoryMatchResult:
    """Check if class has a message type decorator.

    Args:
        node: AST ClassDef node to analyze.

    Returns:
        ModelCategoryMatchResult indicating whether a decorator was found
        and, if so, which category it represents.

    Example:
        >>> # For a class with @event_type decorator
        >>> result = _has_message_decorator(class_node)
        >>> result.matched
        True
        >>> result.category
        <EnumMessageCategory.EVENT: 'event'>

    .. versionchanged:: 0.6.1
        Changed return type from tuple[bool, MessageOutputCategory | None]
        to ModelCategoryMatchResult (OMN-1007).
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
                return ModelCategoryMatchResult.matched_with_category(
                    EnumMessageCategory.EVENT
                )
            if "command" in decorator_name.lower():
                return ModelCategoryMatchResult.matched_with_category(
                    EnumMessageCategory.COMMAND
                )
            if "intent" in decorator_name.lower():
                return ModelCategoryMatchResult.matched_with_category(
                    EnumMessageCategory.INTENT
                )
            if "projection" in decorator_name.lower():
                return ModelCategoryMatchResult.matched_with_category(
                    EnumNodeOutputType.PROJECTION
                )
            # Generic message_type decorator
            return ModelCategoryMatchResult.matched_without_category()

    return ModelCategoryMatchResult.not_matched()


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
) -> dict[str, MessageOutputCategory]:
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
        Dictionary mapping message class names to their categories
        (EnumMessageCategory for EVENT/COMMAND/INTENT, EnumNodeOutputType for PROJECTION).
        Returns empty dict if source_directory is invalid.

    Example:
        >>> types = discover_message_types(Path("src/omnibase_infra"))
        >>> print(types)
        {
            'ModelNodeHeartbeatEvent': EnumMessageCategory.EVENT,
            'ModelNodeIntrospectionEvent': EnumMessageCategory.EVENT,
            ...
        }
    """
    # Defensive type check for source_directory
    if not isinstance(source_directory, Path):
        try:
            source_directory = Path(source_directory)
        except (TypeError, ValueError):
            return {}

    if exclude_patterns is None:
        exclude_patterns = [
            "**/test_*.py",
            "**/*_test.py",
            "**/tests/**",
            "**/__pycache__/**",
        ]
    # Defensive type check for exclude_patterns
    elif not isinstance(exclude_patterns, list):
        exclude_patterns = []

    discovered_types: dict[str, MessageOutputCategory] = {}

    # Collect all Python files
    python_files: list[Path] = []
    for pattern in ["**/*.py"]:
        python_files.extend(source_directory.glob(pattern))

    # Filter out excluded files
    filtered_files: list[Path] = []
    for file_path in python_files:
        excluded = False
        for exclude in exclude_patterns:
            # Skip non-string exclude patterns
            if not isinstance(exclude, str):
                continue
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
            category: MessageOutputCategory | None = None

            # Strategy 1: Check suffix pattern (most common)
            category = _get_category_from_suffix(class_name)

            # Strategy 2: Check for message type decorators
            if category is None:
                match_result = _has_message_decorator(node)
                if match_result.matched:
                    category = match_result.category

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
    registry: RegistryProtocolBinding | None = None,
    source_directory: Path | None = None,
) -> set[str]:
    """Discover routing registrations from registry or source code.

    This function supports two discovery strategies that return DIFFERENT
    types of identifiers. Understanding this distinction is critical for
    correct usage.

    **Strategy 1: Runtime Registry Inspection** (returns HANDLER CATEGORIES)
        When `registry` is provided, inspects the RegistryProtocolBinding
        to find registered protocol handlers.

        Returns: Handler category strings (e.g., "http", "db", "kafka")
        representing the types of protocol handlers registered in the system.

        Use case: Infrastructure health checks - verifying handler infrastructure
        is configured. NOT suitable for message-level routing coverage validation.

        Example return: {"http", "kafka", "db", "grpc"}

    **Strategy 2: Static Source Analysis** (returns MESSAGE TYPE NAMES)
        When `source_directory` is provided, scans source files for message
        registration patterns using regex-based static analysis.

        Returns: Message type class names (e.g., "OrderCreatedEvent",
        "CreateUserCommand") found in registration patterns like
        @route(MessageType) or registry.register(MessageType, handler).

        Use case: Routing coverage validation - discovering which specific
        message types have handler registrations. This is the PREFERRED
        approach for coverage validation.

        Example return: {"OrderCreatedEvent", "UserRegisteredEvent", "PaymentCommand"}

    **Critical Distinction**:
        - Registry strategy: Returns protocol/transport categories (http, kafka)
        - Source analysis: Returns message type class names (OrderCreatedEvent)

        These are fundamentally different identifier spaces and should NOT
        be mixed or compared. For routing coverage validation, always use
        source_directory alone.

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
        - Both: Combined set (not recommended - produces mixed identifier types)

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
        # Defensive type check - convert to Path if needed
        if not isinstance(source_directory, Path):
            try:
                source_directory = Path(source_directory)
            except (TypeError, ValueError):
                pass
            else:
                registered_types.update(_discover_routes_static(source_directory))
        else:
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
        Returns empty set if source_directory is invalid.
    """
    registered_types: set[str] = set()

    # Defensive type check for source_directory
    if not isinstance(source_directory, Path):
        try:
            source_directory = Path(source_directory)
        except (TypeError, ValueError):
            return registered_types

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
        # Skip files in excluded directories (archive, archived, examples, __pycache__)
        # Uses exact path component matching to avoid false positives from substring
        # matching (e.g., "my__pycache__dir" should NOT be skipped)
        if should_skip_path(file_path):
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
        >>> print(f"Coverage: {report.coverage_percent:.1f}%")

        >>> # After source changes, refresh the cache
        >>> validator.refresh()
        >>> violations = validator.validate_coverage()  # Re-discovers
    """

    def __init__(
        self,
        source_directory: Path,
        registry: RegistryProtocolBinding | None = None,
    ) -> None:
        """Initialize the routing coverage validator.

        Args:
            source_directory: Root directory to scan for message types.
            registry: Optional runtime registry for route inspection.
        """
        self.source_directory = source_directory
        self.registry = registry
        self._lock = threading.Lock()
        # Explicit initialization flag - more robust than checking data fields
        # See _ensure_discovery() docstring for thread safety rationale
        self._initialized = False
        self._discovered_types: dict[str, MessageOutputCategory] | None = None
        self._registered_routes: set[str] | None = None

    def _ensure_discovery(self) -> None:
        """Ensure discovery has been performed (lazy initialization).

        Thread Safety:
            Uses double-checked locking with an explicit _initialized flag.

            Why a separate flag instead of checking data fields?
            - Eliminates ordering dependency between field assignments
            - Future refactoring can't accidentally break the invariant
            - The flag is only set True AFTER all fields are fully populated
            - Any thread seeing _initialized=True is guaranteed to see
              both _discovered_types and _registered_routes populated

            The pattern:
            1. Check _initialized without lock (fast path for already-initialized)
            2. Acquire lock and re-check (prevents duplicate initialization)
            3. Set all data fields first
            4. Set _initialized=True last (makes state visible atomically)
        """
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._discovered_types = discover_message_types(
                        self.source_directory
                    )
                    self._registered_routes = discover_registered_routes(
                        registry=self.registry,
                        source_directory=self.source_directory,
                    )
                    # CRITICAL: Set _initialized LAST to ensure all fields
                    # are visible to other threads before they skip the lock
                    self._initialized = True

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

            # Routing coverage is a configuration issue, not specific to any node archetype.
            # node_archetype is None because this violation is about missing routing
            # registration, not about a specific handler's behavior.
            # Use UNMAPPED_MESSAGE_ROUTE for semantic correctness - this is a routing
            # configuration issue, not a topic/category mismatch.
            violation = ModelExecutionShapeViolationResult(
                violation_type=EnumExecutionShapeViolation.UNMAPPED_MESSAGE_ROUTE,
                node_archetype=None,  # Routing coverage is not archetype-specific
                file_path=str(self.source_directory),
                line_number=1,
                message=(
                    f"Message type '{type_name}' ({category_name}) is not registered "
                    f"in routing configuration"
                ),
                severity=EnumValidationSeverity.ERROR,
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

    def get_coverage_report(self) -> ModelCoverageMetrics:
        """Get coverage statistics.

        Returns:
            ModelCoverageMetrics containing:
                - total_types: Total number of discovered message types
                - registered_types: Number of types with registered routes
                - unmapped_types: List of type names without routes
                - coverage_percent: Percentage of types with routes
        """
        self._ensure_discovery()
        assert self._discovered_types is not None
        assert self._registered_routes is not None

        return ModelCoverageMetrics.from_discovery(
            discovered_types=self._discovered_types,
            registered_routes=self._registered_routes,
        )

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

        Thread Safety:
            Clears _initialized FIRST to ensure any concurrent readers
            see the invalidated state and wait for the lock.
        """
        with self._lock:
            # CRITICAL: Clear _initialized FIRST to invalidate
            # Any thread checking this will then acquire the lock
            self._initialized = False
            self._discovered_types = None
            self._registered_routes = None


# =============================================================================
# Integration Functions
# =============================================================================


def validate_routing_coverage_on_startup(
    source_directory: Path,
    fail_on_unmapped: bool = True,
    registry: RegistryProtocolBinding | None = None,
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
    registry: RegistryProtocolBinding | None = None,
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
    # Validator class
    "RoutingCoverageValidator",
    "check_routing_coverage_ci",
    # Discovery functions
    "discover_message_types",
    "discover_registered_routes",
    # Integration functions
    "validate_routing_coverage_on_startup",
]
