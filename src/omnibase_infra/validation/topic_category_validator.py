# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic Category Validator for ONEX Execution Shape Validation.

Validates that message categories match their topic naming patterns in the
ONEX event-driven architecture. This ensures architectural consistency by
enforcing topic naming conventions at both static analysis and runtime.

Topic Naming Conventions:
    - EVENTs: Read from `<domain>.events` topics (e.g., `order.events`)
    - COMMANDs: Read from `<domain>.commands` topics (e.g., `order.commands`)
    - INTENTs: Read from `<domain>.intents` topics (e.g., `checkout.intents`)
    - PROJECTIONs: Can be anywhere (internal state projections)

Validation Modes:
    - Runtime: Validate messages as they flow through the system
    - Static (AST): Analyze Python files for topic/category mismatches in CI

Usage:
    >>> from omnibase_infra.validation import TopicCategoryValidator
    >>> from omnibase_infra.enums import EnumMessageCategory
    >>>
    >>> validator = TopicCategoryValidator()
    >>> result = validator.validate_message_topic(
    ...     EnumMessageCategory.EVENT,
    ...     "order.commands",  # Wrong topic for events
    ... )
    >>> if result is not None:
    ...     print(f"Violation: {result.message}")
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.enums.enum_execution_shape_violation import (
    EnumExecutionShapeViolation,
)
from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)

if TYPE_CHECKING:
    from typing import Literal


# Topic naming patterns for each message category
# Matches patterns like: order.events, user-service.commands, checkout.intents
TOPIC_CATEGORY_PATTERNS: dict[EnumMessageCategory, re.Pattern[str]] = {
    EnumMessageCategory.EVENT: re.compile(r"^[\w-]+\.events$"),
    EnumMessageCategory.COMMAND: re.compile(r"^[\w-]+\.commands$"),
    EnumMessageCategory.INTENT: re.compile(r"^[\w-]+\.intents$"),
}

# Topic suffix mapping for each message category
TOPIC_SUFFIXES: dict[EnumMessageCategory, str] = {
    EnumMessageCategory.EVENT: "events",
    EnumMessageCategory.COMMAND: "commands",
    EnumMessageCategory.INTENT: "intents",
    EnumMessageCategory.PROJECTION: "",  # Projections have no suffix requirement
}

# Handler type to expected message categories mapping
# Defines which message categories each handler type should consume
HANDLER_EXPECTED_CATEGORIES: dict[EnumHandlerType, list[EnumMessageCategory]] = {
    EnumHandlerType.EFFECT: [
        EnumMessageCategory.COMMAND,
        EnumMessageCategory.EVENT,
    ],
    EnumHandlerType.COMPUTE: [
        EnumMessageCategory.EVENT,
        EnumMessageCategory.COMMAND,
        EnumMessageCategory.INTENT,
    ],
    EnumHandlerType.REDUCER: [
        EnumMessageCategory.EVENT,
        EnumMessageCategory.PROJECTION,
    ],
    EnumHandlerType.ORCHESTRATOR: [
        EnumMessageCategory.EVENT,
        EnumMessageCategory.COMMAND,
        EnumMessageCategory.INTENT,
    ],
}


class TopicCategoryValidator:
    """Validator for ensuring message categories match topic patterns.

    Enforces ONEX topic naming conventions by validating that:
    - Events are only read from `*.events` topics
    - Commands are only read from `*.commands` topics
    - Intents are only read from `*.intents` topics
    - Projections can exist anywhere (no naming constraint)

    This validator supports both runtime validation (for message processing)
    and subscription validation (for handler configuration).

    Attributes:
        patterns: Compiled regex patterns for topic validation.
        suffixes: Expected topic suffixes for each message category.

    Example:
        >>> validator = TopicCategoryValidator()
        >>> # Valid: Event on events topic
        >>> result = validator.validate_message_topic(
        ...     EnumMessageCategory.EVENT, "order.events"
        ... )
        >>> assert result is None  # No violation
        >>>
        >>> # Invalid: Event on commands topic
        >>> result = validator.validate_message_topic(
        ...     EnumMessageCategory.EVENT, "order.commands"
        ... )
        >>> assert result is not None  # Violation detected
    """

    def __init__(self) -> None:
        """Initialize the topic category validator with default patterns."""
        self.patterns = TOPIC_CATEGORY_PATTERNS
        self.suffixes = TOPIC_SUFFIXES
        self.handler_categories = HANDLER_EXPECTED_CATEGORIES

    def validate_message_topic(
        self,
        message_category: EnumMessageCategory,
        topic_name: str,
    ) -> ModelExecutionShapeViolationResult | None:
        """Validate that a message category matches its topic pattern.

        Checks if the message category is being read from or written to
        an appropriately named topic according to ONEX conventions.

        Args:
            message_category: The category of the message (EVENT, COMMAND, etc.).
            topic_name: The Kafka topic name being used.

        Returns:
            A ModelExecutionShapeViolationResult if there's a mismatch,
            or None if the message/topic combination is valid.

        Example:
            >>> validator = TopicCategoryValidator()
            >>> # This should pass - event on events topic
            >>> result = validator.validate_message_topic(
            ...     EnumMessageCategory.EVENT, "order.events"
            ... )
            >>> assert result is None
            >>>
            >>> # This should fail - event on commands topic
            >>> result = validator.validate_message_topic(
            ...     EnumMessageCategory.EVENT, "order.commands"
            ... )
            >>> assert result is not None
            >>> assert result.violation_type == EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH
        """
        # Projections have no topic naming constraint
        if message_category == EnumMessageCategory.PROJECTION:
            return None

        # Get the expected pattern for this category
        expected_pattern = self.patterns.get(message_category)
        if expected_pattern is None:
            # Unknown category - should not happen with the enum
            return None

        # Check if topic matches the expected pattern
        if expected_pattern.match(topic_name):
            return None

        # Violation detected - category doesn't match topic pattern
        expected_suffix = self.suffixes.get(message_category, "unknown")
        return ModelExecutionShapeViolationResult(
            violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
            handler_type=EnumHandlerType.EFFECT,  # Default, may be refined by caller
            file_path="<runtime>",  # Runtime validation has no file context
            line_number=1,
            message=(
                f"Message category '{message_category.value}' should be on a topic "
                f"matching '*.<{expected_suffix}>' pattern, but found topic '{topic_name}'"
            ),
            severity="error",
        )

    def validate_subscription(
        self,
        handler_type: EnumHandlerType,
        subscribed_topics: list[str],
        expected_categories: list[EnumMessageCategory],
    ) -> list[ModelExecutionShapeViolationResult]:
        """Validate that handler subscriptions match expected message types.

        Checks if a handler is subscribed to topics that match the message
        categories it should be consuming based on ONEX architecture rules.

        Args:
            handler_type: The type of handler (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
            subscribed_topics: List of Kafka topics the handler subscribes to.
            expected_categories: List of message categories the handler should process.

        Returns:
            List of violations for any topic that doesn't match expected categories.
            Empty list if all subscriptions are valid.

        Example:
            >>> validator = TopicCategoryValidator()
            >>> violations = validator.validate_subscription(
            ...     EnumHandlerType.REDUCER,
            ...     ["order.events", "order.commands"],  # commands not valid for reducer
            ...     [EnumMessageCategory.EVENT, EnumMessageCategory.PROJECTION],
            ... )
            >>> assert len(violations) == 1
            >>> assert "order.commands" in violations[0].message
        """
        violations: list[ModelExecutionShapeViolationResult] = []

        for topic in subscribed_topics:
            # Determine what category this topic implies
            inferred_category = self._infer_category_from_topic(topic)

            if inferred_category is None:
                # Topic doesn't follow any known pattern - warning
                violations.append(
                    ModelExecutionShapeViolationResult(
                        violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                        handler_type=handler_type,
                        file_path="<runtime>",
                        line_number=1,
                        message=(
                            f"Topic '{topic}' does not follow ONEX naming conventions "
                            f"(expected *.events, *.commands, or *.intents)"
                        ),
                        severity="warning",
                    )
                )
                continue

            # Check if the inferred category is in the expected categories
            if inferred_category not in expected_categories:
                violations.append(
                    ModelExecutionShapeViolationResult(
                        violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                        handler_type=handler_type,
                        file_path="<runtime>",
                        line_number=1,
                        message=(
                            f"Handler type '{handler_type.value}' subscribed to topic '{topic}' "
                            f"which implies '{inferred_category.value}' messages, but handler "
                            f"expects categories: {[c.value for c in expected_categories]}"
                        ),
                        severity="error",
                    )
                )

        return violations

    def extract_domain_from_topic(self, topic: str) -> str | None:
        """Extract the domain name from a topic.

        Parses a topic name and returns the domain prefix before the
        category suffix (events, commands, intents).

        Args:
            topic: The Kafka topic name (e.g., 'order.events', 'user-service.commands').

        Returns:
            The domain portion of the topic name, or None if the topic
            doesn't follow the expected pattern.

        Example:
            >>> validator = TopicCategoryValidator()
            >>> validator.extract_domain_from_topic("order.events")
            'order'
            >>> validator.extract_domain_from_topic("user-service.commands")
            'user-service'
            >>> validator.extract_domain_from_topic("invalid-topic")
            None
        """
        for suffix in ("events", "commands", "intents"):
            if topic.endswith(f".{suffix}"):
                domain = topic[: -(len(suffix) + 1)]  # Remove '.' + suffix
                if domain:
                    return domain
        return None

    def get_expected_topic_suffix(
        self,
        category: EnumMessageCategory,
    ) -> str:
        """Get the expected topic suffix for a message category.

        Returns the topic suffix that should be used for topics containing
        messages of the specified category.

        Args:
            category: The message category (EVENT, COMMAND, INTENT, PROJECTION).

        Returns:
            The expected topic suffix ('events', 'commands', 'intents', or ''
            for projections).

        Example:
            >>> validator = TopicCategoryValidator()
            >>> validator.get_expected_topic_suffix(EnumMessageCategory.EVENT)
            'events'
            >>> validator.get_expected_topic_suffix(EnumMessageCategory.COMMAND)
            'commands'
        """
        return self.suffixes.get(category, "")

    def _infer_category_from_topic(
        self,
        topic: str,
    ) -> EnumMessageCategory | None:
        """Infer the message category from a topic name.

        Internal method that determines what type of messages a topic
        is expected to contain based on its naming pattern.

        Args:
            topic: The Kafka topic name.

        Returns:
            The inferred message category, or None if the topic doesn't
            match any known pattern.
        """
        for category, pattern in self.patterns.items():
            if pattern.match(topic):
                return category
        return None


class TopicCategoryASTVisitor(ast.NodeVisitor):
    """AST visitor for detecting topic/category mismatches in Python code.

    Analyzes Python source files to detect potential mismatches between
    message categories and topic names used in producer/consumer calls.

    This visitor looks for patterns like:
    - consumer.subscribe("order.events") with handler processing commands
    - producer.send("user.commands", event_data) - sending event to commands topic

    Attributes:
        violations: List of detected violations.
        file_path: Path to the file being analyzed.
        validator: TopicCategoryValidator instance for validation logic.
        current_handler_type: Inferred handler type from class context.
    """

    def __init__(
        self,
        file_path: Path,
        validator: TopicCategoryValidator,
    ) -> None:
        """Initialize the AST visitor.

        Args:
            file_path: Path to the file being analyzed.
            validator: TopicCategoryValidator instance for validation logic.
        """
        self.violations: list[ModelExecutionShapeViolationResult] = []
        self.file_path = file_path
        self.validator = validator
        self.current_handler_type: EnumHandlerType | None = None
        self.current_class_name: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Visit class definitions to infer handler type from class name.

        Infers the handler type based on class name conventions:
        - *Effect -> EFFECT
        - *Compute -> COMPUTE
        - *Reducer -> REDUCER
        - *Orchestrator -> ORCHESTRATOR

        Args:
            node: The AST ClassDef node.

        Returns:
            The visited node.
        """
        old_handler_type = self.current_handler_type
        old_class_name = self.current_class_name

        self.current_class_name = node.name

        # Infer handler type from class name
        class_name = node.name.lower()
        if "effect" in class_name:
            self.current_handler_type = EnumHandlerType.EFFECT
        elif "compute" in class_name:
            self.current_handler_type = EnumHandlerType.COMPUTE
        elif "reducer" in class_name:
            self.current_handler_type = EnumHandlerType.REDUCER
        elif "orchestrator" in class_name:
            self.current_handler_type = EnumHandlerType.ORCHESTRATOR

        # Visit children
        self.generic_visit(node)

        # Restore context
        self.current_handler_type = old_handler_type
        self.current_class_name = old_class_name

        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Visit function calls to detect topic usage patterns.

        Looks for patterns like:
        - consumer.subscribe("topic_name")
        - producer.send("topic_name", data)
        - event_bus.publish("topic_name", message)

        Args:
            node: The AST Call node.

        Returns:
            The visited node.
        """
        # Check for subscribe/send/publish method calls
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in ("subscribe", "send", "publish", "produce"):
                self._check_topic_call(node, method_name)

        self.generic_visit(node)
        return node

    def _check_topic_call(
        self,
        node: ast.Call,
        method_name: str,
    ) -> None:
        """Check a topic-related method call for category mismatches.

        Args:
            node: The AST Call node.
            method_name: The name of the method being called.
        """
        # Extract topic name from first argument (if string literal)
        if not node.args:
            return

        first_arg = node.args[0]
        topic_name: str | None = None

        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            topic_name = first_arg.value
        elif isinstance(first_arg, ast.JoinedStr):
            # f-string - try to extract static parts
            parts = []
            for value in first_arg.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
            topic_name = "".join(parts) if parts else None

        if topic_name is None:
            return

        # Infer the category from the topic
        inferred_category = self.validator._infer_category_from_topic(topic_name)

        if inferred_category is None:
            # Topic doesn't follow naming convention - add warning
            self.violations.append(
                ModelExecutionShapeViolationResult(
                    violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                    handler_type=self.current_handler_type or EnumHandlerType.EFFECT,
                    file_path=str(self.file_path.absolute()),
                    line_number=node.lineno,
                    message=(
                        f"Topic '{topic_name}' in {method_name}() call does not follow "
                        f"ONEX naming conventions (expected *.events, *.commands, or *.intents)"
                    ),
                    severity="warning",
                )
            )
            return

        # If we have handler context, validate the subscription makes sense
        if self.current_handler_type is not None:
            expected_categories = self.validator.handler_categories.get(
                self.current_handler_type, []
            )
            if inferred_category not in expected_categories:
                self.violations.append(
                    ModelExecutionShapeViolationResult(
                        violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                        handler_type=self.current_handler_type,
                        file_path=str(self.file_path.absolute()),
                        line_number=node.lineno,
                        message=(
                            f"Handler '{self.current_class_name or 'unknown'}' "
                            f"({self.current_handler_type.value}) uses topic '{topic_name}' "
                            f"in {method_name}() call, implying '{inferred_category.value}' "
                            f"messages. Expected categories for this handler: "
                            f"{[c.value for c in expected_categories]}"
                        ),
                        severity="error",
                    )
                )

        # Check for specific anti-patterns
        self._check_send_patterns(node, method_name, topic_name, inferred_category)

    def _check_send_patterns(
        self,
        node: ast.Call,
        method_name: str,
        topic_name: str,
        topic_category: EnumMessageCategory,
    ) -> None:
        """Check for anti-patterns in send/publish calls.

        Looks for patterns like sending events to command topics or
        sending commands to event topics.

        Args:
            node: The AST Call node.
            method_name: The name of the method being called.
            topic_name: The topic name from the call.
            topic_category: The inferred category from the topic name.
        """
        if method_name not in ("send", "publish", "produce"):
            return

        if len(node.args) < 2:
            return

        # Try to infer message type from the second argument (the message/data)
        second_arg = node.args[1]
        message_hint = self._infer_message_category_from_expr(second_arg)

        if message_hint is not None and message_hint != topic_category:
            self.violations.append(
                ModelExecutionShapeViolationResult(
                    violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                    handler_type=self.current_handler_type or EnumHandlerType.EFFECT,
                    file_path=str(self.file_path.absolute()),
                    line_number=node.lineno,
                    message=(
                        f"Message appears to be '{message_hint.value}' type but is being "
                        f"sent to topic '{topic_name}' (expected *.{message_hint.value}s topic)"
                    ),
                    severity="error",
                )
            )

    def _infer_message_category_from_expr(
        self,
        node: ast.expr,
    ) -> EnumMessageCategory | None:
        """Attempt to infer message category from an expression.

        Uses naming conventions to guess the message category:
        - *Event, *Created, *Updated, *Deleted -> EVENT
        - *Command, Create*, Update*, Delete* -> COMMAND
        - *Intent -> INTENT

        Args:
            node: The AST expression node.

        Returns:
            The inferred message category, or None if unable to determine.
        """
        name: str | None = None

        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr

        if name is None:
            return None

        name_lower = name.lower()

        # Event patterns: OrderCreated, UserUpdated, *Event
        if any(
            suffix in name_lower
            for suffix in ("event", "created", "updated", "deleted", "occurred")
        ):
            return EnumMessageCategory.EVENT

        # Command patterns: CreateOrder, *Command
        if any(
            pattern in name_lower
            for pattern in ("command", "create", "update", "delete", "execute", "do")
        ):
            return EnumMessageCategory.COMMAND

        # Intent patterns: CheckoutIntent, *Intent
        if "intent" in name_lower:
            return EnumMessageCategory.INTENT

        return None


def validate_topic_categories_in_file(
    file_path: Path,
) -> list[ModelExecutionShapeViolationResult]:
    """Analyze a Python file for topic/category mismatches using AST.

    Statically analyzes the file to detect:
    - Topics that don't follow ONEX naming conventions
    - Handlers subscribing to inappropriate topic categories
    - Messages being sent to wrong topic types

    This function is designed for CI integration to catch topic
    mismatches before runtime.

    Args:
        file_path: Path to the Python file to analyze.

    Returns:
        List of violations found in the file.

    Example:
        >>> from pathlib import Path
        >>> violations = validate_topic_categories_in_file(
        ...     Path("src/handlers/order_handler.py")
        ... )
        >>> for v in violations:
        ...     print(v.format_for_ci())
    """
    if not file_path.exists():
        return [
            ModelExecutionShapeViolationResult(
                violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                handler_type=EnumHandlerType.EFFECT,
                file_path=str(file_path.absolute()),
                line_number=1,
                message=f"File not found: {file_path}",
                severity="error",
            )
        ]

    if file_path.suffix != ".py":
        return []  # Skip non-Python files

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        return [
            ModelExecutionShapeViolationResult(
                violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                handler_type=EnumHandlerType.EFFECT,
                file_path=str(file_path.absolute()),
                line_number=e.lineno or 1,
                message=f"Syntax error in file: {e.msg}",
                severity="error",
            )
        ]

    validator = TopicCategoryValidator()
    visitor = TopicCategoryASTVisitor(file_path, validator)
    visitor.visit(tree)

    return visitor.violations


def validate_message_on_topic(
    message: object,
    topic: str,
    message_category: EnumMessageCategory,
) -> ModelExecutionShapeViolationResult | None:
    """Runtime validation that message category matches topic.

    Validates at runtime that a message's category is appropriate
    for the topic it's being published to or consumed from.

    This function should be called at message processing boundaries
    to ensure architectural consistency.

    Args:
        message: The message object (used for context in error messages).
        topic: The Kafka topic name.
        message_category: The declared category of the message.

    Returns:
        A ModelExecutionShapeViolationResult if there's a mismatch,
        or None if valid.

    Example:
        >>> from omnibase_infra.validation import validate_message_on_topic
        >>> from omnibase_infra.enums import EnumMessageCategory
        >>>
        >>> result = validate_message_on_topic(
        ...     message=OrderCreatedEvent(...),
        ...     topic="order.events",
        ...     message_category=EnumMessageCategory.EVENT,
        ... )
        >>> assert result is None  # Valid
        >>>
        >>> result = validate_message_on_topic(
        ...     message=OrderCreatedEvent(...),
        ...     topic="order.commands",  # Wrong!
        ...     message_category=EnumMessageCategory.EVENT,
        ... )
        >>> assert result is not None  # Violation
    """
    validator = TopicCategoryValidator()
    result = validator.validate_message_topic(message_category, topic)

    if result is not None:
        # Enhance the message with message type info if available
        message_type_name = type(message).__name__
        enhanced_message = (
            f"Message '{message_type_name}' with category '{message_category.value}' "
            f"is on topic '{topic}'. {result.message}"
        )
        return ModelExecutionShapeViolationResult(
            violation_type=result.violation_type,
            handler_type=result.handler_type,
            file_path=result.file_path,
            line_number=result.line_number,
            message=enhanced_message,
            severity=result.severity,
        )

    return None


def validate_topic_categories_in_directory(
    directory: Path,
    recursive: bool = True,
) -> list[ModelExecutionShapeViolationResult]:
    """Validate all Python files in a directory for topic/category mismatches.

    Convenience function for CI integration that scans a directory
    and validates all Python files.

    Args:
        directory: Path to the directory to scan.
        recursive: Whether to scan subdirectories. Defaults to True.

    Returns:
        List of all violations found across all files.

    Example:
        >>> from pathlib import Path
        >>> violations = validate_topic_categories_in_directory(
        ...     Path("src/handlers/")
        ... )
        >>> # CI gate: fail if any blocking violations
        >>> blocking = [v for v in violations if v.is_blocking()]
        >>> if blocking:
        ...     print(f"Found {len(blocking)} blocking violations")
        ...     exit(1)
    """
    violations: list[ModelExecutionShapeViolationResult] = []

    if not directory.exists():
        return violations

    pattern = "**/*.py" if recursive else "*.py"
    for py_file in directory.glob(pattern):
        if py_file.is_file():
            violations.extend(validate_topic_categories_in_file(py_file))

    return violations


__all__ = [
    # Constants
    "TOPIC_CATEGORY_PATTERNS",
    "TOPIC_SUFFIXES",
    "HANDLER_EXPECTED_CATEGORIES",
    # Classes
    "TopicCategoryValidator",
    "TopicCategoryASTVisitor",
    # Functions
    "validate_topic_categories_in_file",
    "validate_topic_categories_in_directory",
    "validate_message_on_topic",
]
