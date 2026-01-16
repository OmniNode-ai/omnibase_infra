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
import logging
import re
from pathlib import Path

from omnibase_infra.enums import (
    EnumExecutionShapeViolation,
    EnumMessageCategory,
    EnumNodeArchetype,
    EnumNodeOutputType,
    EnumValidationSeverity,
)
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)
from omnibase_infra.types import MessageOutputCategory

logger = logging.getLogger(__name__)

# Topic naming patterns for each message category
# Matches patterns like: order.events, user-service.commands, checkout.intents
#
# DESIGN DECISION - Regex vs Substring Matching:
# We use regex patterns instead of simple substring/suffix checks for validation
# because:
#
# 1. **Domain validation**: The pattern `^[\w-]+\.` ensures the domain portion
#    contains only valid characters (alphanumeric, underscore, hyphen). A simple
#    `.endswith(".events")` check would accept malformed topics like "...events"
#    or topics with invalid characters in the domain.
#
# 2. **Exactness**: The `^` and `$` anchors ensure we match the ENTIRE topic name.
#    This prevents false positives on topics like "order.events.dlq" or
#    "prefix.order.events" which would incorrectly match a suffix check.
#
# 3. **Consistency**: All patterns use the same validation logic, making it
#    easier to reason about and extend (e.g., adding new patterns for other
#    topic types).
#
# Trade-off: Regex is slightly slower than substring checks, but the validation
# accuracy and correctness benefits outweigh the performance cost for this
# use case (topic names are short strings, validation happens at configuration
# time, not in hot paths).
TOPIC_CATEGORY_PATTERNS: dict[EnumMessageCategory, re.Pattern[str]] = {
    EnumMessageCategory.EVENT: re.compile(r"^[\w-]+\.events$"),
    EnumMessageCategory.COMMAND: re.compile(r"^[\w-]+\.commands$"),
    EnumMessageCategory.INTENT: re.compile(r"^[\w-]+\.intents$"),
}

# Topic suffix mapping for each message category
# Note: PROJECTION uses EnumNodeOutputType because it's a node output type, not a message category.
# Projections are internal state outputs from REDUCER nodes, not routed messages on Kafka topics.
TOPIC_SUFFIXES: dict[MessageOutputCategory, str] = {
    EnumMessageCategory.EVENT: "events",
    EnumMessageCategory.COMMAND: "commands",
    EnumMessageCategory.INTENT: "intents",
    EnumNodeOutputType.PROJECTION: "",  # Projections have no suffix requirement
}

# Node archetype to expected message categories mapping
#
# DUAL PURPOSE EXPLANATION:
# This mapping serves two purposes in validation:
# 1. INPUT VALIDATION: Which message categories each node archetype can CONSUME
#    (e.g., REDUCER can consume EVENT messages from *.events topics)
# 2. OUTPUT VALIDATION: Which output types each node archetype can PRODUCE
#    (e.g., REDUCER can produce PROJECTION outputs)
#
# Why the union type (MessageOutputCategory)?
# - EnumMessageCategory values (EVENT, COMMAND, INTENT) are for message routing
# - EnumNodeOutputType values (including PROJECTION) are for node output validation
# - REDUCER is unique: it consumes EVENTs (message category) and produces PROJECTIONs
#   (node output type that is NOT routed as a message)
#
# See ADR: docs/decisions/adr-enum-message-category-vs-node-output-type.md
NODE_ARCHETYPE_EXPECTED_CATEGORIES: dict[
    EnumNodeArchetype, list[MessageOutputCategory]
] = {
    EnumNodeArchetype.EFFECT: [
        EnumMessageCategory.COMMAND,
        EnumMessageCategory.EVENT,
    ],
    EnumNodeArchetype.COMPUTE: [
        EnumMessageCategory.EVENT,
        EnumMessageCategory.COMMAND,
        EnumMessageCategory.INTENT,
    ],
    EnumNodeArchetype.REDUCER: [
        EnumMessageCategory.EVENT,
        EnumNodeOutputType.PROJECTION,
    ],
    EnumNodeArchetype.ORCHESTRATOR: [
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
        self.archetype_categories = NODE_ARCHETYPE_EXPECTED_CATEGORIES

    def validate_message_topic(
        self,
        message_category: MessageOutputCategory,
        topic_name: str,
    ) -> ModelExecutionShapeViolationResult | None:
        """Validate that a message category matches its topic pattern.

        Checks if the message category is being read from or written to
        an appropriately named topic according to ONEX conventions.

        Args:
            message_category: The category of the message (EVENT, COMMAND, etc.)
                or node output type (PROJECTION). Projections have no topic
                naming constraint and are always valid.
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
        # Projections have no topic naming constraint (they are node outputs, not routed messages)
        if message_category == EnumNodeOutputType.PROJECTION:
            return None

        # Get the expected pattern for this category
        # Note: patterns dict uses EnumMessageCategory keys. EnumNodeOutputType values
        # won't match (different enum types even with same string values), so lookup
        # will return None for any EnumNodeOutputType passed here. This is correct
        # behavior - we only validate EnumMessageCategory values against topic patterns.
        if isinstance(message_category, EnumMessageCategory):
            expected_pattern = self.patterns.get(message_category)
        else:
            # EnumNodeOutputType values (other than PROJECTION which is handled above)
            # don't have topic naming constraints
            expected_pattern = None
        if expected_pattern is None:
            # Unknown category or node output type - no topic constraint
            return None

        # Check if topic matches the expected pattern
        if expected_pattern.match(topic_name):
            return None

        # Violation detected - category doesn't match topic pattern
        expected_suffix = self.suffixes.get(message_category, "unknown")
        return ModelExecutionShapeViolationResult(
            violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
            node_archetype=None,  # Unknown at runtime validation without handler context
            file_path="<runtime>",  # Runtime validation has no file context
            line_number=1,
            message=(
                f"Topic category mismatch: Message category '{message_category.name}' "
                f"(EnumMessageCategory.{message_category.name}) requires a topic matching "
                f"pattern '<domain>.{expected_suffix}'. Found topic: '{topic_name}'. "
                f"Expected pattern: '*.{expected_suffix}' (e.g., 'order.{expected_suffix}')."
            ),
            severity=EnumValidationSeverity.ERROR,
        )

    def validate_subscription(
        self,
        node_archetype: EnumNodeArchetype,
        subscribed_topics: list[str],
        expected_categories: list[MessageOutputCategory],
    ) -> list[ModelExecutionShapeViolationResult]:
        """Validate that handler subscriptions match expected message types.

        Checks if a handler is subscribed to topics that match the message
        categories it should be consuming based on ONEX architecture rules.

        Args:
            node_archetype: The node archetype (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).
            subscribed_topics: List of Kafka topics the handler subscribes to.
            expected_categories: List of message categories or node output types
                the handler should process (e.g., EVENT, COMMAND, PROJECTION).

        Returns:
            List of violations for any topic that doesn't match expected categories.
            Empty list if all subscriptions are valid or if inputs are invalid types.

        Example:
            >>> validator = TopicCategoryValidator()
            >>> violations = validator.validate_subscription(
            ...     EnumNodeArchetype.REDUCER,
            ...     ["order.events", "order.commands"],  # commands not valid for reducer
            ...     [EnumMessageCategory.EVENT, EnumNodeOutputType.PROJECTION],
            ... )
            >>> assert len(violations) == 1
            >>> assert "order.commands" in violations[0].message
        """
        violations: list[ModelExecutionShapeViolationResult] = []

        # Defensive type checks for list inputs
        if not isinstance(subscribed_topics, list):
            return violations
        if not isinstance(expected_categories, list):
            return violations

        for topic in subscribed_topics:
            # Skip non-string topics
            if not isinstance(topic, str):
                continue
            # Determine what category this topic implies
            inferred_category = self._infer_category_from_topic(topic)

            if inferred_category is None:
                # Topic doesn't follow any known pattern - warning
                violations.append(
                    ModelExecutionShapeViolationResult(
                        violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                        node_archetype=node_archetype,
                        file_path="<runtime>",
                        line_number=1,
                        message=(
                            f"Topic naming convention violation: Topic '{topic}' does not match "
                            f"ONEX naming conventions. Node archetype: '{node_archetype.name}' "
                            f"(EnumNodeArchetype.{node_archetype.name}). Expected topic patterns: "
                            f"'<domain>.events', '<domain>.commands', or '<domain>.intents'. "
                            f"Example valid topics: 'order.events', 'user.commands'."
                        ),
                        severity=EnumValidationSeverity.WARNING,
                    )
                )
                continue

            # Check if the inferred category is in the expected categories
            if inferred_category not in expected_categories:
                expected_names = [c.name for c in expected_categories]
                violations.append(
                    ModelExecutionShapeViolationResult(
                        violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                        node_archetype=node_archetype,
                        file_path="<runtime>",
                        line_number=1,
                        message=(
                            f"Subscription category mismatch: Node archetype '{node_archetype.name}' "
                            f"(EnumNodeArchetype.{node_archetype.name}) subscribed to topic '{topic}' "
                            f"which implies '{inferred_category.name}' messages. "
                            f"Expected message categories for this archetype: [{', '.join(expected_names)}]. "
                            f"Found: {inferred_category.name}. "
                            f"Review NODE_ARCHETYPE_EXPECTED_CATEGORIES for valid subscriptions."
                        ),
                        severity=EnumValidationSeverity.ERROR,
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
        category: MessageOutputCategory,
    ) -> str:
        """Get the expected topic suffix for a message category or node output type.

        Returns the topic suffix that should be used for topics containing
        messages of the specified category or output type.

        Args:
            category: The message category (EVENT, COMMAND, INTENT) or node
                output type (PROJECTION). Projections have no suffix requirement.

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
        current_node_archetype: Inferred node archetype from class context.
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
        self.current_node_archetype: EnumNodeArchetype | None = None
        self.current_class_name: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Visit class definitions to infer node archetype from class name.

        Infers the node archetype based on class name conventions:
        - *Effect -> EFFECT
        - *Compute -> COMPUTE
        - *Reducer -> REDUCER
        - *Orchestrator -> ORCHESTRATOR

        PATTERN MATCHING ORDER:
        The order of keyword checks (effect, compute, reducer, orchestrator)
        matters when a class name contains multiple keywords. The checks are
        ordered by specificity of the ONEX node archetypes:

        1. "effect" - Checked first because EFFECT nodes are most common
           for I/O operations and have the most restrictive constraints
        2. "compute" - Pure computation nodes, checked second
        3. "reducer" - State projection nodes with strict determinism rules
        4. "orchestrator" - Workflow coordination nodes

        Example edge cases:
        - "EffectReducer" would be classified as EFFECT (first match wins)
        - "ComputeOrchestrator" would be classified as COMPUTE

        In practice, handler class names should be unambiguous and follow
        the convention of using a single node archetype in the name suffix
        (e.g., "OrderEffectHandler", not "OrderEffectReducer").

        Args:
            node: The AST ClassDef node.

        Returns:
            The visited node.
        """
        old_archetype = self.current_node_archetype
        old_class_name = self.current_class_name

        self.current_class_name = node.name

        # Infer node archetype from class name.
        # Order matters: first match wins for ambiguous names.
        class_name = node.name.lower()
        if "effect" in class_name:
            self.current_node_archetype = EnumNodeArchetype.EFFECT
        elif "compute" in class_name:
            self.current_node_archetype = EnumNodeArchetype.COMPUTE
        elif "reducer" in class_name:
            self.current_node_archetype = EnumNodeArchetype.REDUCER
        elif "orchestrator" in class_name:
            self.current_node_archetype = EnumNodeArchetype.ORCHESTRATOR

        # Visit children
        self.generic_visit(node)

        # Restore context
        self.current_node_archetype = old_archetype
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
            # f-string handling - extract and validate static parts carefully.
            #
            # LIMITATION: f-strings with interpolated values (e.g., f"{domain}.events")
            # only yield partial static content. We must avoid false positives/negatives
            # from incomplete topic names like ".events" or "order." that could falsely
            # match or miss patterns.
            #
            # Strategy:
            # 1. Extract all static parts from the f-string
            # 2. Check if result forms a COMPLETE valid topic pattern (domain.suffix)
            # 3. If we only get a partial fragment (starts with "." or ends with "."),
            #    skip validation for this f-string - we can't reliably validate it
            # 4. If no static parts exist, skip validation entirely
            topic_name = self._extract_topic_from_fstring(first_arg)
        elif isinstance(first_arg, ast.BinOp) and isinstance(first_arg.op, ast.Add):
            # String concatenation handling (e.g., "order" + ".events")
            #
            # LIMITATION: String concatenation with variables cannot be fully resolved
            # at static analysis time. We use the same conservative approach as f-strings.
            topic_name = self._extract_topic_from_binop(first_arg)

        if topic_name is None:
            return

        # Infer the category from the topic
        inferred_category = self.validator._infer_category_from_topic(topic_name)

        if inferred_category is None:
            # Topic doesn't follow naming convention - add warning
            # Use current_node_archetype if available (from class context), otherwise None
            archetype_context = (
                f" Node archetype: '{self.current_node_archetype.name}' "
                f"(EnumNodeArchetype.{self.current_node_archetype.name})."
                if self.current_node_archetype
                else ""
            )
            self.violations.append(
                ModelExecutionShapeViolationResult(
                    violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                    node_archetype=self.current_node_archetype,  # May be None if outside handler class
                    file_path=str(self.file_path.absolute()),
                    line_number=node.lineno,
                    message=(
                        f"Topic naming convention violation at line {node.lineno}: "
                        f"Topic '{topic_name}' in {method_name}() call does not match ONEX "
                        f"naming conventions.{archetype_context} Expected topic patterns: "
                        f"'<domain>.events', '<domain>.commands', or '<domain>.intents'. "
                        f"Example: 'order.events', 'user.commands'."
                    ),
                    severity=EnumValidationSeverity.WARNING,
                )
            )
            return

        # If we have handler context, validate the subscription makes sense
        if self.current_node_archetype is not None:
            expected_categories = self.validator.archetype_categories.get(
                self.current_node_archetype, []
            )
            if inferred_category not in expected_categories:
                expected_names = [c.name for c in expected_categories]
                self.violations.append(
                    ModelExecutionShapeViolationResult(
                        violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                        node_archetype=self.current_node_archetype,
                        file_path=str(self.file_path.absolute()),
                        line_number=node.lineno,
                        message=(
                            f"Topic category mismatch at line {node.lineno}: "
                            f"Handler '{self.current_class_name or 'unknown'}' with node archetype "
                            f"'{self.current_node_archetype.name}' (EnumNodeArchetype.{self.current_node_archetype.name}) "
                            f"uses topic '{topic_name}' in {method_name}() call. Topic implies "
                            f"'{inferred_category.name}' messages. Expected categories for this "
                            f"archetype: [{', '.join(expected_names)}]. Found: {inferred_category.name}."
                        ),
                        severity=EnumValidationSeverity.ERROR,
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
            # Use current_node_archetype if available (from class context), otherwise None
            expected_topic_suffix = self.validator.suffixes.get(message_hint, "unknown")
            archetype_context = (
                f" Node archetype: '{self.current_node_archetype.name}' "
                f"(EnumNodeArchetype.{self.current_node_archetype.name})."
                if self.current_node_archetype
                else ""
            )
            self.violations.append(
                ModelExecutionShapeViolationResult(
                    violation_type=EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH,
                    node_archetype=self.current_node_archetype,  # May be None if outside handler class
                    file_path=str(self.file_path.absolute()),
                    line_number=node.lineno,
                    message=(
                        f"Message-topic category mismatch at line {node.lineno}: Message appears "
                        f"to be '{message_hint.name}' type (EnumMessageCategory.{message_hint.name}) "
                        f"but is being sent to topic '{topic_name}' which expects "
                        f"'{topic_category.name}' messages.{archetype_context} "
                        f"Expected topic pattern for {message_hint.name}: '*.{expected_topic_suffix}'."
                    ),
                    severity=EnumValidationSeverity.ERROR,
                )
            )

    def _infer_message_category_from_expr(
        self,
        node: ast.expr,
    ) -> EnumMessageCategory | None:
        """Attempt to infer message category from an expression.

        Uses naming conventions to guess the message category. Patterns are
        checked in order of specificity to minimize false positives:

        1. **Suffix patterns** (most reliable):
           - ``*Event``, ``*Created``, ``*Updated``, ``*Deleted`` -> EVENT
           - ``*Command`` -> COMMAND
           - ``*Intent`` -> INTENT

        2. **Prefix patterns** (for CQRS-style naming):
           - ``Create*``, ``Update*``, ``Delete*``, ``Execute*`` -> COMMAND

        Known Limitations:
            - **False positives**: Names like ``EventEmitter`` or ``CommandLine``
              may be incorrectly classified as message types.
            - **Order dependence**: A name ending in both ``Created`` and
              containing ``Command`` (e.g., ``CommandCreated``) will be
              classified as EVENT (suffix match first).
            - Substring matching on prefixes is less reliable than suffix matching.

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

        # ==================================================================
        # Pattern Matching Order: By Specificity (most specific first)
        # ==================================================================
        #
        # Phase 1: Check suffix patterns (most reliable, fewest false positives)
        # Suffix matching is preferred because message types conventionally
        # END with their category: OrderCreatedEvent, CreateOrderCommand
        #
        # IMPORTANT: Suffix order matters! Longer/more-specific suffixes are
        # checked first to avoid partial matches. For example:
        # - "OrderCreatedEvent" should match "Event" suffix (not just "Created")
        # - The suffix list is ordered by semantic specificity, not length
        event_suffixes = ("Event", "Created", "Updated", "Deleted", "Occurred")
        for suffix in event_suffixes:
            if name.endswith(suffix):
                return EnumMessageCategory.EVENT

        if name.endswith("Command"):
            return EnumMessageCategory.COMMAND

        if name.endswith("Intent"):
            return EnumMessageCategory.INTENT

        # Phase 2: Check prefix patterns for CQRS-style command naming
        # Commands often start with verbs: CreateOrder, UpdateUser, DeleteItem
        #
        # NOTE: Prefix matching is LESS reliable than suffix matching because
        # many non-message types start with verbs (e.g., CreateUserService,
        # UpdateHandler, DeleteButton). This phase runs after suffix matching
        # to ensure names like "CreateOrderCommand" match as COMMAND via suffix.
        command_prefixes = ("Create", "Update", "Delete", "Execute", "Do")
        for prefix in command_prefixes:
            if name.startswith(prefix):
                return EnumMessageCategory.COMMAND

        # Phase 3: Check for Model* prefix patterns (ONEX naming convention)
        # ONEX models use "Model" prefix: ModelEvent, ModelCommand, etc.
        #
        # This phase is LAST because it's ONEX-specific and the generic suffix
        # patterns in Phase 1 would already catch most cases (e.g., ModelOrderEvent
        # ends with "Event" and would be caught in Phase 1).
        if name.startswith("ModelEvent"):
            return EnumMessageCategory.EVENT
        if name.startswith("ModelCommand"):
            return EnumMessageCategory.COMMAND
        if name.startswith("ModelIntent"):
            return EnumMessageCategory.INTENT

        return None

    def _extract_topic_from_fstring(
        self,
        node: ast.JoinedStr,
    ) -> str | None:
        """Safely extract a topic name from an f-string AST node.

        f-strings with interpolated values (e.g., f"{domain}.events") only yield
        partial static content when analyzed statically. This method extracts
        the static parts and validates that the result forms a complete, valid
        topic pattern before returning it for validation.

        IMPORTANT - INCOMPLETE TOPIC NAME HANDLING:
        This method intentionally skips validation for f-strings that produce
        incomplete topic fragments. The result may be an incomplete topic name
        for f-strings with interpolated variables. For example:

        - f"{domain}.events" yields only ".events" (domain is unknown)
        - f"order.{suffix}" yields only "order." (suffix is unknown)
        - f"{get_prefix()}.{get_suffix()}" yields "" (all dynamic)

        These incomplete fragments are NOT returned for validation because:
        - False positives: ".events" could falsely match as a valid topic
        - False negatives: "order." would be flagged as invalid when the full
          topic might be "order.events"

        This is a deliberate design decision to prefer missing violations over
        incorrect violations. Runtime validation should catch what static
        analysis cannot.

        Args:
            node: The AST JoinedStr node representing an f-string.

        Returns:
            The extracted topic name if it forms a complete valid pattern,
            or None if the f-string cannot be reliably validated (including
            when only incomplete fragments are available).

        Examples:
            - f"order.events" -> "order.events" (fully static, valid)
            - f"{domain}.events" -> None (partial: ".events" is incomplete)
            - f"order.{suffix}" -> None (partial: "order." is incomplete)
            - f"{prefix}.{suffix}" -> None (no static parts)
            - f"{get_topic()}" -> None (no static parts)
        """
        # Extract all static string parts from the f-string
        static_parts: list[str] = []
        has_interpolation = False

        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                static_parts.append(value.value)
            else:
                # This is a FormattedValue (interpolated expression)
                has_interpolation = True

        # If no static parts, we can't validate anything
        if not static_parts:
            return None

        # Join the static parts to see what we have
        joined = "".join(static_parts)

        # If there are interpolations, check if the result is a valid partial
        if has_interpolation:
            # Skip validation for incomplete fragments that could cause
            # false positives or negatives:
            # - Starts with "." (e.g., ".events" from f"{domain}.events")
            # - Ends with "." (e.g., "order." from f"order.{suffix}")
            # - Contains only a suffix without domain (e.g., ".events", ".commands")
            if joined.startswith(".") or joined.endswith("."):
                return None

            # Check if the joined result matches a complete topic pattern.
            # Only validate if we have what looks like a complete topic name.
            # A complete topic should match: domain.suffix (e.g., "order.events")
            for pattern in self.validator.patterns.values():
                if pattern.match(joined):
                    # This partial f-string happens to form a valid complete topic
                    # This is rare but possible (e.g., f"{'order'}.events" with
                    # a constant expression that evaluates to a string literal)
                    return joined

            # The static parts don't form a valid complete topic pattern.
            # Skip validation to avoid false positives/negatives.
            return None

        # No interpolations - this is a fully static f-string (unusual but valid)
        # For example: f"order.events" (no interpolated values)
        return joined if joined else None

    def _extract_topic_from_binop(
        self,
        node: ast.BinOp,
    ) -> str | None:
        """Safely extract a topic name from a string concatenation BinOp node.

        String concatenation with variables (e.g., prefix + ".events") cannot be
        fully resolved at static analysis time. This method extracts the static
        string parts and applies the same conservative validation as f-strings.

        Args:
            node: The AST BinOp node representing string concatenation.

        Returns:
            The extracted topic name if it forms a complete valid pattern,
            or None if the concatenation cannot be reliably validated.

        Examples:
            - "order" + ".events" -> "order.events" (fully static, valid)
            - prefix + ".events" -> None (partial: ".events" is incomplete)
            - "order." + suffix -> None (partial: "order." is incomplete)
            - prefix + suffix -> None (no static parts that form valid pattern)
        """
        # Recursively extract static string parts from the binary operation
        static_parts: list[str] = []
        has_variable = False

        def collect_static_parts(n: ast.expr) -> None:
            nonlocal has_variable
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                static_parts.append(n.value)
            elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                # Recursively handle nested concatenations
                collect_static_parts(n.left)
                collect_static_parts(n.right)
            else:
                # This is a variable or other non-constant expression
                has_variable = True

        collect_static_parts(node)

        # If no static parts, we can't validate anything
        if not static_parts:
            return None

        # Join the static parts to see what we have
        joined = "".join(static_parts)

        # If there are variables, apply the same conservative approach as f-strings
        if has_variable:
            # Skip validation for incomplete fragments
            if joined.startswith(".") or joined.endswith("."):
                return None

            # Check if the joined result matches a complete topic pattern
            for pattern in self.validator.patterns.values():
                if pattern.match(joined):
                    return joined

            # Skip validation to avoid false positives/negatives
            return None

        # No variables - this is fully static concatenation
        # (e.g., "order" + ".events")
        return joined if joined else None


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

    Uses a cached singleton TopicCategoryValidator for performance in hot paths.
    The TopicCategoryASTVisitor is still created per-file since it stores
    file-specific state (violations, current handler context).

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
        # File not existing is not a topic/category violation - it's a file system issue.
        # Log a warning and return empty list since this function analyzes code content,
        # not file existence. Callers should validate file existence if needed.
        logger.warning(
            "Cannot validate topic categories: file not found: %s", file_path
        )
        return []

    if file_path.suffix != ".py":
        return []  # Skip non-Python files

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        # Syntax error is a file-level issue, not a handler-specific violation.
        # Use SYNTAX_ERROR violation type for AST parse failures.
        # node_archetype is None because we can't analyze the code structure.
        return [
            ModelExecutionShapeViolationResult(
                violation_type=EnumExecutionShapeViolation.SYNTAX_ERROR,
                node_archetype=None,  # Cannot determine archetype from unparseable file
                file_path=str(file_path.absolute()),
                line_number=e.lineno or 1,
                message=(
                    f"Validation error: Cannot parse Python source file for topic category "
                    f"validation. Syntax error at line {e.lineno or 1}: {e.msg}. "
                    f"File: {file_path.name}. Fix the syntax error to enable topic "
                    f"category validation."
                ),
                severity=EnumValidationSeverity.ERROR,
            )
        ]

    # Use cached singleton validator for performance
    # TopicCategoryASTVisitor needs fresh instance per-file (stores file state)
    visitor = TopicCategoryASTVisitor(file_path, _default_validator)
    visitor.visit(tree)

    return visitor.violations


def validate_message_on_topic(
    message: object,
    topic: str,
    message_category: MessageOutputCategory,
) -> ModelExecutionShapeViolationResult | None:
    """Runtime validation that message category matches topic.

    Validates at runtime that a message's category is appropriate
    for the topic it's being published to or consumed from.

    This function should be called at message processing boundaries
    to ensure architectural consistency.

    Uses a cached singleton TopicCategoryValidator for performance in hot paths.

    Args:
        message: The message object (used for context in error messages).
        topic: The Kafka topic name.
        message_category: The declared category of the message or node output type.
            Projections (EnumNodeOutputType.PROJECTION) have no topic constraint.

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
    # Use cached singleton validator for performance in hot paths
    result = _default_validator.validate_message_topic(message_category, topic)

    if result is not None:
        # Enhance the message with message type info if available
        message_type_name = type(message).__name__
        category_name = (
            message_category.name
            if hasattr(message_category, "name")
            else str(message_category)
        )
        enhanced_message = (
            f"Runtime topic validation failure: Message type '{message_type_name}' with "
            f"category '{category_name}' (EnumMessageCategory.{category_name}) is on topic "
            f"'{topic}'. {result.message}"
        )
        return ModelExecutionShapeViolationResult(
            violation_type=result.violation_type,
            node_archetype=result.node_archetype,
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


# ==============================================================================
# Module-Level Singleton Validator
# ==============================================================================
#
# Performance Optimization: The TopicCategoryValidator is stateless after
# initialization (only stores patterns, suffixes, and archetype_categories which
# are all module-level constants). Creating new instances on every validation
# call is wasteful in hot paths. Instead, we use a module-level singleton.
#
# Why a singleton is safe here:
# - The validator only stores references to module-level immutable constants
# - No per-validation state is stored in the validator instance
# - All mutable state is in TopicCategoryASTVisitor (created per-file)
#
# Note: TopicCategoryASTVisitor still requires per-file instantiation because
# it stores file-specific state (violations list, current_node_archetype, etc.).

_default_validator = TopicCategoryValidator()


__all__ = [
    "NODE_ARCHETYPE_EXPECTED_CATEGORIES",
    # Constants
    "TOPIC_CATEGORY_PATTERNS",
    "TOPIC_SUFFIXES",
    "TopicCategoryASTVisitor",
    # Classes
    "TopicCategoryValidator",
    "validate_message_on_topic",
    "validate_topic_categories_in_directory",
    # Functions
    "validate_topic_categories_in_file",
]
