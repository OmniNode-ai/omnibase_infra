# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Tests for TopicCategoryValidator and related functions.

Validates that:
- Topic patterns match message categories correctly
- Static (AST) analysis detects topic/category mismatches
- Runtime validation catches category violations
- Handler type to category mappings are enforced
"""

import ast
import tempfile
from pathlib import Path

import pytest

from omnibase_infra.enums.enum_execution_shape_violation import (
    EnumExecutionShapeViolation,
)
from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType
from omnibase_infra.validation.topic_category_validator import (
    HANDLER_EXPECTED_CATEGORIES,
    TOPIC_CATEGORY_PATTERNS,
    TOPIC_SUFFIXES,
    TopicCategoryASTVisitor,
    TopicCategoryValidator,
    validate_message_on_topic,
    validate_topic_categories_in_directory,
    validate_topic_categories_in_file,
)


class TestTopicCategoryPatterns:
    """Test the topic category patterns constants."""

    def test_event_pattern_matches_events_topic(self) -> None:
        """Verify event pattern matches *.events topics."""
        pattern = TOPIC_CATEGORY_PATTERNS[EnumMessageCategory.EVENT]
        assert pattern.match("order.events")
        assert pattern.match("user.events")
        assert pattern.match("payment-service.events")
        assert pattern.match("my_domain.events")

    def test_event_pattern_rejects_non_events_topics(self) -> None:
        """Verify event pattern rejects non-events topics."""
        pattern = TOPIC_CATEGORY_PATTERNS[EnumMessageCategory.EVENT]
        assert not pattern.match("order.commands")
        assert not pattern.match("order.intents")
        assert not pattern.match("order")
        assert not pattern.match("events.order")

    def test_command_pattern_matches_commands_topic(self) -> None:
        """Verify command pattern matches *.commands topics."""
        pattern = TOPIC_CATEGORY_PATTERNS[EnumMessageCategory.COMMAND]
        assert pattern.match("order.commands")
        assert pattern.match("user.commands")
        assert pattern.match("payment-service.commands")

    def test_command_pattern_rejects_non_commands_topics(self) -> None:
        """Verify command pattern rejects non-commands topics."""
        pattern = TOPIC_CATEGORY_PATTERNS[EnumMessageCategory.COMMAND]
        assert not pattern.match("order.events")
        assert not pattern.match("order.intents")
        assert not pattern.match("commands")

    def test_intent_pattern_matches_intents_topic(self) -> None:
        """Verify intent pattern matches *.intents topics."""
        pattern = TOPIC_CATEGORY_PATTERNS[EnumMessageCategory.INTENT]
        assert pattern.match("checkout.intents")
        assert pattern.match("subscription.intents")
        assert pattern.match("transfer-service.intents")

    def test_intent_pattern_rejects_non_intents_topics(self) -> None:
        """Verify intent pattern rejects non-intents topics."""
        pattern = TOPIC_CATEGORY_PATTERNS[EnumMessageCategory.INTENT]
        assert not pattern.match("checkout.events")
        assert not pattern.match("checkout.commands")
        assert not pattern.match("intents")


class TestTopicSuffixes:
    """Test the topic suffix mappings."""

    def test_event_suffix(self) -> None:
        """Verify event category maps to 'events' suffix."""
        assert TOPIC_SUFFIXES[EnumMessageCategory.EVENT] == "events"

    def test_command_suffix(self) -> None:
        """Verify command category maps to 'commands' suffix."""
        assert TOPIC_SUFFIXES[EnumMessageCategory.COMMAND] == "commands"

    def test_intent_suffix(self) -> None:
        """Verify intent category maps to 'intents' suffix."""
        assert TOPIC_SUFFIXES[EnumMessageCategory.INTENT] == "intents"

    def test_projection_has_no_suffix_requirement(self) -> None:
        """Verify projection output type has empty suffix (no naming constraint).

        Note: PROJECTION is now in EnumNodeOutputType, not EnumMessageCategory,
        because projections are node outputs (REDUCER output), not routed messages.
        """
        assert TOPIC_SUFFIXES[EnumNodeOutputType.PROJECTION] == ""


class TestHandlerExpectedCategories:
    """Test the handler to expected categories mapping."""

    def test_effect_handler_categories(self) -> None:
        """Verify effect handlers can process commands and events."""
        categories = HANDLER_EXPECTED_CATEGORIES[EnumHandlerType.EFFECT]
        assert EnumMessageCategory.COMMAND in categories
        assert EnumMessageCategory.EVENT in categories
        assert EnumNodeOutputType.PROJECTION not in categories

    def test_compute_handler_categories(self) -> None:
        """Verify compute handlers can process all message types except projections."""
        categories = HANDLER_EXPECTED_CATEGORIES[EnumHandlerType.COMPUTE]
        assert EnumMessageCategory.EVENT in categories
        assert EnumMessageCategory.COMMAND in categories
        assert EnumMessageCategory.INTENT in categories

    def test_reducer_handler_categories(self) -> None:
        """Verify reducer handlers can process events and output projections.

        Note: PROJECTION is in EnumNodeOutputType because it's a node output type
        (REDUCERs produce projections), not a message category for routing.
        """
        categories = HANDLER_EXPECTED_CATEGORIES[EnumHandlerType.REDUCER]
        assert EnumMessageCategory.EVENT in categories
        assert EnumNodeOutputType.PROJECTION in categories
        assert EnumMessageCategory.COMMAND not in categories

    def test_orchestrator_handler_categories(self) -> None:
        """Verify orchestrator handlers can process events, commands, and intents."""
        categories = HANDLER_EXPECTED_CATEGORIES[EnumHandlerType.ORCHESTRATOR]
        assert EnumMessageCategory.EVENT in categories
        assert EnumMessageCategory.COMMAND in categories
        assert EnumMessageCategory.INTENT in categories


class TestTopicCategoryValidatorValidateMessageTopic:
    """Test TopicCategoryValidator.validate_message_topic method."""

    def test_valid_event_on_events_topic(self) -> None:
        """Verify event on events topic returns no violation."""
        validator = TopicCategoryValidator()
        result = validator.validate_message_topic(
            EnumMessageCategory.EVENT, "order.events"
        )
        assert result is None

    def test_valid_command_on_commands_topic(self) -> None:
        """Verify command on commands topic returns no violation."""
        validator = TopicCategoryValidator()
        result = validator.validate_message_topic(
            EnumMessageCategory.COMMAND, "order.commands"
        )
        assert result is None

    def test_valid_intent_on_intents_topic(self) -> None:
        """Verify intent on intents topic returns no violation."""
        validator = TopicCategoryValidator()
        result = validator.validate_message_topic(
            EnumMessageCategory.INTENT, "checkout.intents"
        )
        assert result is None

    def test_projection_on_any_topic(self) -> None:
        """Verify projection on any topic returns no violation.

        Note: PROJECTION is now in EnumNodeOutputType because it's a node output
        type (not a routed message category). Projections have no topic naming
        constraint because they are internal state outputs from REDUCER nodes.
        """
        validator = TopicCategoryValidator()
        # Projections have no topic naming constraint
        assert (
            validator.validate_message_topic(
                EnumNodeOutputType.PROJECTION, "order.events"
            )
            is None
        )
        assert (
            validator.validate_message_topic(EnumNodeOutputType.PROJECTION, "any.topic")
            is None
        )
        assert (
            validator.validate_message_topic(
                EnumNodeOutputType.PROJECTION, "state.projections"
            )
            is None
        )

    def test_event_on_commands_topic_violation(self) -> None:
        """Verify event on commands topic returns violation."""
        validator = TopicCategoryValidator()
        result = validator.validate_message_topic(
            EnumMessageCategory.EVENT, "order.commands"
        )
        assert result is not None
        assert (
            result.violation_type == EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH
        )
        assert "event" in result.message.lower()
        assert "order.commands" in result.message

    def test_command_on_events_topic_violation(self) -> None:
        """Verify command on events topic returns violation."""
        validator = TopicCategoryValidator()
        result = validator.validate_message_topic(
            EnumMessageCategory.COMMAND, "order.events"
        )
        assert result is not None
        assert (
            result.violation_type == EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH
        )
        assert "command" in result.message.lower()

    def test_intent_on_events_topic_violation(self) -> None:
        """Verify intent on events topic returns violation."""
        validator = TopicCategoryValidator()
        result = validator.validate_message_topic(
            EnumMessageCategory.INTENT, "checkout.events"
        )
        assert result is not None
        assert (
            result.violation_type == EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH
        )
        assert "intent" in result.message.lower()

    def test_event_on_non_conforming_topic_violation(self) -> None:
        """Verify event on non-conforming topic returns violation."""
        validator = TopicCategoryValidator()
        result = validator.validate_message_topic(EnumMessageCategory.EVENT, "order")
        assert result is not None
        assert result.severity == "error"


class TestTopicCategoryValidatorValidateSubscription:
    """Test TopicCategoryValidator.validate_subscription method."""

    def test_valid_reducer_subscription(self) -> None:
        """Verify valid reducer subscription to events topic.

        Note: PROJECTION is in EnumNodeOutputType because it's a node output type.
        """
        validator = TopicCategoryValidator()
        violations = validator.validate_subscription(
            EnumHandlerType.REDUCER,
            ["order.events"],
            [EnumMessageCategory.EVENT, EnumNodeOutputType.PROJECTION],
        )
        assert len(violations) == 0

    def test_invalid_reducer_subscription_to_commands(self) -> None:
        """Verify reducer subscription to commands topic is a violation.

        Note: PROJECTION is in EnumNodeOutputType because it's a node output type.
        """
        validator = TopicCategoryValidator()
        violations = validator.validate_subscription(
            EnumHandlerType.REDUCER,
            ["order.commands"],
            [EnumMessageCategory.EVENT, EnumNodeOutputType.PROJECTION],
        )
        assert len(violations) == 1
        assert (
            violations[0].violation_type
            == EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH
        )
        assert violations[0].handler_type == EnumHandlerType.REDUCER

    def test_multiple_subscriptions_mixed_validity(self) -> None:
        """Verify multiple subscriptions with mixed validity.

        Note: PROJECTION is in EnumNodeOutputType because it's a node output type.
        """
        validator = TopicCategoryValidator()
        violations = validator.validate_subscription(
            EnumHandlerType.REDUCER,
            ["order.events", "order.commands"],
            [EnumMessageCategory.EVENT, EnumNodeOutputType.PROJECTION],
        )
        # commands topic should cause one violation
        assert len(violations) == 1
        assert "order.commands" in violations[0].message

    def test_non_conforming_topic_name_warning(self) -> None:
        """Verify non-conforming topic names generate warnings."""
        validator = TopicCategoryValidator()
        violations = validator.validate_subscription(
            EnumHandlerType.EFFECT,
            ["weird-topic-name"],
            [EnumMessageCategory.EVENT, EnumMessageCategory.COMMAND],
        )
        assert len(violations) == 1
        assert violations[0].severity == "warning"
        assert "does not follow ONEX naming conventions" in violations[0].message


class TestTopicCategoryValidatorExtractDomain:
    """Test TopicCategoryValidator.extract_domain_from_topic method."""

    def test_extract_domain_from_events_topic(self) -> None:
        """Verify domain extraction from events topic."""
        validator = TopicCategoryValidator()
        assert validator.extract_domain_from_topic("order.events") == "order"
        assert (
            validator.extract_domain_from_topic("user-service.events") == "user-service"
        )

    def test_extract_domain_from_commands_topic(self) -> None:
        """Verify domain extraction from commands topic."""
        validator = TopicCategoryValidator()
        assert validator.extract_domain_from_topic("order.commands") == "order"

    def test_extract_domain_from_intents_topic(self) -> None:
        """Verify domain extraction from intents topic."""
        validator = TopicCategoryValidator()
        assert validator.extract_domain_from_topic("checkout.intents") == "checkout"

    def test_extract_domain_from_invalid_topic(self) -> None:
        """Verify None returned for invalid topic names."""
        validator = TopicCategoryValidator()
        assert validator.extract_domain_from_topic("invalid") is None
        assert validator.extract_domain_from_topic("no-suffix") is None
        assert validator.extract_domain_from_topic("events") is None


class TestTopicCategoryValidatorGetExpectedSuffix:
    """Test TopicCategoryValidator.get_expected_topic_suffix method."""

    def test_get_suffix_for_each_category(self) -> None:
        """Verify expected suffixes for all categories."""
        validator = TopicCategoryValidator()
        assert (
            validator.get_expected_topic_suffix(EnumMessageCategory.EVENT) == "events"
        )
        assert (
            validator.get_expected_topic_suffix(EnumMessageCategory.COMMAND)
            == "commands"
        )
        assert (
            validator.get_expected_topic_suffix(EnumMessageCategory.INTENT) == "intents"
        )
        # PROJECTION is now in EnumNodeOutputType (node output, not routed message)
        assert validator.get_expected_topic_suffix(EnumNodeOutputType.PROJECTION) == ""


class TestTopicCategoryASTVisitor:
    """Test TopicCategoryASTVisitor for static analysis."""

    def test_infers_handler_type_from_class_name(self) -> None:
        """Verify handler type inference from class names.

        We verify handler type inference by testing that the inferred type
        produces correct validation results when subscribing to topics.
        An Effect handler subscribing to commands is valid, confirming
        the handler type was correctly inferred as EFFECT.
        """
        source = """
class OrderEffect:
    def setup(self, consumer):
        consumer.subscribe("order.commands")  # Valid for Effect handler
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Effect handler subscribing to commands should be valid (no violations).
        # This confirms the handler type was correctly inferred as EFFECT,
        # since Reducers cannot subscribe to commands topics.
        assert len(visitor.violations) == 0

    def test_detects_subscribe_call_with_wrong_topic(self) -> None:
        """Verify detection of subscribe with wrong topic for handler type."""
        source = """
class OrderReducer:
    def setup(self, consumer):
        consumer.subscribe("order.commands")  # Wrong for reducer
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Reducer subscribing to commands topic should be flagged
        assert len(visitor.violations) == 1
        assert (
            visitor.violations[0].violation_type
            == EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH
        )

    def test_allows_valid_subscription(self) -> None:
        """Verify valid subscriptions don't generate violations."""
        source = """
class OrderReducer:
    def setup(self, consumer):
        consumer.subscribe("order.events")  # Valid for reducer
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        assert len(visitor.violations) == 0

    def test_detects_non_conforming_topic_name(self) -> None:
        """Verify detection of non-conforming topic names."""
        source = """
class OrderEffect:
    def setup(self, consumer):
        consumer.subscribe("weird-topic")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Non-conforming topic should generate warning
        assert len(visitor.violations) == 1
        assert visitor.violations[0].severity == "warning"


class TestValidateTopicCategoriesInFile:
    """Test validate_topic_categories_in_file function."""

    def test_file_not_found(self) -> None:
        """Verify non-existent files return empty violations list.

        File existence is not a topic/category concern - this function
        validates code content, not file system state. Missing files
        are logged as warnings but don't produce violations.
        """
        violations = validate_topic_categories_in_file(Path("/nonexistent/file.py"))
        assert len(violations) == 0

    def test_non_python_file_skipped(self) -> None:
        """Verify non-Python files are skipped."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not python")
            f.flush()
            file_path = Path(f.name)

        try:
            violations = validate_topic_categories_in_file(file_path)
            assert len(violations) == 0
        finally:
            file_path.unlink(missing_ok=True)

    def test_syntax_error_handling(self) -> None:
        """Verify syntax error handling uses correct violation type."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("def broken(:\n")  # Invalid syntax
            f.flush()
            file_path = Path(f.name)

        try:
            violations = validate_topic_categories_in_file(file_path)
            assert len(violations) == 1
            assert (
                violations[0].violation_type == EnumExecutionShapeViolation.SYNTAX_ERROR
            )
            assert (
                violations[0].handler_type is None
            )  # Can't determine from unparseable file
            assert "syntax error" in violations[0].message.lower()
        finally:
            file_path.unlink(missing_ok=True)

    def test_valid_python_file(self) -> None:
        """Verify valid Python file analysis."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("""
class OrderReducer:
    def setup(self, consumer):
        consumer.subscribe("order.events")
""")
            f.flush()
            file_path = Path(f.name)

        try:
            violations = validate_topic_categories_in_file(file_path)
            assert len(violations) == 0
        finally:
            file_path.unlink(missing_ok=True)


class TestValidateMessageOnTopic:
    """Test validate_message_on_topic function."""

    def test_valid_event_on_events_topic(self) -> None:
        """Verify no violation for event on events topic."""

        class OrderCreatedEvent:
            pass

        result = validate_message_on_topic(
            message=OrderCreatedEvent(),
            topic="order.events",
            message_category=EnumMessageCategory.EVENT,
        )
        assert result is None

    def test_event_on_commands_topic_violation(self) -> None:
        """Verify violation for event on commands topic."""

        class OrderCreatedEvent:
            pass

        result = validate_message_on_topic(
            message=OrderCreatedEvent(),
            topic="order.commands",
            message_category=EnumMessageCategory.EVENT,
        )
        assert result is not None
        assert "OrderCreatedEvent" in result.message
        assert "order.commands" in result.message

    def test_projection_on_any_topic(self) -> None:
        """Verify projections can be on any topic.

        Note: PROJECTION is in EnumNodeOutputType because it's a node output type
        (not a routed message category).
        """

        class OrderProjection:
            pass

        result = validate_message_on_topic(
            message=OrderProjection(),
            topic="any.topic",
            message_category=EnumNodeOutputType.PROJECTION,
        )
        assert result is None


class TestValidateTopicCategoriesInDirectory:
    """Test validate_topic_categories_in_directory function."""

    def test_non_existent_directory(self) -> None:
        """Verify empty result for non-existent directory."""
        violations = validate_topic_categories_in_directory(Path("/nonexistent/dir"))
        assert len(violations) == 0

    def test_empty_directory(self) -> None:
        """Verify empty result for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            violations = validate_topic_categories_in_directory(Path(tmpdir))
        assert len(violations) == 0

    def test_directory_with_violations(self) -> None:
        """Verify violations are found in directory scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with violations
            test_file = Path(tmpdir) / "test_handler.py"
            test_file.write_text("""
class OrderReducer:
    def setup(self, consumer):
        consumer.subscribe("order.commands")  # Wrong for reducer
""")
            violations = validate_topic_categories_in_directory(Path(tmpdir))
        assert len(violations) == 1

    def test_recursive_scan(self) -> None:
        """Verify recursive directory scanning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = Path(tmpdir) / "nested"
            subdir.mkdir()
            test_file = subdir / "handler.py"
            test_file.write_text("""
class OrderReducer:
    def setup(self, consumer):
        consumer.subscribe("order.commands")  # Wrong
""")
            violations = validate_topic_categories_in_directory(
                Path(tmpdir), recursive=True
            )
        assert len(violations) == 1

    def test_non_recursive_scan(self) -> None:
        """Verify non-recursive scanning ignores subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = Path(tmpdir) / "nested"
            subdir.mkdir()
            test_file = subdir / "handler.py"
            test_file.write_text("""
class OrderReducer:
    def setup(self, consumer):
        consumer.subscribe("order.commands")  # Wrong
""")
            violations = validate_topic_categories_in_directory(
                Path(tmpdir), recursive=False
            )
        # Should not find violations in subdirectory
        assert len(violations) == 0


class TestFStringTopicExtraction:
    """Test f-string topic extraction in AST analysis.

    These tests verify that the _extract_topic_from_fstring method correctly
    handles various f-string patterns to avoid false positives and negatives
    when extracting topic names for validation.
    """

    def test_fstring_with_interpolated_domain_skipped(self) -> None:
        """Verify f-strings like f"{domain}.events" are skipped.

        When the domain is interpolated, we can only extract ".events" which
        is an incomplete fragment. This should be skipped to avoid false positives.
        """
        source = """
class OrderEffect:
    def setup(self, consumer, domain):
        consumer.subscribe(f"{domain}.events")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - f-string with interpolated domain is skipped
        assert len(visitor.violations) == 0

    def test_fstring_with_interpolated_suffix_skipped(self) -> None:
        """Verify f-strings like f"order.{suffix}" are skipped.

        When the suffix is interpolated, we can only extract "order." which
        is an incomplete fragment. This should be skipped to avoid false negatives.
        """
        source = """
class OrderEffect:
    def setup(self, consumer, suffix):
        consumer.subscribe(f"order.{suffix}")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - f-string with interpolated suffix is skipped
        assert len(visitor.violations) == 0

    def test_fstring_fully_interpolated_skipped(self) -> None:
        """Verify f-strings like f"{prefix}.{suffix}" are skipped.

        When both parts are interpolated, we have no static content to validate.
        """
        source = """
class OrderEffect:
    def setup(self, consumer, prefix, suffix):
        consumer.subscribe(f"{prefix}.{suffix}")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - fully interpolated f-string is skipped
        assert len(visitor.violations) == 0

    def test_fstring_single_expression_skipped(self) -> None:
        """Verify f-strings like f"{get_topic()}" are skipped.

        When the entire topic is a single expression, there's nothing to validate.
        """
        source = """
class OrderEffect:
    def setup(self, consumer):
        consumer.subscribe(f"{self.get_topic()}")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - single expression f-string is skipped
        assert len(visitor.violations) == 0

    def test_fstring_static_content_validated(self) -> None:
        """Verify fully static f-strings are validated.

        f-strings without interpolation (unusual but valid) should be validated
        just like regular string literals.
        """
        source = """
class OrderReducer:
    def setup(self, consumer):
        # Unusual but valid: f-string with no interpolation
        consumer.subscribe(f"order.commands")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should detect the violation - reducer shouldn't subscribe to commands
        assert len(visitor.violations) == 1
        assert (
            visitor.violations[0].violation_type
            == EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH
        )

    def test_fstring_valid_static_no_violation(self) -> None:
        """Verify fully static f-strings with valid topics pass."""
        source = """
class OrderReducer:
    def setup(self, consumer):
        consumer.subscribe(f"order.events")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - valid static f-string
        assert len(visitor.violations) == 0

    def test_fstring_partial_matches_complete_pattern_validated(self) -> None:
        """Verify f-strings where static parts form complete pattern are validated.

        This is a rare edge case where the static parts of an f-string happen
        to form a complete valid topic pattern.
        """
        # This tests a rare case where static parts form a complete pattern
        # For example, an f-string with an empty string expression wouldn't
        # affect the final topic name
        source = """
class OrderReducer:
    def setup(self, consumer):
        # Static parts form complete pattern (empty expression doesn't affect result)
        consumer.subscribe(f"order.events{''}")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # The static part "order.events" is complete and valid for reducer
        # Note: has_interpolation is True, but joined result matches pattern
        assert len(visitor.violations) == 0

    def test_fstring_no_false_positive_from_suffix_only(self) -> None:
        """Verify ".events" from f"{domain}.events" doesn't cause false positives.

        Previously, extracting only static parts could yield ".events" which
        might be incorrectly processed. This test ensures we skip such cases.
        """
        source = """
class OrderReducer:
    def setup(self, consumer, domain):
        # This should NOT generate a warning about non-conforming topic name
        # because we can't reliably determine the full topic name
        consumer.subscribe(f"{domain}.events")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - incomplete f-string is skipped entirely
        assert len(visitor.violations) == 0

    def test_fstring_no_false_negative_from_prefix_only(self) -> None:
        """Verify "order." from f"order.{suffix}" doesn't cause false negatives.

        Previously, extracting only static parts could yield "order." which
        might be incorrectly flagged as invalid. This test ensures we skip such cases.
        """
        source = """
class OrderReducer:
    def setup(self, consumer, suffix):
        # This should NOT generate a warning about non-conforming topic name
        # because we can't reliably determine the full topic name
        consumer.subscribe(f"order.{suffix}")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - incomplete f-string is skipped entirely
        assert len(visitor.violations) == 0


class TestStringConcatenationExtraction:
    """Test string concatenation topic extraction in AST analysis.

    These tests verify that the _extract_topic_from_binop method correctly
    handles various string concatenation patterns to avoid false positives
    and negatives when extracting topic names for validation.
    """

    def test_concat_fully_static_validated(self) -> None:
        """Verify fully static concatenation is validated.

        "order" + ".events" should be evaluated to "order.events" and validated.
        """
        source = """
class OrderReducer:
    def setup(self, consumer):
        # Fully static concatenation - should be validated
        consumer.subscribe("order" + ".commands")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should detect the violation - reducer shouldn't subscribe to commands
        assert len(visitor.violations) == 1
        assert (
            visitor.violations[0].violation_type
            == EnumExecutionShapeViolation.TOPIC_CATEGORY_MISMATCH
        )

    def test_concat_fully_static_valid_no_violation(self) -> None:
        """Verify fully static concatenation with valid topic passes."""
        source = """
class OrderReducer:
    def setup(self, consumer):
        # Fully static concatenation - valid for reducer
        consumer.subscribe("order" + ".events")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - valid static concatenation
        assert len(visitor.violations) == 0

    def test_concat_with_variable_prefix_skipped(self) -> None:
        """Verify concatenation with variable prefix is skipped.

        prefix + ".events" only yields ".events" which is incomplete.
        """
        source = """
class OrderReducer:
    def setup(self, consumer, prefix):
        # Variable prefix - should be skipped
        consumer.subscribe(prefix + ".events")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - incomplete concatenation is skipped
        assert len(visitor.violations) == 0

    def test_concat_with_variable_suffix_skipped(self) -> None:
        """Verify concatenation with variable suffix is skipped.

        "order." + suffix only yields "order." which is incomplete.
        """
        source = """
class OrderReducer:
    def setup(self, consumer, suffix):
        # Variable suffix - should be skipped
        consumer.subscribe("order." + suffix)
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - incomplete concatenation is skipped
        assert len(visitor.violations) == 0

    def test_concat_fully_variable_skipped(self) -> None:
        """Verify fully variable concatenation is skipped.

        prefix + suffix has no static parts that form a valid pattern.
        """
        source = """
class OrderReducer:
    def setup(self, consumer, prefix, suffix):
        # Fully variable - should be skipped
        consumer.subscribe(prefix + suffix)
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - fully variable is skipped
        assert len(visitor.violations) == 0

    def test_concat_nested_static_validated(self) -> None:
        """Verify nested static concatenation is validated.

        "ord" + "er" + ".events" should be evaluated to "order.events".
        """
        source = """
class OrderReducer:
    def setup(self, consumer):
        # Nested static concatenation - should be validated
        consumer.subscribe("ord" + "er" + ".events")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - valid nested static concatenation
        assert len(visitor.violations) == 0

    def test_concat_nested_with_variable_skipped(self) -> None:
        """Verify nested concatenation with variable is skipped.

        "order" + mid + ".events" has a variable in the middle.
        """
        source = """
class OrderReducer:
    def setup(self, consumer, mid):
        # Nested with variable - should be skipped
        consumer.subscribe("order" + mid + ".events")
"""
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("test.py"), validator)
        visitor.visit(tree)
        # Should have no violations - has variable so skipped
        assert len(visitor.violations) == 0


class TestIntegration:
    """Integration tests for the topic category validator."""

    def test_full_handler_analysis(self) -> None:
        """Test complete handler file analysis."""
        source = '''
class OrderEffectHandler:
    """Effect handler for order processing."""

    def __init__(self, producer, consumer):
        self.producer = producer
        self.consumer = consumer

    def setup(self):
        # Valid subscription for effect handler
        self.consumer.subscribe("order.commands")
        self.consumer.subscribe("order.events")

    def handle_create(self, command):
        # Valid: publishing event to events topic
        self.producer.send("order.events", {"type": "OrderCreated"})

class OrderReducerHandler:
    """Reducer handler for order state management."""

    def setup(self, consumer):
        # Valid subscription for reducer
        consumer.subscribe("order.events")

    def handle_event(self, event):
        pass
'''
        tree = ast.parse(source)
        validator = TopicCategoryValidator()
        visitor = TopicCategoryASTVisitor(Path("order_handlers.py"), validator)
        visitor.visit(tree)

        # Should have no violations - all subscriptions are valid
        assert len(visitor.violations) == 0

    def test_violation_format_for_ci(self) -> None:
        """Test that violations can be formatted for CI output."""
        validator = TopicCategoryValidator()
        result = validator.validate_message_topic(
            EnumMessageCategory.EVENT, "order.commands"
        )
        assert result is not None

        # Check CI format
        ci_output = result.format_for_ci()
        assert "::error" in ci_output
        # The CI format uses the enum value (lowercase)
        assert "topic_category_mismatch" in ci_output
