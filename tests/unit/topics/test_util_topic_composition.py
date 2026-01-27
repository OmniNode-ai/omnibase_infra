"""Tests for topic composition utilities."""

import pytest

from omnibase_core.errors import OnexError
from omnibase_infra.topics import (
    SUFFIX_NODE_INTROSPECTION,
    SUFFIX_NODE_REGISTRATION,
    TopicCompositionError,
    build_full_topic,
)


class TestBuildFullTopic:
    """Tests for build_full_topic() helper."""

    def test_build_full_topic_format(self) -> None:
        """Full topic should be {env}.{namespace}.{suffix}."""
        topic = build_full_topic("dev", "omnibase", SUFFIX_NODE_INTROSPECTION)
        assert topic == f"dev.omnibase.{SUFFIX_NODE_INTROSPECTION}"

    def test_build_full_topic_with_different_envs(self) -> None:
        """Should work with all valid environment prefixes."""
        for env in ["dev", "staging", "prod", "test", "local"]:
            topic = build_full_topic(env, "myapp", SUFFIX_NODE_INTROSPECTION)
            assert topic.startswith(f"{env}.myapp.")

    def test_invalid_suffix_raises(self) -> None:
        """Invalid suffix should raise TopicCompositionError."""
        with pytest.raises(TopicCompositionError):
            build_full_topic("dev", "omnibase", "bad.format")

    def test_invalid_env_raises(self) -> None:
        """Invalid environment should raise TopicCompositionError."""
        with pytest.raises(TopicCompositionError):
            build_full_topic("invalid-env", "omnibase", SUFFIX_NODE_INTROSPECTION)

    def test_empty_namespace_raises(self) -> None:
        """Empty namespace should raise TopicCompositionError."""
        with pytest.raises(TopicCompositionError):
            build_full_topic("dev", "", SUFFIX_NODE_INTROSPECTION)

    def test_namespace_with_hyphen(self) -> None:
        """Namespace with hyphen should be valid."""
        topic = build_full_topic("dev", "my-app", SUFFIX_NODE_INTROSPECTION)
        assert topic == f"dev.my-app.{SUFFIX_NODE_INTROSPECTION}"

    def test_namespace_with_underscore(self) -> None:
        """Namespace with underscore should be valid."""
        topic = build_full_topic("dev", "my_app", SUFFIX_NODE_INTROSPECTION)
        assert topic == f"dev.my_app.{SUFFIX_NODE_INTROSPECTION}"

    def test_namespace_with_invalid_chars_raises(self) -> None:
        """Namespace with invalid characters should raise TopicCompositionError."""
        with pytest.raises(TopicCompositionError):
            build_full_topic("dev", "my.app", SUFFIX_NODE_INTROSPECTION)

    def test_namespace_with_spaces_raises(self) -> None:
        """Namespace with spaces should raise TopicCompositionError."""
        with pytest.raises(TopicCompositionError):
            build_full_topic("dev", "my app", SUFFIX_NODE_INTROSPECTION)

    def test_namespace_starting_with_number(self) -> None:
        """Namespace starting with number is valid.

        Documents current behavior: namespaces like "123app" are accepted
        because isalnum() returns True for strings containing only letters
        and digits, regardless of position.

        Note: If this behavior should change (e.g., to require namespaces
        start with a letter like Python identifiers), update the validation
        in build_full_topic() and change this test to expect
        TopicCompositionError.
        """
        topic = build_full_topic("dev", "123app", SUFFIX_NODE_INTROSPECTION)
        assert topic == f"dev.123app.{SUFFIX_NODE_INTROSPECTION}"

    def test_namespace_all_numbers(self) -> None:
        """Namespace with only numbers is valid.

        Documents current behavior: pure numeric namespaces like "12345"
        are accepted because isalnum() returns True for digit-only strings.
        """
        topic = build_full_topic("dev", "12345", SUFFIX_NODE_INTROSPECTION)
        assert topic == f"dev.12345.{SUFFIX_NODE_INTROSPECTION}"

    def test_build_full_topic_with_different_suffixes(self) -> None:
        """Should work with different valid suffixes."""
        topic1 = build_full_topic("prod", "omnibase", SUFFIX_NODE_INTROSPECTION)
        topic2 = build_full_topic("prod", "omnibase", SUFFIX_NODE_REGISTRATION)
        assert topic1 != topic2
        assert topic1.endswith(SUFFIX_NODE_INTROSPECTION)
        assert topic2.endswith(SUFFIX_NODE_REGISTRATION)

    def test_error_message_contains_valid_envs(self) -> None:
        """Error message for invalid env should list valid options."""
        with pytest.raises(TopicCompositionError) as exc_info:
            build_full_topic("invalid", "omnibase", SUFFIX_NODE_INTROSPECTION)
        error_message = str(exc_info.value)
        assert "dev" in error_message
        assert "prod" in error_message
        assert "staging" in error_message

    def test_error_message_for_invalid_suffix(self) -> None:
        """Error message for invalid suffix should explain the issue."""
        with pytest.raises(TopicCompositionError) as exc_info:
            build_full_topic("dev", "omnibase", "not-a-valid-suffix")
        error_message = str(exc_info.value)
        assert "not-a-valid-suffix" in error_message

    def test_env_case_sensitivity(self) -> None:
        """Environment prefix is case-sensitive - 'Dev' should fail."""
        with pytest.raises(TopicCompositionError):
            build_full_topic("Dev", "omnibase", SUFFIX_NODE_INTROSPECTION)

    def test_namespace_with_unicode_is_valid(self) -> None:
        """Namespace with unicode letters is valid.

        Documents current behavior: Python's str.isalnum() returns True for
        Unicode letters (not just ASCII), so namespaces like "my-äpp" are
        accepted. This is intentional as it enables internationalized namespaces.

        Note: If ASCII-only namespaces are required in the future, update
        the validation in build_full_topic() to use str.isascii() check and
        change this test to expect TopicCompositionError.
        """
        topic = build_full_topic("dev", "my-äpp", SUFFIX_NODE_INTROSPECTION)
        assert topic == f"dev.my-äpp.{SUFFIX_NODE_INTROSPECTION}"

    def test_namespace_at_max_length(self) -> None:
        """Namespace at exactly MAX_NAMESPACE_LENGTH should be valid."""
        from omnibase_infra.topics.util_topic_composition import MAX_NAMESPACE_LENGTH

        max_namespace = "a" * MAX_NAMESPACE_LENGTH
        topic = build_full_topic("dev", max_namespace, SUFFIX_NODE_INTROSPECTION)
        assert f"dev.{max_namespace}." in topic

    def test_namespace_exceeds_max_length_raises(self) -> None:
        """Namespace exceeding MAX_NAMESPACE_LENGTH should raise TopicCompositionError."""
        from omnibase_infra.topics.util_topic_composition import MAX_NAMESPACE_LENGTH

        too_long_namespace = "a" * (MAX_NAMESPACE_LENGTH + 1)
        with pytest.raises(TopicCompositionError) as exc_info:
            build_full_topic("dev", too_long_namespace, SUFFIX_NODE_INTROSPECTION)
        assert "maximum length" in str(exc_info.value).lower()
        assert str(MAX_NAMESPACE_LENGTH) in str(exc_info.value)

    def test_very_long_namespace_raises(self) -> None:
        """Very long namespace (100+ chars) should raise TopicCompositionError."""
        long_namespace = "a" * 150
        with pytest.raises(TopicCompositionError):
            build_full_topic("dev", long_namespace, SUFFIX_NODE_INTROSPECTION)

    def test_namespace_with_consecutive_hyphens(self) -> None:
        """Namespace with consecutive hyphens should be valid."""
        topic = build_full_topic("dev", "my--app", SUFFIX_NODE_INTROSPECTION)
        assert topic == f"dev.my--app.{SUFFIX_NODE_INTROSPECTION}"

    def test_namespace_with_mixed_case_raises(self) -> None:
        """Namespace with mixed case should raise TopicCompositionError.

        Namespaces must be lowercase to ensure consistent topic naming
        since ONEX topic suffixes are lowercase.
        """
        with pytest.raises(TopicCompositionError) as exc_info:
            build_full_topic("dev", "MyApp", SUFFIX_NODE_INTROSPECTION)
        assert "lowercase" in str(exc_info.value).lower()


class TestTopicCompositionError:
    """Tests for TopicCompositionError exception class."""

    def test_is_onex_error_subclass(self) -> None:
        """TopicCompositionError should be an OnexError subclass."""
        assert issubclass(TopicCompositionError, OnexError)

    def test_error_can_be_raised_with_message(self) -> None:
        """TopicCompositionError should accept a message."""
        with pytest.raises(TopicCompositionError) as exc_info:
            raise TopicCompositionError("Custom error message")
        assert "Custom error message" in str(exc_info.value)

    def test_error_can_be_caught_as_onex_error(self) -> None:
        """TopicCompositionError should be catchable as OnexError."""
        try:
            build_full_topic("invalid", "omnibase", SUFFIX_NODE_INTROSPECTION)
        except OnexError:
            pass  # Should be caught
        else:
            pytest.fail("Expected OnexError to be raised")
