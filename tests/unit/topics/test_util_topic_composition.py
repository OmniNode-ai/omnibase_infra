"""Tests for topic composition utilities."""

import pytest

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


class TestTopicCompositionError:
    """Tests for TopicCompositionError exception class."""

    def test_is_value_error_subclass(self) -> None:
        """TopicCompositionError should be a ValueError subclass."""
        assert issubclass(TopicCompositionError, ValueError)

    def test_error_can_be_raised_with_message(self) -> None:
        """TopicCompositionError should accept a message."""
        with pytest.raises(TopicCompositionError) as exc_info:
            raise TopicCompositionError("Custom error message")
        assert "Custom error message" in str(exc_info.value)

    def test_error_can_be_caught_as_value_error(self) -> None:
        """TopicCompositionError should be catchable as ValueError."""
        try:
            build_full_topic("invalid", "omnibase", SUFFIX_NODE_INTROSPECTION)
        except ValueError:
            pass  # Should be caught
        else:
            pytest.fail("Expected ValueError to be raised")
