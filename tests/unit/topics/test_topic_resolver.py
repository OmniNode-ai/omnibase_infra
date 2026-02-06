# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for canonical TopicResolver.

Regression tests for OMN-1972: Ensure TopicResolver validates and resolves
realm-agnostic topic suffixes correctly, and rejects environment-prefixed
topics that violate the ONEX topic naming convention.

The TopicResolver is the single canonical function that maps topic suffix
to concrete Kafka topic. All scattered resolve_topic() methods across the
codebase MUST delegate to this class.
"""

import pytest

from omnibase_infra.topics import TopicResolutionError, TopicResolver


class TestTopicResolver:
    """Unit tests for canonical TopicResolver."""

    def test_resolve_valid_evt_topic(self) -> None:
        """Event topics with onex.evt prefix resolve successfully."""
        resolver = TopicResolver()
        result = resolver.resolve("onex.evt.platform.node-introspection.v1")
        assert result == "onex.evt.platform.node-introspection.v1"

    def test_resolve_valid_cmd_topic(self) -> None:
        """Command topics with onex.cmd prefix resolve successfully."""
        resolver = TopicResolver()
        result = resolver.resolve("onex.cmd.platform.request-introspection.v1")
        assert result == "onex.cmd.platform.request-introspection.v1"

    def test_resolve_valid_intent_topic(self) -> None:
        """Intent topics with onex.intent prefix resolve successfully."""
        resolver = TopicResolver()
        result = resolver.resolve("onex.intent.platform.runtime-tick.v1")
        assert result == "onex.intent.platform.runtime-tick.v1"

    def test_resolve_valid_snapshot_topic(self) -> None:
        """Snapshot topics with onex.snapshot prefix resolve successfully."""
        resolver = TopicResolver()
        result = resolver.resolve("onex.snapshot.platform.registration-snapshots.v1")
        assert result == "onex.snapshot.platform.registration-snapshots.v1"

    def test_resolve_is_passthrough(self) -> None:
        """TopicResolver returns suffix unchanged (realm-agnostic)."""
        resolver = TopicResolver()
        suffix = "onex.evt.omniclaude.prompt-submitted.v1"
        assert resolver.resolve(suffix) == suffix

    def test_resolve_rejects_env_prefixed_topic(self) -> None:
        """Environment-prefixed topics are invalid and must be rejected."""
        resolver = TopicResolver()
        with pytest.raises(TopicResolutionError):
            resolver.resolve("dev.onex.evt.platform.node-introspection.v1")

    def test_resolve_rejects_prod_prefixed_topic(self) -> None:
        """Production-prefixed topics are invalid and must be rejected."""
        resolver = TopicResolver()
        with pytest.raises(TopicResolutionError):
            resolver.resolve("prod.onex.evt.platform.node-introspection.v1")

    def test_resolve_rejects_staging_prefixed_topic(self) -> None:
        """Staging-prefixed topics are invalid and must be rejected."""
        resolver = TopicResolver()
        with pytest.raises(TopicResolutionError):
            resolver.resolve("staging.onex.evt.platform.node-introspection.v1")

    def test_resolve_rejects_empty_string(self) -> None:
        """Empty string is not a valid topic suffix."""
        resolver = TopicResolver()
        with pytest.raises(TopicResolutionError):
            resolver.resolve("")

    def test_resolve_rejects_arbitrary_string(self) -> None:
        """Arbitrary strings that do not follow ONEX format are rejected."""
        resolver = TopicResolver()
        with pytest.raises(TopicResolutionError):
            resolver.resolve("not-a-valid-topic")

    def test_resolve_rejects_legacy_dot_format(self) -> None:
        """Legacy dot-separated topics without onex prefix are rejected."""
        resolver = TopicResolver()
        with pytest.raises(TopicResolutionError):
            resolver.resolve("agent.routing.requested.v1")

    def test_error_is_onex_error(self) -> None:
        """TopicResolutionError inherits from OnexError."""
        from omnibase_core.errors import OnexError

        assert issubclass(TopicResolutionError, OnexError)

    def test_error_contains_invalid_suffix_in_message(self) -> None:
        """Error message includes the invalid suffix for debugging."""
        resolver = TopicResolver()
        with pytest.raises(TopicResolutionError, match="bad-topic"):
            resolver.resolve("bad-topic")

    def test_all_platform_suffixes_resolve(self) -> None:
        """All platform-reserved suffixes should resolve successfully."""
        from omnibase_infra.topics.platform_topic_suffixes import ALL_PLATFORM_SUFFIXES

        resolver = TopicResolver()
        for suffix in ALL_PLATFORM_SUFFIXES:
            result = resolver.resolve(suffix)
            assert result == suffix, (
                f"Platform suffix '{suffix}' should resolve to itself"
            )

    def test_all_platform_suffixes_are_realm_agnostic(self) -> None:
        """All platform suffixes start with 'onex.' (no environment prefix)."""
        from omnibase_infra.topics.platform_topic_suffixes import ALL_PLATFORM_SUFFIXES

        for suffix in ALL_PLATFORM_SUFFIXES:
            assert suffix.startswith("onex."), (
                f"Platform suffix '{suffix}' must start with 'onex.' "
                "to be realm-agnostic."
            )

    def test_resolver_instances_are_independent(self) -> None:
        """Multiple TopicResolver instances behave identically."""
        r1 = TopicResolver()
        r2 = TopicResolver()
        topic = "onex.evt.platform.node-registration.v1"
        assert r1.resolve(topic) == r2.resolve(topic)
