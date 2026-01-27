"""Tests for platform topic suffix constants."""

import pytest

from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.topics import (
    ALL_PLATFORM_SUFFIXES,
    SUFFIX_FSM_STATE_TRANSITIONS,
    SUFFIX_NODE_HEARTBEAT,
    SUFFIX_NODE_INTROSPECTION,
    SUFFIX_NODE_REGISTRATION,
    SUFFIX_REGISTRATION_SNAPSHOTS,
    SUFFIX_REQUEST_INTROSPECTION,
    SUFFIX_RUNTIME_TICK,
)


class TestPlatformTopicSuffixes:
    """Tests for platform-reserved topic suffix constants."""

    def test_all_platform_suffixes_are_valid(self) -> None:
        """Every platform suffix constant must pass validation."""
        for suffix in ALL_PLATFORM_SUFFIXES:
            result = validate_topic_suffix(suffix)
            assert result.is_valid, f"Invalid suffix: {suffix} - {result.error}"

    def test_all_platform_suffixes_list_is_complete(self) -> None:
        """ALL_PLATFORM_SUFFIXES should contain all defined constants."""
        expected_suffixes = {
            SUFFIX_NODE_REGISTRATION,
            SUFFIX_NODE_INTROSPECTION,
            SUFFIX_NODE_HEARTBEAT,
            SUFFIX_REQUEST_INTROSPECTION,
            SUFFIX_FSM_STATE_TRANSITIONS,
            SUFFIX_RUNTIME_TICK,
            SUFFIX_REGISTRATION_SNAPSHOTS,
        }
        assert set(ALL_PLATFORM_SUFFIXES) == expected_suffixes

    def test_suffixes_follow_onex_format(self) -> None:
        """All suffixes should follow onex.{kind}.{producer}.{event}.v{n} format."""
        for suffix in ALL_PLATFORM_SUFFIXES:
            assert suffix.startswith("onex."), (
                f"Suffix must start with 'onex.': {suffix}"
            )
            parts = suffix.split(".")
            assert len(parts) == 5, f"Suffix must have 5 parts: {suffix}"
            assert parts[-1].startswith("v"), f"Suffix must end with version: {suffix}"

    def test_suffix_constants_are_strings(self) -> None:
        """All suffix constants should be strings."""
        assert isinstance(SUFFIX_NODE_REGISTRATION, str)
        assert isinstance(SUFFIX_NODE_INTROSPECTION, str)
        assert isinstance(SUFFIX_NODE_HEARTBEAT, str)
        assert isinstance(SUFFIX_REQUEST_INTROSPECTION, str)
        assert isinstance(SUFFIX_FSM_STATE_TRANSITIONS, str)
        assert isinstance(SUFFIX_RUNTIME_TICK, str)
        assert isinstance(SUFFIX_REGISTRATION_SNAPSHOTS, str)

    def test_suffix_kinds_are_valid(self) -> None:
        """All suffixes should use valid message kinds."""
        valid_kinds = {"evt", "cmd", "intent", "snapshot", "dlq"}
        for suffix in ALL_PLATFORM_SUFFIXES:
            parts = suffix.split(".")
            kind = parts[1]
            assert kind in valid_kinds, f"Invalid kind '{kind}' in suffix: {suffix}"

    def test_node_registration_suffix_format(self) -> None:
        """Node registration suffix should have correct format."""
        assert SUFFIX_NODE_REGISTRATION == "onex.evt.platform.node-registration.v1"

    def test_node_introspection_suffix_format(self) -> None:
        """Node introspection suffix should have correct format."""
        assert SUFFIX_NODE_INTROSPECTION == "onex.evt.platform.node-introspection.v1"

    def test_node_heartbeat_suffix_format(self) -> None:
        """Node heartbeat suffix should have correct format."""
        assert SUFFIX_NODE_HEARTBEAT == "onex.evt.platform.node-heartbeat.v1"

    def test_request_introspection_suffix_format(self) -> None:
        """Request introspection suffix should have correct format."""
        assert (
            SUFFIX_REQUEST_INTROSPECTION == "onex.cmd.platform.request-introspection.v1"
        )

    def test_fsm_state_transitions_suffix_format(self) -> None:
        """FSM state transitions suffix should have correct format."""
        assert (
            SUFFIX_FSM_STATE_TRANSITIONS == "onex.evt.platform.fsm-state-transitions.v1"
        )

    def test_runtime_tick_suffix_format(self) -> None:
        """Runtime tick suffix should have correct format."""
        assert SUFFIX_RUNTIME_TICK == "onex.intent.platform.runtime-tick.v1"

    def test_registration_snapshots_suffix_format(self) -> None:
        """Registration snapshots suffix should have correct format."""
        assert (
            SUFFIX_REGISTRATION_SNAPSHOTS
            == "onex.snapshot.platform.registration-snapshots.v1"
        )

    def test_all_platform_suffixes_is_tuple(self) -> None:
        """ALL_PLATFORM_SUFFIXES should be an immutable tuple."""
        assert isinstance(ALL_PLATFORM_SUFFIXES, tuple)

    def test_no_duplicate_suffixes(self) -> None:
        """ALL_PLATFORM_SUFFIXES should not contain duplicates."""
        assert len(ALL_PLATFORM_SUFFIXES) == len(set(ALL_PLATFORM_SUFFIXES))

    def test_suffixes_use_platform_producer(self) -> None:
        """All platform suffixes should use 'platform' as producer."""
        for suffix in ALL_PLATFORM_SUFFIXES:
            parts = suffix.split(".")
            producer = parts[2]
            assert producer == "platform", f"Expected 'platform' producer in: {suffix}"
