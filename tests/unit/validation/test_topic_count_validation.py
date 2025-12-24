"""
Tests for ONEX Event Streaming Topic validation.

Validates that:
- The canonical topic list matches documentation (12 topics, LOCKED for MVP)
- All topics follow the ONEX naming convention (onex.<domain>.<entity>.<event>.v<version>)
- Topic count matches the documented specification in EVENT_STREAMING_TOPICS.md Section 10

This test serves as a contract validation to ensure documentation and implementation stay in sync.
If this test fails, either:
1. A topic was added/removed without updating documentation, OR
2. Documentation was updated without updating this test

Reference: docs/architecture/EVENT_STREAMING_TOPICS.md Section 10
"""

import re

# ============================================================================
# CANONICAL TOPIC DEFINITION - LOCKED FOR MVP
# ============================================================================
# These 12 topics are defined in EVENT_STREAMING_TOPICS.md Section 10.
# Any changes to this list MUST be reflected in the documentation.
# This list is the single source of truth for topic count validation.
# ============================================================================

EXPECTED_TOPIC_COUNT = 12

CANONICAL_ONEX_TOPICS: list[str] = [
    # Node Lifecycle Events (Node -> Registry)
    "onex.node.introspection.published.v1",
    "onex.node.heartbeat.published.v1",
    "onex.node.shutdown.announced.v1",
    # Registry Control Events (Registry -> Nodes)
    "onex.registry.introspection.requested.v1",
    # Registry State Change Events
    "onex.registry.node.registered.v1",
    "onex.registry.node.registration_failed.v1",
    "onex.registry.node.deregistered.v1",
    # Workflow Observability Events
    "onex.registry.workflow.started.v1",
    "onex.registry.workflow.completed.v1",
    "onex.registry.workflow.failed.v1",
    # Infrastructure Signals
    "onex.infra.circuit_breaker.state_changed.v1",
    "onex.infra.error.detected.v1",
]

# Topic naming convention regex pattern
# Format: onex.<domain>.<entity>.<event>.v<version>
ONEX_TOPIC_PATTERN = re.compile(
    r"^onex\."  # Must start with "onex."
    r"[a-z][a-z0-9_-]*\."  # Domain (lowercase, alphanumeric, _, -)
    r"[a-z][a-z0-9_.-]*\."  # Entity (can have dots for nested entities)
    r"[a-z][a-z0-9_-]*\."  # Event (lowercase, alphanumeric, _, -)
    r"v[0-9]+$"  # Version suffix (v1, v2, etc.)
)


class TestTopicCountValidation:
    """Test that canonical topic count matches documentation."""

    def test_canonical_topic_count_matches_expected(self) -> None:
        """Verify the canonical topic list has exactly 12 topics.

        This is the primary validation - if this fails, either topics
        were added/removed or the expected count needs updating.
        """
        actual_count = len(CANONICAL_ONEX_TOPICS)
        assert actual_count == EXPECTED_TOPIC_COUNT, (
            f"Topic count mismatch! Expected {EXPECTED_TOPIC_COUNT} topics "
            f"(documented in EVENT_STREAMING_TOPICS.md Section 10), "
            f"but found {actual_count} topics. "
            f"If topics were intentionally added/removed, update both "
            f"EVENT_STREAMING_TOPICS.md and this test file."
        )

    def test_no_duplicate_topics(self) -> None:
        """Verify no duplicate topics in the canonical list."""
        unique_topics = set(CANONICAL_ONEX_TOPICS)
        assert len(unique_topics) == len(CANONICAL_ONEX_TOPICS), (
            f"Duplicate topics found! "
            f"Canonical list has {len(CANONICAL_ONEX_TOPICS)} entries "
            f"but only {len(unique_topics)} unique topics. "
            f"Duplicates: {[t for t in CANONICAL_ONEX_TOPICS if CANONICAL_ONEX_TOPICS.count(t) > 1]}"
        )


class TestTopicNamingConvention:
    """Test that all topics follow ONEX naming conventions."""

    def test_all_topics_start_with_onex_prefix(self) -> None:
        """Verify all canonical topics start with 'onex.' prefix."""
        for topic in CANONICAL_ONEX_TOPICS:
            assert topic.startswith("onex."), (
                f"Topic '{topic}' does not start with 'onex.' prefix. "
                f"All ONEX topics must follow the naming convention: "
                f"onex.<domain>.<entity>.<event>.v<version>"
            )

    def test_all_topics_have_version_suffix(self) -> None:
        """Verify all canonical topics end with version suffix (e.g., .v1)."""
        version_pattern = re.compile(r"\.v[0-9]+$")
        for topic in CANONICAL_ONEX_TOPICS:
            assert version_pattern.search(topic), (
                f"Topic '{topic}' does not have a version suffix. "
                f"All ONEX topics must end with .v<number> (e.g., .v1, .v2)"
            )

    def test_all_topics_match_naming_convention(self) -> None:
        """Verify all topics match the full ONEX naming pattern."""
        for topic in CANONICAL_ONEX_TOPICS:
            assert ONEX_TOPIC_PATTERN.match(topic), (
                f"Topic '{topic}' does not match ONEX naming convention. "
                f"Expected format: onex.<domain>.<entity>.<event>.v<version>"
            )

    def test_topics_use_lowercase_only(self) -> None:
        """Verify all topics use lowercase characters only."""
        for topic in CANONICAL_ONEX_TOPICS:
            assert topic == topic.lower(), (
                f"Topic '{topic}' contains uppercase characters. "
                f"ONEX topics must be lowercase."
            )

    def test_topics_contain_no_spaces(self) -> None:
        """Verify no topics contain whitespace."""
        for topic in CANONICAL_ONEX_TOPICS:
            assert " " not in topic and "\t" not in topic, (
                f"Topic '{topic}' contains whitespace. "
                f"ONEX topics must not contain spaces or tabs."
            )


class TestTopicDomainCoverage:
    """Test that topics cover all expected domains."""

    def test_node_lifecycle_topics_present(self) -> None:
        """Verify all node lifecycle topics are present."""
        node_topics = [t for t in CANONICAL_ONEX_TOPICS if t.startswith("onex.node.")]
        expected_node_topics = [
            "onex.node.introspection.published.v1",
            "onex.node.heartbeat.published.v1",
            "onex.node.shutdown.announced.v1",
        ]
        assert len(node_topics) == 3, (
            f"Expected 3 node lifecycle topics, found {len(node_topics)}: {node_topics}"
        )
        for expected in expected_node_topics:
            assert expected in node_topics, (
                f"Missing node lifecycle topic: {expected}"
            )

    def test_registry_topics_present(self) -> None:
        """Verify all registry topics are present."""
        registry_topics = [
            t for t in CANONICAL_ONEX_TOPICS if t.startswith("onex.registry.")
        ]
        expected_count = 7  # 1 control + 3 state change + 3 workflow
        assert len(registry_topics) == expected_count, (
            f"Expected {expected_count} registry topics, "
            f"found {len(registry_topics)}: {registry_topics}"
        )

    def test_infra_topics_present(self) -> None:
        """Verify all infrastructure signal topics are present."""
        infra_topics = [t for t in CANONICAL_ONEX_TOPICS if t.startswith("onex.infra.")]
        expected_infra_topics = [
            "onex.infra.circuit_breaker.state_changed.v1",
            "onex.infra.error.detected.v1",
        ]
        assert len(infra_topics) == 2, (
            f"Expected 2 infrastructure topics, found {len(infra_topics)}: {infra_topics}"
        )
        for expected in expected_infra_topics:
            assert expected in infra_topics, (
                f"Missing infrastructure topic: {expected}"
            )


class TestTopicCategorization:
    """Test topic categorization matches documentation."""

    def test_workflow_topics_grouped_correctly(self) -> None:
        """Verify workflow observability topics are correctly identified."""
        workflow_topics = [
            t for t in CANONICAL_ONEX_TOPICS if "workflow" in t
        ]
        expected_workflow_topics = [
            "onex.registry.workflow.started.v1",
            "onex.registry.workflow.completed.v1",
            "onex.registry.workflow.failed.v1",
        ]
        assert len(workflow_topics) == 3, (
            f"Expected 3 workflow topics, found {len(workflow_topics)}: {workflow_topics}"
        )
        for expected in expected_workflow_topics:
            assert expected in workflow_topics, (
                f"Missing workflow topic: {expected}"
            )

    def test_registration_state_topics_grouped_correctly(self) -> None:
        """Verify registration state change topics are correctly identified."""
        registration_topics = [
            t for t in CANONICAL_ONEX_TOPICS
            if t.startswith("onex.registry.node.")
        ]
        expected_registration_topics = [
            "onex.registry.node.registered.v1",
            "onex.registry.node.registration_failed.v1",
            "onex.registry.node.deregistered.v1",
        ]
        assert len(registration_topics) == 3, (
            f"Expected 3 registration state topics, "
            f"found {len(registration_topics)}: {registration_topics}"
        )
        for expected in expected_registration_topics:
            assert expected in registration_topics, (
                f"Missing registration state topic: {expected}"
            )
