"""Tests for platform, intelligence, omnimemory, and omniclaude topic suffix constants."""

import pytest

from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.topics import (
    ALL_INTELLIGENCE_TOPIC_SPECS,
    ALL_OMNICLAUDE_TOPIC_SPECS,
    ALL_OMNIMEMORY_TOPIC_SPECS,
    ALL_PLATFORM_SUFFIXES,
    ALL_PROVISIONED_SUFFIXES,
    ALL_PROVISIONED_TOPIC_SPECS,
    SUFFIX_CONTRACT_DEREGISTERED,
    SUFFIX_CONTRACT_REGISTERED,
    SUFFIX_FSM_STATE_TRANSITIONS,
    SUFFIX_INTELLIGENCE_CLAUDE_HOOK_EVENT,
    SUFFIX_INTELLIGENCE_INTENT_CLASSIFIED,
    SUFFIX_INTELLIGENCE_PATTERN_DISCOVERED,
    SUFFIX_INTELLIGENCE_PATTERN_LEARNED,
    SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITION,
    SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITIONED,
    SUFFIX_INTELLIGENCE_PATTERN_PROMOTED,
    SUFFIX_INTELLIGENCE_PATTERN_STORED,
    SUFFIX_INTELLIGENCE_SESSION_OUTCOME,
    SUFFIX_NODE_HEARTBEAT,
    SUFFIX_NODE_INTROSPECTION,
    SUFFIX_NODE_REGISTRATION,
    SUFFIX_NODE_REGISTRATION_ACKED,
    SUFFIX_OMNIMEMORY_CRAWL_REQUESTED,
    SUFFIX_OMNIMEMORY_CRAWL_TICK,
    SUFFIX_OMNIMEMORY_DOCUMENT_CHANGED,
    SUFFIX_OMNIMEMORY_DOCUMENT_DISCOVERED,
    SUFFIX_OMNIMEMORY_DOCUMENT_INDEXED,
    SUFFIX_OMNIMEMORY_DOCUMENT_REMOVED,
    SUFFIX_REGISTRATION_SNAPSHOTS,
    SUFFIX_REGISTRY_REQUEST_INTROSPECTION,
    SUFFIX_REQUEST_INTROSPECTION,
    SUFFIX_RESOLUTION_DECIDED,
    SUFFIX_RUNTIME_TICK,
    SUFFIX_TOPIC_CATALOG_CHANGED,
    SUFFIX_TOPIC_CATALOG_QUERY,
    SUFFIX_TOPIC_CATALOG_RESPONSE,
)

pytestmark = [pytest.mark.unit]


class TestPlatformTopicSuffixes:
    """Tests for platform-reserved topic suffix constants."""

    def test_all_platform_suffixes_are_valid(self) -> None:
        """Every platform suffix constant must pass validation."""
        for suffix in ALL_PLATFORM_SUFFIXES:
            result = validate_topic_suffix(suffix)
            assert result.is_valid, f"Invalid suffix: {suffix} - {result.error}"

    def test_all_platform_suffixes_list_is_complete(self) -> None:
        """ALL_PLATFORM_SUFFIXES should contain all defined platform constants."""
        expected_suffixes = {
            SUFFIX_NODE_REGISTRATION,
            SUFFIX_NODE_INTROSPECTION,
            SUFFIX_NODE_HEARTBEAT,
            SUFFIX_REQUEST_INTROSPECTION,
            SUFFIX_REGISTRY_REQUEST_INTROSPECTION,
            SUFFIX_FSM_STATE_TRANSITIONS,
            SUFFIX_RUNTIME_TICK,
            SUFFIX_REGISTRATION_SNAPSHOTS,
            SUFFIX_CONTRACT_REGISTERED,
            SUFFIX_CONTRACT_DEREGISTERED,
            SUFFIX_NODE_REGISTRATION_ACKED,
            SUFFIX_RESOLUTION_DECIDED,
            SUFFIX_TOPIC_CATALOG_QUERY,
            SUFFIX_TOPIC_CATALOG_RESPONSE,
            SUFFIX_TOPIC_CATALOG_CHANGED,
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
        for suffix in ALL_PLATFORM_SUFFIXES:
            assert isinstance(suffix, str), f"Suffix must be a string: {suffix}"

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

    def test_node_registration_acked_suffix_format(self) -> None:
        """Node registration ACK suffix should have correct format."""
        assert (
            SUFFIX_NODE_REGISTRATION_ACKED
            == "onex.cmd.platform.node-registration-acked.v1"
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

    def test_all_suffix_constants_exported(self) -> None:
        """All SUFFIX_* constants should be exported from topics package."""
        from omnibase_infra import topics
        from omnibase_infra.topics import platform_topic_suffixes

        # Find all SUFFIX_* constants in the module
        suffix_constants = [
            name for name in dir(platform_topic_suffixes) if name.startswith("SUFFIX_")
        ]

        # Verify each is exported from the package
        for name in suffix_constants:
            assert hasattr(topics, name), (
                f"SUFFIX constant '{name}' not exported from omnibase_infra.topics. "
                f"Add it to __all__ in topics/__init__.py"
            )


class TestIntelligenceTopicSuffixes:
    """Tests for intelligence domain topic suffix constants."""

    def test_all_intelligence_suffixes_are_valid(self) -> None:
        """Every intelligence suffix must pass ONEX topic validation."""
        for spec in ALL_INTELLIGENCE_TOPIC_SPECS:
            result = validate_topic_suffix(spec.suffix)
            assert result.is_valid, (
                f"Invalid intelligence suffix: {spec.suffix} - {result.error}"
            )

    def test_intelligence_suffixes_use_correct_producers(self) -> None:
        """Intelligence suffixes should use 'omniintelligence' or 'pattern' as producer."""
        valid_producers = {"omniintelligence", "pattern"}
        for spec in ALL_INTELLIGENCE_TOPIC_SPECS:
            parts = spec.suffix.split(".")
            producer = parts[2]
            assert producer in valid_producers, (
                f"Expected 'omniintelligence' or 'pattern' producer in: {spec.suffix}"
            )

    def test_intelligence_topic_count(self) -> None:
        """Intelligence spec registry should have 12 topics (10 original + 2 decision-recorded added in OMN-2943)."""
        assert len(ALL_INTELLIGENCE_TOPIC_SPECS) == 12

    def test_intelligence_command_topics(self) -> None:
        """Intelligence command topics should be defined."""
        assert (
            SUFFIX_INTELLIGENCE_CLAUDE_HOOK_EVENT
            == "onex.cmd.omniintelligence.claude-hook-event.v1"
        )
        assert (
            SUFFIX_INTELLIGENCE_SESSION_OUTCOME
            == "onex.cmd.omniintelligence.session-outcome.v1"
        )
        assert (
            SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITION
            == "onex.cmd.omniintelligence.pattern-lifecycle-transition.v1"
        )

    def test_intelligence_event_topics(self) -> None:
        """Intelligence event topics should be defined."""
        assert (
            SUFFIX_INTELLIGENCE_INTENT_CLASSIFIED
            == "onex.evt.omniintelligence.intent-classified.v1"
        )
        assert (
            SUFFIX_INTELLIGENCE_PATTERN_LEARNED
            == "onex.evt.omniintelligence.pattern-learned.v1"
        )
        assert (
            SUFFIX_INTELLIGENCE_PATTERN_STORED
            == "onex.evt.omniintelligence.pattern-stored.v1"
        )
        assert (
            SUFFIX_INTELLIGENCE_PATTERN_PROMOTED
            == "onex.evt.omniintelligence.pattern-promoted.v1"
        )
        assert (
            SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITIONED
            == "onex.evt.omniintelligence.pattern-lifecycle-transitioned.v1"
        )
        assert (
            SUFFIX_INTELLIGENCE_PATTERN_DISCOVERED == "onex.evt.pattern.discovered.v1"
        )

    def test_intelligence_topics_use_3_partitions(self) -> None:
        """All intelligence topics should use 3 partitions."""
        for spec in ALL_INTELLIGENCE_TOPIC_SPECS:
            assert spec.partitions == 3, (
                f"Expected 3 partitions for {spec.suffix}, got {spec.partitions}"
            )

    def test_no_duplicate_intelligence_suffixes(self) -> None:
        """Intelligence topic specs should not contain duplicates."""
        suffixes = [spec.suffix for spec in ALL_INTELLIGENCE_TOPIC_SPECS]
        assert len(suffixes) == len(set(suffixes))


class TestOmniMemoryTopicSuffixes:
    """Tests for OmniMemory domain topic suffix constants."""

    def test_all_omnimemory_suffixes_are_valid(self) -> None:
        """Every OmniMemory suffix must pass ONEX topic validation."""
        for spec in ALL_OMNIMEMORY_TOPIC_SPECS:
            result = validate_topic_suffix(spec.suffix)
            assert result.is_valid, (
                f"Invalid OmniMemory suffix: {spec.suffix} - {result.error}"
            )

    def test_omnimemory_suffixes_use_correct_producer(self) -> None:
        """OmniMemory suffixes should use 'omnimemory' as producer."""
        for spec in ALL_OMNIMEMORY_TOPIC_SPECS:
            parts = spec.suffix.split(".")
            producer = parts[2]
            assert producer == "omnimemory", (
                f"Expected 'omnimemory' producer in: {spec.suffix}"
            )

    def test_omnimemory_topic_count(self) -> None:
        """OmniMemory spec registry should have 27 topics (OMN-2941)."""
        assert len(ALL_OMNIMEMORY_TOPIC_SPECS) == 27

    def test_omnimemory_event_topics(self) -> None:
        """OmniMemory event topics should be defined with correct suffixes."""
        assert (
            SUFFIX_OMNIMEMORY_DOCUMENT_DISCOVERED
            == "onex.evt.omnimemory.document-discovered.v1"
        )
        assert (
            SUFFIX_OMNIMEMORY_DOCUMENT_CHANGED
            == "onex.evt.omnimemory.document-changed.v1"
        )
        assert (
            SUFFIX_OMNIMEMORY_DOCUMENT_REMOVED
            == "onex.evt.omnimemory.document-removed.v1"
        )
        assert (
            SUFFIX_OMNIMEMORY_DOCUMENT_INDEXED
            == "onex.evt.omnimemory.document-indexed.v1"
        )

    def test_omnimemory_command_topics(self) -> None:
        """OmniMemory command topics should be defined with correct suffixes."""
        assert SUFFIX_OMNIMEMORY_CRAWL_TICK == "onex.cmd.omnimemory.crawl-tick.v1"
        assert (
            SUFFIX_OMNIMEMORY_CRAWL_REQUESTED
            == "onex.cmd.omnimemory.crawl-requested.v1"
        )

    def test_omnimemory_topics_use_3_partitions(self) -> None:
        """All OmniMemory topics should use 3 partitions."""
        for spec in ALL_OMNIMEMORY_TOPIC_SPECS:
            assert spec.partitions == 3, (
                f"Expected 3 partitions for {spec.suffix}, got {spec.partitions}"
            )

    def test_no_duplicate_omnimemory_suffixes(self) -> None:
        """OmniMemory topic specs should not contain duplicates."""
        suffixes = [spec.suffix for spec in ALL_OMNIMEMORY_TOPIC_SPECS]
        assert len(suffixes) == len(set(suffixes))


class TestProvisionedTopicSpecs:
    """Tests for the combined provisioned topic spec registry."""

    def test_provisioned_contains_all_platform(self) -> None:
        """ALL_PROVISIONED_SUFFIXES must include all platform suffixes."""
        for suffix in ALL_PLATFORM_SUFFIXES:
            assert suffix in ALL_PROVISIONED_SUFFIXES, (
                f"Platform suffix missing from provisioned: {suffix}"
            )

    def test_provisioned_contains_all_intelligence(self) -> None:
        """ALL_PROVISIONED_SUFFIXES must include all intelligence suffixes."""
        intelligence_suffixes = {spec.suffix for spec in ALL_INTELLIGENCE_TOPIC_SPECS}
        for suffix in intelligence_suffixes:
            assert suffix in ALL_PROVISIONED_SUFFIXES, (
                f"Intelligence suffix missing from provisioned: {suffix}"
            )

    def test_provisioned_contains_all_omnimemory(self) -> None:
        """ALL_PROVISIONED_SUFFIXES must include all OmniMemory suffixes."""
        omnimemory_suffixes = {spec.suffix for spec in ALL_OMNIMEMORY_TOPIC_SPECS}
        for suffix in omnimemory_suffixes:
            assert suffix in ALL_PROVISIONED_SUFFIXES, (
                f"OmniMemory suffix missing from provisioned: {suffix}"
            )

    def test_provisioned_count(self) -> None:
        """Combined provisioned specs should equal platform + intelligence + omnimemory + omniclaude + omnibase_infra."""
        from omnibase_infra.topics import (
            ALL_OMNIBASE_INFRA_TOPIC_SPECS,
            ALL_PLATFORM_TOPIC_SPECS,
        )

        expected = (
            len(ALL_PLATFORM_TOPIC_SPECS)
            + len(ALL_INTELLIGENCE_TOPIC_SPECS)
            + len(ALL_OMNIMEMORY_TOPIC_SPECS)
            + len(ALL_OMNIBASE_INFRA_TOPIC_SPECS)
            + len(ALL_OMNICLAUDE_TOPIC_SPECS)
        )
        assert len(ALL_PROVISIONED_TOPIC_SPECS) == expected

    def test_no_duplicate_provisioned_suffixes(self) -> None:
        """Combined provisioned specs should not contain duplicates."""
        assert len(ALL_PROVISIONED_SUFFIXES) == len(set(ALL_PROVISIONED_SUFFIXES))

    def test_all_provisioned_suffixes_are_valid(self) -> None:
        """Every provisioned suffix must pass ONEX topic validation."""
        for suffix in ALL_PROVISIONED_SUFFIXES:
            result = validate_topic_suffix(suffix)
            assert result.is_valid, f"Invalid suffix: {suffix} - {result.error}"

    def test_provisioned_contains_all_omniclaude(self) -> None:
        """ALL_PROVISIONED_SUFFIXES must include all OmniClaude skill suffixes."""
        omniclaude_suffixes = {spec.suffix for spec in ALL_OMNICLAUDE_TOPIC_SPECS}
        for suffix in omniclaude_suffixes:
            assert suffix in ALL_PROVISIONED_SUFFIXES, (
                f"OmniClaude suffix missing from provisioned: {suffix}"
            )

    def test_provisioned_is_tuple(self) -> None:
        """ALL_PROVISIONED_TOPIC_SPECS and ALL_PROVISIONED_SUFFIXES should be tuples."""
        assert isinstance(ALL_PROVISIONED_TOPIC_SPECS, tuple)
        assert isinstance(ALL_PROVISIONED_SUFFIXES, tuple)


class TestOmniClaudeTopicSuffixes:
    """Tests for OmniClaude skill topic suffix constants."""

    def test_all_omniclaude_suffixes_are_valid(self) -> None:
        """Every OmniClaude suffix must pass ONEX topic validation."""
        for spec in ALL_OMNICLAUDE_TOPIC_SPECS:
            result = validate_topic_suffix(spec.suffix)
            assert result.is_valid, (
                f"Invalid OmniClaude suffix: {spec.suffix} - {result.error}"
            )

    def test_omniclaude_suffixes_use_correct_producer(self) -> None:
        """OmniClaude suffixes should use 'omniclaude' as producer."""
        for spec in ALL_OMNICLAUDE_TOPIC_SPECS:
            parts = spec.suffix.split(".")
            producer = parts[2]
            assert producer == "omniclaude", (
                f"Expected 'omniclaude' producer in: {spec.suffix}"
            )

    def test_omniclaude_topic_count(self) -> None:
        """OmniClaude spec registry should have 207 topics (68 skills x 3 each + 2 lifecycle topics [OMN-2934] + 1 DLQ [OMN-2945])."""
        assert len(ALL_OMNICLAUDE_TOPIC_SPECS) == 207

    def test_omniclaude_skill_topics_use_1_partition(self) -> None:
        """Skill dispatch topics should use 1 partition; DLQ topics use 3 partitions."""
        from omnibase_infra.topics import SUFFIX_OMNICLAUDE_AGENT_ACTIONS_DLQ

        dlq_suffixes = {SUFFIX_OMNICLAUDE_AGENT_ACTIONS_DLQ}
        for spec in ALL_OMNICLAUDE_TOPIC_SPECS:
            if spec.suffix in dlq_suffixes:
                assert spec.partitions == 3, (
                    f"Expected 3 partitions for DLQ topic {spec.suffix}, got {spec.partitions}"
                )
            else:
                assert spec.partitions == 1, (
                    f"Expected 1 partition for skill topic {spec.suffix}, got {spec.partitions}"
                )

    def test_no_duplicate_omniclaude_suffixes(self) -> None:
        """OmniClaude topic specs should not contain duplicates."""
        suffixes = [spec.suffix for spec in ALL_OMNICLAUDE_TOPIC_SPECS]
        assert len(suffixes) == len(set(suffixes))

    def test_omniclaude_has_cmd_and_evt_topics(self) -> None:
        """OmniClaude topics should include both cmd and evt kinds."""
        cmd_topics = [s for s in ALL_OMNICLAUDE_TOPIC_SPECS if ".cmd." in s.suffix]
        evt_topics = [s for s in ALL_OMNICLAUDE_TOPIC_SPECS if ".evt." in s.suffix]
        assert len(cmd_topics) > 0, "Expected cmd topics in OmniClaude registry"
        assert len(evt_topics) > 0, "Expected evt topics in OmniClaude registry"
        # 68 cmd topics + 139 evt topics:
        #   68 completed + 68 failed (skill) + 2 lifecycle [OMN-2934] + 1 DLQ [OMN-2945] = 207
        assert len(cmd_topics) == 68
        assert len(evt_topics) == 139

    def test_epic_team_topic_in_registry(self) -> None:
        """Spot check: epic-team skill topics should be in the registry."""
        suffixes = {spec.suffix for spec in ALL_OMNICLAUDE_TOPIC_SPECS}
        assert "onex.cmd.omniclaude.epic-team.v1" in suffixes
        assert "onex.evt.omniclaude.epic-team-completed.v1" in suffixes
        assert "onex.evt.omniclaude.epic-team-failed.v1" in suffixes

    def test_ticket_work_topic_in_registry(self) -> None:
        """Spot check: ticket-work skill topics should be in the registry."""
        suffixes = {spec.suffix for spec in ALL_OMNICLAUDE_TOPIC_SPECS}
        assert "onex.cmd.omniclaude.ticket-work.v1" in suffixes
        assert "onex.evt.omniclaude.ticket-work-completed.v1" in suffixes
        assert "onex.evt.omniclaude.ticket-work-failed.v1" in suffixes

    def test_create_ticket_topic_in_registry(self) -> None:
        """Spot check: create-ticket skill topics should be in the registry."""
        suffixes = {spec.suffix for spec in ALL_OMNICLAUDE_TOPIC_SPECS}
        assert "onex.cmd.omniclaude.create-ticket.v1" in suffixes
        assert "onex.evt.omniclaude.create-ticket-completed.v1" in suffixes
        assert "onex.evt.omniclaude.create-ticket-failed.v1" in suffixes

    def test_ticket_pipeline_topic_in_registry(self) -> None:
        """Spot check: ticket-pipeline skill topics should be in the registry."""
        suffixes = {spec.suffix for spec in ALL_OMNICLAUDE_TOPIC_SPECS}
        assert "onex.cmd.omniclaude.ticket-pipeline.v1" in suffixes
        assert "onex.evt.omniclaude.ticket-pipeline-completed.v1" in suffixes
        assert "onex.evt.omniclaude.ticket-pipeline-failed.v1" in suffixes

    def test_local_review_topic_in_registry(self) -> None:
        """Spot check: local-review skill topics should be in the registry."""
        suffixes = {spec.suffix for spec in ALL_OMNICLAUDE_TOPIC_SPECS}
        assert "onex.cmd.omniclaude.local-review.v1" in suffixes
        assert "onex.evt.omniclaude.local-review-completed.v1" in suffixes
        assert "onex.evt.omniclaude.local-review-failed.v1" in suffixes

    def test_omniclaude_is_tuple(self) -> None:
        """ALL_OMNICLAUDE_TOPIC_SPECS should be an immutable tuple."""
        assert isinstance(ALL_OMNICLAUDE_TOPIC_SPECS, tuple)
