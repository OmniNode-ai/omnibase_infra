# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for ServicePolicyEngine.

Tests verify that:
    - Topic allowed when in allowlist
    - Topic blocked when not in allowlist
    - Empty allowlist allows all topics
    - Wildcard matching works (e.g., "events.*" matches "events.created")
    - Realm enforcement works correctly

Related Tickets:
    - OMN-1899: Runtime gateway envelope signing
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

from omnibase_infra.gateway import (
    EnumPolicyDecision,
    PolicyDecision,
    ServicePolicyEngine,
)


class TestPolicyEngineTopicAllowlist:
    """Tests for topic allowlist enforcement."""

    def test_topic_allowed_when_in_allowlist(
        self,
        policy_engine: ServicePolicyEngine,
    ) -> None:
        """Topic is allowed when it matches an allowlist pattern."""
        # The fixture has allowed_topics=("events.*", "commands.*")

        # Act
        decision = policy_engine.evaluate_inbound(
            topic="events.order.created",
            realm="test",
        )

        # Assert
        assert decision.decision == EnumPolicyDecision.ALLOW
        assert decision.reason is None
        assert bool(decision) is True

    def test_topic_blocked_when_not_in_allowlist(
        self,
        policy_engine: ServicePolicyEngine,
    ) -> None:
        """Topic is blocked when it doesn't match any allowlist pattern."""
        # The fixture has allowed_topics=("events.*", "commands.*")

        # Act
        decision = policy_engine.evaluate_inbound(
            topic="internal.secret.data",
            realm="test",
        )

        # Assert
        assert decision.decision == EnumPolicyDecision.DENY
        assert decision.reason is not None
        assert "not in allowlist" in decision.reason
        assert bool(decision) is False

    def test_empty_allowlist_allows_all_topics(self) -> None:
        """Empty allowlist allows all topics (open policy)."""
        # Arrange
        open_engine = ServicePolicyEngine(
            allowed_topics=None,  # No allowlist
            expected_realm=None,  # No realm enforcement
            log_rejections=False,
        )

        # Act
        decision1 = open_engine.evaluate_inbound(topic="any.topic.here")
        decision2 = open_engine.evaluate_inbound(topic="internal.secret.data")
        decision3 = open_engine.evaluate_inbound(topic="totally.random.topic")

        # Assert
        assert decision1.decision == EnumPolicyDecision.ALLOW
        assert decision2.decision == EnumPolicyDecision.ALLOW
        assert decision3.decision == EnumPolicyDecision.ALLOW

    def test_empty_list_allowlist_allows_all_topics(self) -> None:
        """Empty list allowlist allows all topics (open policy)."""
        # Arrange
        open_engine = ServicePolicyEngine(
            allowed_topics=[],  # Empty list
            expected_realm=None,
            log_rejections=False,
        )

        # Act
        decision = open_engine.evaluate_inbound(topic="any.topic.here")

        # Assert
        assert decision.decision == EnumPolicyDecision.ALLOW


class TestPolicyEngineWildcardMatching:
    """Tests for wildcard pattern matching."""

    def test_wildcard_star_matches_any_suffix(self) -> None:
        """Wildcard '*' matches any sequence of characters."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["events.*"],
            log_rejections=False,
        )

        # Act & Assert - should all match
        assert engine.is_topic_allowed("events.order") is True
        assert engine.is_topic_allowed("events.order.created") is True
        assert engine.is_topic_allowed("events.user.login.success") is True
        assert engine.is_topic_allowed("events.a") is True

        # Should not match
        assert engine.is_topic_allowed("commands.user") is False
        assert engine.is_topic_allowed("event.order") is False  # No 's'

    def test_wildcard_matches_prefix(self) -> None:
        """Wildcard at end matches any suffix."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["*.created"],
            log_rejections=False,
        )

        # Act & Assert
        assert engine.is_topic_allowed("events.order.created") is True
        assert engine.is_topic_allowed("commands.user.created") is True
        assert engine.is_topic_allowed("something.created") is True

        # Should not match
        assert engine.is_topic_allowed("events.order.updated") is False

    def test_wildcard_in_middle(self) -> None:
        """Wildcard in middle matches any characters."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["events.*.created"],
            log_rejections=False,
        )

        # Act & Assert
        assert engine.is_topic_allowed("events.order.created") is True
        assert engine.is_topic_allowed("events.user.created") is True
        assert engine.is_topic_allowed("events.anything.here.created") is True

        # Should not match
        assert engine.is_topic_allowed("events.order.updated") is False
        assert engine.is_topic_allowed("commands.order.created") is False

    def test_exact_match(self) -> None:
        """Exact topic name matches exactly."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["events.order.created"],
            log_rejections=False,
        )

        # Act & Assert
        assert engine.is_topic_allowed("events.order.created") is True
        assert engine.is_topic_allowed("events.order.created.v1") is False
        assert engine.is_topic_allowed("events.order") is False

    def test_multiple_patterns(self) -> None:
        """Multiple patterns are evaluated, any match allows."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["events.*", "commands.*", "notifications.email.*"],
            log_rejections=False,
        )

        # Act & Assert
        assert engine.is_topic_allowed("events.order.created") is True
        assert engine.is_topic_allowed("commands.user.create") is True
        assert engine.is_topic_allowed("notifications.email.sent") is True

        # Should not match
        assert engine.is_topic_allowed("internal.secret") is False
        assert engine.is_topic_allowed("notifications.sms.sent") is False

    def test_question_mark_matches_single_char(self) -> None:
        """Question mark '?' matches exactly one character."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["events.?"],
            log_rejections=False,
        )

        # Act & Assert
        assert engine.is_topic_allowed("events.a") is True
        assert engine.is_topic_allowed("events.b") is True

        # Should not match (more than one char)
        assert engine.is_topic_allowed("events.ab") is False
        assert engine.is_topic_allowed("events.order") is False

    def test_character_class_matching(self) -> None:
        """Character class [seq] matches any character in sequence."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["events.[ou]rder.*"],
            log_rejections=False,
        )

        # Act & Assert
        assert engine.is_topic_allowed("events.order.created") is True
        assert engine.is_topic_allowed("events.urder.created") is True

        # Should not match
        assert engine.is_topic_allowed("events.arder.created") is False


class TestPolicyEngineRealmEnforcement:
    """Tests for realm boundary enforcement."""

    def test_realm_match_allows_message(
        self,
        policy_engine: ServicePolicyEngine,
    ) -> None:
        """Message with matching realm is allowed (if topic also allowed)."""
        # Act
        decision = policy_engine.evaluate_inbound(
            topic="events.order.created",
            realm="test",  # Matches expected realm
        )

        # Assert
        assert decision.decision == EnumPolicyDecision.ALLOW

    def test_realm_mismatch_blocks_message(
        self,
        policy_engine: ServicePolicyEngine,
    ) -> None:
        """Message with wrong realm is blocked."""
        # Act
        decision = policy_engine.evaluate_inbound(
            topic="events.order.created",  # Allowed topic
            realm="wrong-realm",  # Does not match expected "test"
        )

        # Assert
        assert decision.decision == EnumPolicyDecision.DENY
        assert decision.reason is not None
        assert "Realm mismatch" in decision.reason
        assert "test" in decision.reason
        assert "wrong-realm" in decision.reason

    def test_realm_none_blocks_when_realm_required(
        self,
        policy_engine: ServicePolicyEngine,
    ) -> None:
        """Message with None realm is blocked when realm enforcement is enabled."""
        # Act
        decision = policy_engine.evaluate_inbound(
            topic="events.order.created",
            realm=None,  # No realm provided
        )

        # Assert
        assert decision.decision == EnumPolicyDecision.DENY
        assert decision.reason is not None
        assert "Realm mismatch" in decision.reason

    def test_no_realm_enforcement_when_expected_realm_none(self) -> None:
        """No realm enforcement when expected_realm is None."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["events.*"],
            expected_realm=None,  # No realm enforcement
            log_rejections=False,
        )

        # Act - any realm should be allowed
        decision1 = engine.evaluate_inbound(topic="events.order", realm=None)
        decision2 = engine.evaluate_inbound(topic="events.order", realm="any-realm")
        decision3 = engine.evaluate_inbound(topic="events.order", realm="different")

        # Assert
        assert decision1.decision == EnumPolicyDecision.ALLOW
        assert decision2.decision == EnumPolicyDecision.ALLOW
        assert decision3.decision == EnumPolicyDecision.ALLOW


class TestPolicyEngineCombinedPolicies:
    """Tests for combined topic and realm policies."""

    def test_both_topic_and_realm_must_match(self) -> None:
        """Both topic allowlist AND realm must match for ALLOW."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["events.*"],
            expected_realm="tenant-123",
            log_rejections=False,
        )

        # Act & Assert - both match = ALLOW
        decision = engine.evaluate_inbound(topic="events.order", realm="tenant-123")
        assert decision.decision == EnumPolicyDecision.ALLOW

        # Topic matches, realm doesn't = DENY
        decision = engine.evaluate_inbound(topic="events.order", realm="tenant-456")
        assert decision.decision == EnumPolicyDecision.DENY

        # Realm matches, topic doesn't = DENY
        decision = engine.evaluate_inbound(topic="internal.data", realm="tenant-123")
        assert decision.decision == EnumPolicyDecision.DENY

        # Neither matches = DENY (topic checked first)
        decision = engine.evaluate_inbound(topic="internal.data", realm="tenant-456")
        assert decision.decision == EnumPolicyDecision.DENY

    def test_topic_check_happens_before_realm_check(self) -> None:
        """Topic allowlist is checked before realm enforcement."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["events.*"],
            expected_realm="tenant-123",
            log_rejections=False,
        )

        # Act - blocked topic should fail on topic, not realm
        decision = engine.evaluate_inbound(
            topic="internal.secret",
            realm="wrong-realm",
        )

        # Assert - should fail on topic allowlist
        assert decision.decision == EnumPolicyDecision.DENY
        assert decision.reason is not None
        assert "not in allowlist" in decision.reason


class TestPolicyEngineOutbound:
    """Tests for outbound message policy."""

    def test_outbound_always_allowed(self) -> None:
        """Outbound messages are always allowed (future extension point)."""
        # Arrange
        engine = ServicePolicyEngine(
            allowed_topics=["events.*"],  # Allowlist shouldn't affect outbound
            expected_realm="test",
            log_rejections=False,
        )

        # Act
        decision1 = engine.evaluate_outbound(topic="events.order")
        decision2 = engine.evaluate_outbound(topic="internal.secret")
        decision3 = engine.evaluate_outbound(topic="anything.goes.here")

        # Assert - all outbound is allowed
        assert decision1.decision == EnumPolicyDecision.ALLOW
        assert decision2.decision == EnumPolicyDecision.ALLOW
        assert decision3.decision == EnumPolicyDecision.ALLOW


class TestPolicyEngineProperties:
    """Tests for policy engine property accessors."""

    def test_allowed_topics_property(
        self,
        policy_engine: ServicePolicyEngine,
    ) -> None:
        """Allowed topics property returns configured patterns."""
        topics = policy_engine.allowed_topics
        assert topics is not None
        assert "events.*" in topics
        assert "commands.*" in topics

    def test_expected_realm_property(
        self,
        policy_engine: ServicePolicyEngine,
    ) -> None:
        """Expected realm property returns configured realm."""
        assert policy_engine.expected_realm == "test"

    def test_allowed_topics_is_immutable(
        self,
        policy_engine: ServicePolicyEngine,
    ) -> None:
        """Allowed topics is returned as immutable tuple."""
        topics = policy_engine.allowed_topics
        assert isinstance(topics, tuple)


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""

    def test_policy_decision_is_truthy_when_allow(self) -> None:
        """PolicyDecision is truthy when decision is ALLOW."""
        decision = PolicyDecision(decision=EnumPolicyDecision.ALLOW)
        assert bool(decision) is True

    def test_policy_decision_is_falsy_when_deny(self) -> None:
        """PolicyDecision is falsy when decision is DENY."""
        decision = PolicyDecision(
            decision=EnumPolicyDecision.DENY,
            reason="Blocked",
        )
        assert bool(decision) is False

    def test_policy_decision_frozen(self) -> None:
        """PolicyDecision is frozen (immutable)."""
        decision = PolicyDecision(decision=EnumPolicyDecision.ALLOW)
        with pytest.raises(AttributeError):
            decision.decision = EnumPolicyDecision.DENY  # type: ignore[misc]

    def test_policy_decision_with_reason(self) -> None:
        """PolicyDecision can include a reason."""
        decision = PolicyDecision(
            decision=EnumPolicyDecision.DENY,
            reason="Topic not in allowlist",
        )
        assert decision.reason == "Topic not in allowlist"


class TestEnumPolicyDecision:
    """Tests for EnumPolicyDecision enum."""

    def test_allow_value(self) -> None:
        """ALLOW has correct string value."""
        assert EnumPolicyDecision.ALLOW.value == "allow"

    def test_deny_value(self) -> None:
        """DENY has correct string value."""
        assert EnumPolicyDecision.DENY.value == "deny"

    def test_is_string_enum(self) -> None:
        """EnumPolicyDecision is a string enum."""
        assert isinstance(EnumPolicyDecision.ALLOW, str)
        assert EnumPolicyDecision.ALLOW == "allow"
