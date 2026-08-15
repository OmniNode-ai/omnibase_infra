# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the ONEX topic format validator."""

from __future__ import annotations

import pytest

from omnibase_infra.utils.util_onex_topic_format import (
    TopicValidationResult,
    validate_onex_topic_format,
)


class TestValidateOnexTopicFormat:
    """Test suite for validate_onex_topic_format()."""

    # ------------------------------------------------------------------
    # Valid canonical ONEX topics
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "topic",
        [
            "onex.evt.omniclaude.session-started.v1",
            "onex.cmd.omniintelligence.code-analysis.v1",
            "onex.intent.omnimemory.crawl-requested.v1",
            "onex.dlq.omniclaude.agent-actions.v1",
            "onex.evt.platform.node-heartbeat.v1",
            "onex.evt.omniclaude.transformation.completed.v1",
            "onex.evt.platform.node-registration.v12",
        ],
    )
    def test_valid_onex_topics(self, topic: str) -> None:
        result, reason = validate_onex_topic_format(topic)
        assert result == TopicValidationResult.VALID
        assert reason == ""

    # ------------------------------------------------------------------
    # Valid legacy DLQ topics
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "topic",
        [
            "onex.dlq.intelligence.v1",
            "local.dlq.intents.v1",
        ],
    )
    def test_valid_legacy_dlq_topics(self, topic: str) -> None:
        result, reason = validate_onex_topic_format(topic)
        assert result == TopicValidationResult.VALID_LEGACY_DLQ
        assert reason == "legacy DLQ format"

    # ------------------------------------------------------------------
    # Invalid topics
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "topic",
        [
            "dev.onex.evt.omniclaude.session-started.v1",  # env prefix
            "onex.evt.omniclaude.session-started",  # missing version
            "agent-actions",  # legacy flat name
            "staging.onex.cmd.omniintelligence.code-analysis.v1",  # env prefix
            "",  # empty
            "onex.evt",  # too few segments
            "onex.unknown.omniclaude.session-started.v1",  # invalid kind
            "onex.evt.omniclaude.session-started.v0",  # v0 not allowed
        ],
    )
    def test_invalid_topics(self, topic: str) -> None:
        result, reason = validate_onex_topic_format(topic)
        assert result == TopicValidationResult.INVALID
        assert reason != ""

    # ------------------------------------------------------------------
    # Kafka internal topics (skipped)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "topic",
        [
            "__consumer_offsets",
            "__transaction_state",
        ],
    )
    def test_kafka_internal_topics_skipped(self, topic: str) -> None:
        result, reason = validate_onex_topic_format(topic)
        assert result == TopicValidationResult.SKIPPED_INTERNAL
        assert reason == ""

    # ------------------------------------------------------------------
    # Tenant wire topics (OMN-15792: delegates to the shared runtime
    # resolver -- service_gateway_topic_transform.resolve_tenant_from_wire_topic
    # -- instead of a private, independently-maintained regex.
    # ------------------------------------------------------------------

    def test_valid_tenant_wire_topic(self) -> None:
        result, reason = validate_onex_topic_format(
            "tenant-acme.onex.evt.omniclaude.session-started.v1"
        )
        assert result == TopicValidationResult.VALID_TENANT_WIRE
        assert reason == ""

    @pytest.mark.parametrize(
        "topic",
        [
            # Reserved slug -- structurally valid shape, but reserved. This is
            # the exact OMN-15757/OMN-15778 divergence: this module previously
            # ALLOWED it (own regex, no RESERVED_TENANT_SLUGS check) while the
            # subscribe-side resolver already rejected it.
            "tenant-system.onex.evt.omniclaude.session-started.v1",
            # Too short to be a valid DNS-compatible slug.
            "tenant-ab.onex.evt.omniclaude.session-started.v1",
            # Uppercase -- not a valid lowercase DNS-compatible slug.
            "tenant-Acme.onex.evt.omniclaude.session-started.v1",
        ],
    )
    def test_invalid_tenant_wire_topics_rejected(self, topic: str) -> None:
        """Publish-enforce must agree with subscribe-resolve on every case.

        Regression for the OMN-15792 corrective verify pass: a tenant-<slug>.
        wire topic with an invalid embedded slug (reserved, malformed, wrong
        case) is INVALID here -- the same verdict
        ``resolve_tenant_from_wire_topic`` (the subscribe-side resolver)
        reaches by raising ``ValueError`` for the identical string.
        """
        result, reason = validate_onex_topic_format(topic)
        assert result == TopicValidationResult.INVALID
        assert reason != ""

        from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
            resolve_tenant_from_wire_topic,
        )

        with pytest.raises(ValueError):
            resolve_tenant_from_wire_topic(topic)
