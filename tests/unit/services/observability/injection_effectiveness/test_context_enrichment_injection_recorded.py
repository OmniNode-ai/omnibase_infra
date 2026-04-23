# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for context enrichment and injection recorded consumer handlers.

Tests model validation, consumer topic routing, and writer method signatures
for the OMN-6158 pipeline gap fix.

Related Tickets:
    - OMN-6158: Add consumers for context-enrichment and injection-recorded events
    - OMN-2274: Enrichment observability event emission (producer)
    - OMN-1673: INJECT-004 injection tracking event emission (producer)
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.services.observability.injection_effectiveness.config import (
    ConfigInjectionEffectivenessConsumer,
)
from omnibase_infra.services.observability.injection_effectiveness.consumer import (
    TOPIC_TO_MODEL,
    TOPIC_TO_WRITER_METHOD,
)
from omnibase_infra.services.observability.injection_effectiveness.models import (
    ModelContextEnrichmentEvent,
    ModelInjectionRecordedEvent,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENRICHMENT_TOPIC = "onex.evt.omniclaude.context-enrichment.v1"
_INJECTION_RECORDED_TOPIC = "onex.evt.omniclaude.injection-recorded.v1"


# ---------------------------------------------------------------------------
# ModelContextEnrichmentEvent — model validation tests
# ---------------------------------------------------------------------------


class TestModelContextEnrichmentEvent:
    """Unit tests for ModelContextEnrichmentEvent."""

    def _minimal(self, **overrides: object) -> ModelContextEnrichmentEvent:
        """Return a minimal valid event."""
        defaults = {
            "session_id": str(uuid4()),
            "correlation_id": str(uuid4()),
            "timestamp": "2026-04-03T12:00:00+00:00",
            "channel": "summarization",
            "outcome": "hit",
        }
        defaults.update(overrides)
        return ModelContextEnrichmentEvent(**defaults)

    def test_minimal_valid(self) -> None:
        event = self._minimal()
        assert event.channel == "summarization"
        assert event.outcome == "hit"
        assert event.cache_hit is False
        assert event.tokens_before == 0
        assert event.tokens_after == 0
        assert event.net_tokens_saved == 0

    def test_full_payload(self) -> None:
        event = self._minimal(
            model_name="qwen3-coder-14b",
            cache_hit=True,
            latency_ms=42.5,
            tokens_before=500,
            tokens_after=200,
            net_tokens_saved=300,
            similarity_score=0.85,
            repo="omniclaude",
            agent_name="general-purpose",
        )
        assert event.model_name == "qwen3-coder-14b"
        assert event.cache_hit is True
        assert event.latency_ms == 42.5
        assert event.tokens_before == 500
        assert event.tokens_after == 200
        assert event.net_tokens_saved == 300
        assert event.similarity_score == 0.85
        assert event.repo == "omniclaude"
        assert event.agent_name == "general-purpose"

    def test_frozen(self) -> None:
        event = self._minimal()
        with pytest.raises(Exception):
            event.channel = "code_analysis"

    def test_extra_fields_ignored(self) -> None:
        """Extra fields from producer are ignored (extra='ignore')."""
        event = ModelContextEnrichmentEvent(
            session_id=str(uuid4()),
            timestamp="2026-04-03T12:00:00+00:00",
            channel="summarization",
            outcome="hit",
            fallback_used=True,  # internal producer field, not in model
            was_dropped=False,  # internal producer field, not in model
            prompt_version="v2",  # internal producer field, not in model
        )
        assert event.channel == "summarization"

    def test_all_outcome_values(self) -> None:
        for outcome in ("hit", "miss", "error", "inflated"):
            event = self._minimal(outcome=outcome)
            assert event.outcome == outcome

    def test_all_channel_values(self) -> None:
        for channel in ("summarization", "code_analysis", "similarity"):
            event = self._minimal(channel=channel)
            assert event.channel == channel


# ---------------------------------------------------------------------------
# ModelInjectionRecordedEvent — model validation tests
# ---------------------------------------------------------------------------


class TestModelInjectionRecordedEvent:
    """Unit tests for ModelInjectionRecordedEvent."""

    def _minimal(self, **overrides: object) -> ModelInjectionRecordedEvent:
        """Return a minimal valid event."""
        defaults = {
            "session_id": str(uuid4()),
            "correlation_id": str(uuid4()),
            "emitted_at": "2026-04-03T12:00:00+00:00",
        }
        defaults.update(overrides)
        return ModelInjectionRecordedEvent(**defaults)

    def test_minimal_valid(self) -> None:
        event = self._minimal()
        assert event.patterns_injected == 0
        assert event.total_injected_tokens == 0
        assert event.injection_latency_ms == 0.0
        assert event.cache_hit is False

    def test_full_payload(self) -> None:
        event = self._minimal(
            patterns_injected=5,
            total_injected_tokens=1200,
            injection_latency_ms=15.3,
            agent_name="general-purpose",
            repo="omniclaude",
            cache_hit=True,
        )
        assert event.patterns_injected == 5
        assert event.total_injected_tokens == 1200
        assert event.injection_latency_ms == 15.3
        assert event.agent_name == "general-purpose"
        assert event.repo == "omniclaude"
        assert event.cache_hit is True

    def test_frozen(self) -> None:
        event = self._minimal()
        with pytest.raises(Exception):
            event.patterns_injected = 10

    def test_extra_fields_ignored(self) -> None:
        """Extra fields from producer are ignored (extra='ignore')."""
        event = ModelInjectionRecordedEvent(
            session_id=str(uuid4()),
            emitted_at="2026-04-03T12:00:00+00:00",
            unknown_field="should_be_ignored",
        )
        assert event.patterns_injected == 0


# ---------------------------------------------------------------------------
# Consumer wiring tests — topic routing and config
# ---------------------------------------------------------------------------


class TestConsumerWiring:
    """Verify that new topics are wired into consumer routing and config."""

    def test_context_enrichment_topic_in_model_map(self) -> None:
        assert _ENRICHMENT_TOPIC in TOPIC_TO_MODEL
        assert TOPIC_TO_MODEL[_ENRICHMENT_TOPIC] is ModelContextEnrichmentEvent

    def test_injection_recorded_topic_in_model_map(self) -> None:
        assert _INJECTION_RECORDED_TOPIC in TOPIC_TO_MODEL
        assert TOPIC_TO_MODEL[_INJECTION_RECORDED_TOPIC] is ModelInjectionRecordedEvent

    def test_context_enrichment_topic_in_writer_map(self) -> None:
        assert _ENRICHMENT_TOPIC in TOPIC_TO_WRITER_METHOD
        assert TOPIC_TO_WRITER_METHOD[_ENRICHMENT_TOPIC] == "write_context_enrichment"

    def test_injection_recorded_topic_in_writer_map(self) -> None:
        assert _INJECTION_RECORDED_TOPIC in TOPIC_TO_WRITER_METHOD
        assert (
            TOPIC_TO_WRITER_METHOD[_INJECTION_RECORDED_TOPIC]
            == "write_injection_recorded"
        )

    def test_topics_in_default_config(self) -> None:
        """New topics must appear in the default config topic list."""
        config = ConfigInjectionEffectivenessConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:test@localhost:5432/test",
        )
        assert _ENRICHMENT_TOPIC in config.topics
        assert _INJECTION_RECORDED_TOPIC in config.topics
