# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for manifest injection lifecycle event model and writer method.

Tests model validation, consumer topic routing, and writer SQL generation
for the OMN-1888 audit trail (OMN-2942).

Related Tickets:
    - OMN-2942: Add consumer for manifest injection lifecycle events
    - OMN-1888: Manifest injection effectiveness measurement loop
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
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
    ModelManifestInjectionLifecycleEvent,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MANIFEST_TOPICS = [
    "onex.evt.omniclaude.manifest-injection-started.v1",
    "onex.evt.omniclaude.manifest-injected.v1",
    "onex.evt.omniclaude.manifest-injection-failed.v1",
]

_EVENT_TYPES = [
    "manifest_injection_started",
    "manifest_injected",
    "manifest_injection_failed",
]


# ---------------------------------------------------------------------------
# ModelManifestInjectionLifecycleEvent — model validation tests
# ---------------------------------------------------------------------------


class TestModelManifestInjectionLifecycleEvent:
    """Unit tests for ModelManifestInjectionLifecycleEvent."""

    def _minimal(
        self,
        event_type: str = "manifest_injected",
        injection_success: bool | None = True,
        injection_duration_ms: int | None = 42,
    ) -> ModelManifestInjectionLifecycleEvent:
        """Return a minimal valid event."""
        return ModelManifestInjectionLifecycleEvent(
            event_type=event_type,
            entity_id=uuid4(),
            session_id=uuid4(),
            correlation_id=uuid4(),
            causation_id=uuid4(),
            emitted_at=datetime(2026, 2, 27, 12, 0, 0, tzinfo=UTC),
            agent_label="agent-api-architect",
            agent_domain="api-development",
            injection_success=injection_success,
            injection_duration_ms=injection_duration_ms,
        )

    @pytest.mark.parametrize("event_type", _EVENT_TYPES)
    def test_all_lifecycle_stages_accepted(self, event_type: str) -> None:
        """All three lifecycle stage discriminators are valid."""
        event = self._minimal(event_type=event_type)
        assert event.event_type == event_type

    def test_started_event_no_outcome(self) -> None:
        """manifest_injection_started events may omit injection_success and duration."""
        event = self._minimal(
            event_type="manifest_injection_started",
            injection_success=None,
            injection_duration_ms=None,
        )
        assert event.injection_success is None
        assert event.injection_duration_ms is None

    def test_injected_event_success_true(self) -> None:
        """manifest_injected events carry injection_success=True."""
        event = self._minimal(
            event_type="manifest_injected",
            injection_success=True,
            injection_duration_ms=45,
        )
        assert event.injection_success is True
        assert event.injection_duration_ms == 45

    def test_failed_event_success_false(self) -> None:
        """manifest_injection_failed events carry injection_success=False and error fields."""
        event = ModelManifestInjectionLifecycleEvent(
            event_type="manifest_injection_failed",
            entity_id=uuid4(),
            session_id=uuid4(),
            correlation_id=uuid4(),
            causation_id=uuid4(),
            emitted_at=datetime(2026, 2, 27, 12, 0, 0, tzinfo=UTC),
            agent_label="agent-api-architect",
            injection_success=False,
            injection_duration_ms=None,
            error_message="YAML not found",
            error_type="FileNotFoundError",
        )
        assert event.injection_success is False
        assert event.error_message == "YAML not found"
        assert event.error_type == "FileNotFoundError"

    def test_frozen_immutable(self) -> None:
        """Event is frozen — attribute assignment raises ValidationError."""
        import pydantic

        event = self._minimal()
        with pytest.raises(pydantic.ValidationError):
            event.agent_label = "other-agent"  # type: ignore[misc]

    def test_invalid_event_type_rejected(self) -> None:
        """Unknown event_type discriminators are rejected by validation."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            ModelManifestInjectionLifecycleEvent(
                event_type="unknown_stage",  # type: ignore[arg-type]
                entity_id=uuid4(),
                session_id=uuid4(),
                correlation_id=uuid4(),
                causation_id=uuid4(),
                emitted_at=datetime(2026, 2, 27, 12, 0, 0, tzinfo=UTC),
                agent_label="agent-api-architect",
            )

    def test_extra_fields_ignored(self) -> None:
        """Extra fields are ignored (extra='ignore') for forward compatibility."""
        event = ModelManifestInjectionLifecycleEvent(
            event_type="manifest_injected",
            entity_id=uuid4(),
            session_id=uuid4(),
            correlation_id=uuid4(),
            causation_id=uuid4(),
            emitted_at=datetime(2026, 2, 27, 12, 0, 0, tzinfo=UTC),
            agent_label="agent-api-architect",
            injection_success=True,
            injection_duration_ms=20,
            unknown_future_field="ignored",  # type: ignore[call-arg]
        )
        assert event.agent_label == "agent-api-architect"

    def test_default_agent_domain(self) -> None:
        """agent_domain defaults to empty string when not supplied."""
        event = ModelManifestInjectionLifecycleEvent(
            event_type="manifest_injected",
            entity_id=uuid4(),
            session_id=uuid4(),
            correlation_id=uuid4(),
            causation_id=uuid4(),
            emitted_at=datetime(2026, 2, 27, 12, 0, 0, tzinfo=UTC),
            agent_label="agent-api-architect",
            injection_success=True,
            injection_duration_ms=20,
        )
        assert event.agent_domain == ""

    def test_optional_metadata_defaults_none(self) -> None:
        """Optional fields default to None."""
        event = self._minimal()
        assert event.routing_source is None
        assert event.agent_version is None
        assert event.yaml_path is None
        assert event.error_message is None
        assert event.error_type is None


# ---------------------------------------------------------------------------
# Consumer routing — TOPIC_TO_MODEL and TOPIC_TO_WRITER_METHOD
# ---------------------------------------------------------------------------


class TestConsumerTopicRouting:
    """Verify the 3 manifest injection topics are wired into the consumer."""

    @pytest.mark.parametrize("topic", _MANIFEST_TOPICS)
    def test_topic_has_model_mapping(self, topic: str) -> None:
        """Each manifest injection topic maps to ModelManifestInjectionLifecycleEvent."""
        assert topic in TOPIC_TO_MODEL
        assert TOPIC_TO_MODEL[topic] is ModelManifestInjectionLifecycleEvent

    @pytest.mark.parametrize("topic", _MANIFEST_TOPICS)
    def test_topic_has_writer_method_mapping(self, topic: str) -> None:
        """Each manifest injection topic maps to write_manifest_injection_lifecycle."""
        assert topic in TOPIC_TO_WRITER_METHOD
        assert TOPIC_TO_WRITER_METHOD[topic] == "write_manifest_injection_lifecycle"

    def test_existing_topics_unchanged(self) -> None:
        """Adding new topics does not alter existing topic routing."""
        assert (
            TOPIC_TO_MODEL["onex.evt.omniclaude.context-utilization.v1"].__name__
            == "ModelContextUtilizationEvent"
        )
        assert (
            TOPIC_TO_MODEL["onex.evt.omniclaude.agent-match.v1"].__name__
            == "ModelAgentMatchEvent"
        )
        assert (
            TOPIC_TO_MODEL["onex.evt.omniclaude.latency-breakdown.v1"].__name__
            == "ModelLatencyBreakdownEvent"
        )
        assert (
            TOPIC_TO_WRITER_METHOD["onex.evt.omniclaude.context-utilization.v1"]
            == "write_context_utilization"
        )
        assert (
            TOPIC_TO_WRITER_METHOD["onex.evt.omniclaude.agent-match.v1"]
            == "write_agent_match"
        )
        assert (
            TOPIC_TO_WRITER_METHOD["onex.evt.omniclaude.latency-breakdown.v1"]
            == "write_latency_breakdowns"
        )


# ---------------------------------------------------------------------------
# Config — default topics include manifest lifecycle topics
# ---------------------------------------------------------------------------


class TestConfigManifestTopics:
    """Verify config includes manifest injection lifecycle topics by default."""

    def test_default_topics_include_manifest_injection(self) -> None:
        """All 3 manifest injection lifecycle topics are in default topic list."""
        config = ConfigInjectionEffectivenessConsumer(
            postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
        )
        for topic in _MANIFEST_TOPICS:
            assert topic in config.topics, f"Missing topic: {topic}"

    def test_total_topic_count(self) -> None:
        """Default config has 6 topics (3 existing + 3 manifest lifecycle)."""
        config = ConfigInjectionEffectivenessConsumer(
            postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
        )
        assert len(config.topics) == 6


# ---------------------------------------------------------------------------
# WriterInjectionEffectivenessPostgres — write_manifest_injection_lifecycle
# ---------------------------------------------------------------------------


class TestWriterManifestInjectionLifecycle:
    """Unit tests for write_manifest_injection_lifecycle (mocked asyncpg pool)."""

    def _make_events(
        self, count: int = 2
    ) -> list[ModelManifestInjectionLifecycleEvent]:
        """Generate a list of test events."""
        return [
            ModelManifestInjectionLifecycleEvent(
                event_type="manifest_injected",
                entity_id=uuid4(),
                session_id=uuid4(),
                correlation_id=uuid4(),
                causation_id=uuid4(),
                emitted_at=datetime(2026, 2, 27, 12, 0, 0, tzinfo=UTC),
                agent_label="agent-api-architect",
                injection_success=True,
                injection_duration_ms=40 + i,
            )
            for i in range(count)
        ]

    @pytest.mark.asyncio
    async def test_empty_batch_returns_zero(self) -> None:
        """Writing an empty batch returns 0 without touching the pool."""
        from omnibase_infra.services.observability.injection_effectiveness.writer_postgres import (
            WriterInjectionEffectivenessPostgres,
        )

        pool = MagicMock()
        writer = WriterInjectionEffectivenessPostgres(pool)
        result = await writer.write_manifest_injection_lifecycle(
            [], correlation_id=uuid4()
        )
        assert result == 0
        pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_calls_executemany(self) -> None:
        """write_manifest_injection_lifecycle calls executemany with correct row count."""
        from omnibase_infra.services.observability.injection_effectiveness.writer_postgres import (
            WriterInjectionEffectivenessPostgres,
        )

        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        writer = WriterInjectionEffectivenessPostgres(mock_pool)
        events = self._make_events(3)
        correlation_id = uuid4()

        with (
            patch(
                "omnibase_infra.services.observability.injection_effectiveness.writer_postgres.set_statement_timeout",
                new_callable=AsyncMock,
            ),
            patch(
                "omnibase_infra.services.observability.injection_effectiveness.writer_postgres.db_operation_error_context",
            ) as mock_ctx,
        ):
            mock_ctx.return_value.__aenter__ = AsyncMock()
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await writer.write_manifest_injection_lifecycle(
                events, correlation_id=correlation_id
            )

        assert result == 3
        mock_conn.executemany.assert_called_once()
        # Verify the SQL contains the table name
        call_args = mock_conn.executemany.call_args
        assert "manifest_injection_lifecycle" in call_args[0][0]
        # Verify the rows list has 3 entries
        rows = call_args[0][1]
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_write_returns_event_count(self) -> None:
        """Return value equals number of events passed (not affected rows)."""
        from omnibase_infra.services.observability.injection_effectiveness.writer_postgres import (
            WriterInjectionEffectivenessPostgres,
        )

        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        writer = WriterInjectionEffectivenessPostgres(mock_pool)
        events = self._make_events(5)

        with (
            patch(
                "omnibase_infra.services.observability.injection_effectiveness.writer_postgres.set_statement_timeout",
                new_callable=AsyncMock,
            ),
            patch(
                "omnibase_infra.services.observability.injection_effectiveness.writer_postgres.db_operation_error_context",
            ) as mock_ctx,
        ):
            mock_ctx.return_value.__aenter__ = AsyncMock()
            mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await writer.write_manifest_injection_lifecycle(
                events, correlation_id=uuid4()
            )

        assert result == 5
