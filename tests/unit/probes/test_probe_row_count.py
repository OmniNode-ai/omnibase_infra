# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for RowCountProbe (OMN-5653)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.models.health.model_row_count_probe_result import (
    ModelRowCountProbeResult,
)
from omnibase_infra.models.health.model_table_row_count import (
    ModelTableRowCount,
)
from omnibase_infra.probes.probe_row_count import (
    DEFAULT_EXPECTED_EMPTY,
    PgStatRow,
    RowCountProbe,
)


@pytest.mark.unit
class TestRowCountProbe:
    """Tests for the row count probe evaluation logic."""

    def _make_probe(
        self,
        expected_empty: frozenset[str] | None = None,
    ) -> RowCountProbe:
        return RowCountProbe(
            dsn="postgresql://test:test@localhost:5432/test",
            expected_empty=expected_empty,
            db_display_label="test_db",
        )

    def test_evaluate_all_populated(self) -> None:
        """All tables have rows -> healthy."""
        probe = self._make_probe()
        rows = [
            PgStatRow("public", "epic_run_events", 42),
            PgStatRow("public", "pr_watch_state", 10),
        ]
        result = probe._evaluate(rows)

        assert result.healthy is True
        assert result.total_tables == 2
        assert result.empty_tables == 0
        assert result.populated_tables == 2
        assert result.empty_table_names == []
        assert result.db_display_label == "test_db"

    def test_evaluate_with_empty_tables(self) -> None:
        """Some tables have zero rows -> unhealthy, flagged."""
        probe = self._make_probe()
        rows = [
            PgStatRow("public", "epic_run_events", 42),
            PgStatRow("public", "agent_actions", 0),
            PgStatRow("public", "llm_cost_aggregates", 0),
        ]
        result = probe._evaluate(rows)

        assert result.healthy is False
        assert result.total_tables == 3
        assert result.empty_tables == 2
        assert result.populated_tables == 1
        assert "agent_actions" in result.empty_table_names
        assert "llm_cost_aggregates" in result.empty_table_names

    def test_evaluate_expected_empty_excluded(self) -> None:
        """Tables in expected_empty set are not counted as unhealthy."""
        probe = self._make_probe(
            expected_empty=frozenset({"schema_migrations", "staging_table"})
        )
        rows = [
            PgStatRow("public", "epic_run_events", 10),
            PgStatRow("public", "schema_migrations", 0),
            PgStatRow("public", "staging_table", 0),
        ]
        result = probe._evaluate(rows)

        assert result.healthy is True
        assert result.empty_table_names == []

    def test_evaluate_empty_database(self) -> None:
        """No tables at all -> healthy (vacuously)."""
        probe = self._make_probe()
        result = probe._evaluate([])

        assert result.healthy is True
        assert result.total_tables == 0
        assert result.empty_tables == 0

    def test_default_expected_empty_contains_migrations(self) -> None:
        """Default expected_empty includes migration tables."""
        assert "schema_migrations" in DEFAULT_EXPECTED_EMPTY
        assert "drizzle_migrations" in DEFAULT_EXPECTED_EMPTY

    def test_table_details_populated(self) -> None:
        """Table details include per-table row counts."""
        probe = self._make_probe()
        rows = [
            PgStatRow("public", "my_table", 100),
        ]
        result = probe._evaluate(rows)

        assert len(result.table_details) == 1
        detail = result.table_details[0]
        assert detail.relation_key == "my_table"
        assert detail.schema_label == "public"
        assert detail.n_live_tup == 100
        assert detail.is_empty is False

    @pytest.mark.asyncio
    async def test_run_calls_query_and_evaluate(self) -> None:
        """run() queries the database and evaluates results."""
        probe = self._make_probe()
        mock_rows = [
            PgStatRow("public", "test_table", 5),
        ]

        with patch.object(
            probe, "_query_row_counts", new_callable=AsyncMock, return_value=mock_rows
        ):
            result = await probe.run()

        assert isinstance(result, ModelRowCountProbeResult)
        assert result.healthy is True
        assert result.total_tables == 1


@pytest.mark.unit
class TestModelRowCountProbeResult:
    """Tests for the probe result model."""

    def test_model_frozen(self) -> None:
        """Model is immutable."""
        result = ModelRowCountProbeResult(
            db_display_label="test",
            total_tables=1,
            empty_tables=0,
            populated_tables=1,
            healthy=True,
        )
        with pytest.raises(Exception):
            result.db_display_label = "other"  # type: ignore[misc]

    def test_model_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(Exception):
            ModelRowCountProbeResult(
                db_display_label="test",
                total_tables=1,
                empty_tables=0,
                populated_tables=1,
                healthy=True,
                extra_field="bad",  # type: ignore[call-arg]
            )


@pytest.mark.unit
class TestModelTableRowCount:
    """Tests for the table row count model."""

    def test_model_construction(self) -> None:
        """Model constructs with required fields."""
        m = ModelTableRowCount(
            relation_key="my_table",
            n_live_tup=42,
            is_empty=False,
        )
        assert m.relation_key == "my_table"
        assert m.schema_label == "public"  # default
        assert m.n_live_tup == 42
        assert m.is_empty is False
