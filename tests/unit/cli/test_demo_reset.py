# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for demo reset engine and CLI command.

Tests cover:
- DemoResetEngine with dry-run and live execution
- Scoped resource filtering (only demo resources affected)
- Shared infrastructure preservation
- Idempotent behavior
- CLI command integration (Click runner)
- Error handling for missing infrastructure

Related:
    - OMN-2299: Demo Reset scoped command for safe environment reset
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.demo_reset import (
    DEMO_CONSUMER_GROUP_PATTERN,
    DEMO_PROJECTION_TABLE,
    DEMO_TOPIC_PREFIXES,
    PRESERVED_RESOURCES,
    DemoResetConfig,
    DemoResetEngine,
    DemoResetReport,
    EnumResetAction,
    ResetActionResult,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# Config Tests
# =============================================================================


class TestDemoResetConfig:
    """Tests for DemoResetConfig construction."""

    def test_default_config(self) -> None:
        """Default config has expected defaults."""
        config = DemoResetConfig()
        assert config.postgres_dsn == ""
        assert config.kafka_bootstrap_servers == ""
        assert config.purge_topics is False
        assert config.projection_table == DEMO_PROJECTION_TABLE

    def test_from_env_reads_environment(self) -> None:
        """from_env reads OMNIBASE_INFRA_DB_URL and KAFKA_BOOTSTRAP_SERVERS."""
        with patch.dict(
            "os.environ",
            {
                "OMNIBASE_INFRA_DB_URL": "postgresql://localhost/test",
                "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            },
        ):
            config = DemoResetConfig.from_env(purge_topics=True)
            assert config.postgres_dsn == "postgresql://localhost/test"
            assert config.kafka_bootstrap_servers == "localhost:9092"
            assert config.purge_topics is True

    def test_from_env_missing_vars(self) -> None:
        """from_env uses empty strings when env vars are missing."""
        with patch.dict("os.environ", {}, clear=True):
            config = DemoResetConfig.from_env()
            assert config.postgres_dsn == ""
            assert config.kafka_bootstrap_servers == ""


# =============================================================================
# Consumer Group Pattern Tests
# =============================================================================


class TestConsumerGroupPattern:
    """Tests for demo consumer group pattern matching."""

    def test_matches_registration_groups(self) -> None:
        """Pattern matches groups containing 'registration'."""
        assert DEMO_CONSUMER_GROUP_PATTERN.search(
            "dev.omnibase.registration_orchestrator.consume.v1"
        )

    def test_matches_projector_groups(self) -> None:
        """Pattern matches groups containing 'projector'."""
        assert DEMO_CONSUMER_GROUP_PATTERN.search(
            "dev.omnibase.projector_shell.consume.v1"
        )

    def test_matches_introspection_groups(self) -> None:
        """Pattern matches groups containing 'introspection'."""
        assert DEMO_CONSUMER_GROUP_PATTERN.search(
            "dev.omnibase.introspection_consumer.consume.v1"
        )

    def test_no_match_unrelated_groups(self) -> None:
        """Pattern does not match unrelated consumer groups."""
        assert not DEMO_CONSUMER_GROUP_PATTERN.search(
            "dev.omniintelligence.pattern_feedback.consume.v1"
        )

    def test_case_insensitive(self) -> None:
        """Pattern is case-insensitive."""
        assert DEMO_CONSUMER_GROUP_PATTERN.search("dev.REGISTRATION.orchestrator.v1")


# =============================================================================
# Topic Prefix Tests
# =============================================================================


class TestDemoTopicPrefixes:
    """Tests for demo topic prefix classification."""

    @pytest.mark.parametrize(
        "topic",
        [
            "onex.evt.platform.node-registration.v1",
            "onex.cmd.platform.node-introspection.v1",
            "onex.evt.omniintelligence.session-outcome.v1",
            "onex.cmd.omniintelligence.pattern-lifecycle.v1",
            "onex.evt.omniclaude.phase-metrics.v1",
            "onex.evt.agent.status.v1",
        ],
    )
    def test_demo_topics_match_prefixes(self, topic: str) -> None:
        """Demo topics match at least one prefix."""
        assert any(topic.startswith(p) for p in DEMO_TOPIC_PREFIXES)

    @pytest.mark.parametrize(
        "topic",
        [
            "__consumer_offsets",
            "_schemas",
            "custom.business.events.v1",
        ],
    )
    def test_non_demo_topics_do_not_match(self, topic: str) -> None:
        """Non-demo topics do not match any prefix."""
        assert not any(topic.startswith(p) for p in DEMO_TOPIC_PREFIXES)


# =============================================================================
# Report Tests
# =============================================================================


class TestDemoResetReport:
    """Tests for DemoResetReport formatting and properties."""

    def test_empty_report(self) -> None:
        """Empty report has zero counts."""
        report = DemoResetReport()
        assert report.reset_count == 0
        assert report.preserved_count == 0
        assert report.error_count == 0
        assert report.skipped_count == 0

    def test_counts_by_action_type(self) -> None:
        """Report counts actions by type correctly."""
        report = DemoResetReport(
            actions=[
                ResetActionResult("a", EnumResetAction.RESET, "done"),
                ResetActionResult("b", EnumResetAction.RESET, "done"),
                ResetActionResult("c", EnumResetAction.PRESERVED, "kept"),
                ResetActionResult("d", EnumResetAction.SKIPPED, "na"),
                ResetActionResult("e", EnumResetAction.ERROR, "fail"),
            ]
        )
        assert report.reset_count == 2
        assert report.preserved_count == 1
        assert report.skipped_count == 1
        assert report.error_count == 1

    def test_format_summary_dry_run(self) -> None:
        """Summary indicates dry run mode."""
        report = DemoResetReport(dry_run=True)
        summary = report.format_summary()
        assert "DRY RUN" in summary

    def test_format_summary_executed(self) -> None:
        """Summary indicates execution mode."""
        report = DemoResetReport(dry_run=False)
        summary = report.format_summary()
        assert "EXECUTED" in summary

    def test_format_summary_includes_actions(self) -> None:
        """Summary includes action details."""
        report = DemoResetReport(
            actions=[
                ResetActionResult(
                    "Projector state",
                    EnumResetAction.RESET,
                    "Deleted 5 rows",
                ),
            ]
        )
        summary = report.format_summary()
        assert "Projector state" in summary
        assert "Deleted 5 rows" in summary


# =============================================================================
# Engine Tests -- Projector State
# =============================================================================


class TestDemoResetEngineProjectorState:
    """Tests for projector state reset operations."""

    @pytest.mark.asyncio
    async def test_skip_when_no_postgres_dsn(self) -> None:
        """Projector reset skips when no DSN configured."""
        config = DemoResetConfig(postgres_dsn="")
        engine = DemoResetEngine(config)
        report = await engine.execute(dry_run=True)

        projector_actions = [a for a in report.actions if "Projector" in a.resource]
        assert len(projector_actions) == 1
        assert projector_actions[0].action == EnumResetAction.SKIPPED
        assert "not configured" in projector_actions[0].detail

    @pytest.mark.asyncio
    async def test_dry_run_counts_rows(self) -> None:
        """Dry run reports row count without deleting."""
        config = DemoResetConfig(postgres_dsn="postgresql://localhost/test")
        engine = DemoResetEngine(config)

        with patch.object(engine, "_count_projection_rows", return_value=42):
            report = await engine.execute(dry_run=True)

        projector_actions = [a for a in report.actions if "Projector" in a.resource]
        assert len(projector_actions) == 1
        assert projector_actions[0].action == EnumResetAction.RESET
        assert "42" in projector_actions[0].detail
        assert "Would delete" in projector_actions[0].detail

    @pytest.mark.asyncio
    async def test_live_deletes_rows(self) -> None:
        """Live execution deletes rows and reports count."""
        config = DemoResetConfig(postgres_dsn="postgresql://localhost/test")
        engine = DemoResetEngine(config)

        with patch.object(engine, "_delete_projection_rows", return_value=10):
            report = await engine.execute(dry_run=False)

        projector_actions = [a for a in report.actions if "Projector" in a.resource]
        assert len(projector_actions) == 1
        assert projector_actions[0].action == EnumResetAction.RESET
        assert "10" in projector_actions[0].detail
        assert "Deleted" in projector_actions[0].detail

    @pytest.mark.asyncio
    async def test_error_handling_on_db_failure(self) -> None:
        """Database errors are caught and reported."""
        config = DemoResetConfig(postgres_dsn="postgresql://localhost/test")
        engine = DemoResetEngine(config)

        with patch.object(
            engine,
            "_delete_projection_rows",
            side_effect=ConnectionError("connection refused"),
        ):
            report = await engine.execute(dry_run=False)

        projector_actions = [a for a in report.actions if "Projector" in a.resource]
        assert any(a.action == EnumResetAction.ERROR for a in projector_actions)


# =============================================================================
# Engine Tests -- Consumer Groups
# =============================================================================


class TestDemoResetEngineConsumerGroups:
    """Tests for consumer group reset operations."""

    @pytest.mark.asyncio
    async def test_skip_when_no_kafka_configured(self) -> None:
        """Consumer group reset skips when Kafka not configured."""
        config = DemoResetConfig(kafka_bootstrap_servers="")
        engine = DemoResetEngine(config)
        report = await engine.execute(dry_run=True)

        cg_actions = [a for a in report.actions if "Consumer group" in a.resource]
        assert len(cg_actions) == 1
        assert cg_actions[0].action == EnumResetAction.SKIPPED


# =============================================================================
# Engine Tests -- Topic Purge
# =============================================================================


class TestDemoResetEngineTopicPurge:
    """Tests for topic purge operations."""

    @pytest.mark.asyncio
    async def test_skip_when_not_requested(self) -> None:
        """Topic purge skips when purge_topics is False."""
        config = DemoResetConfig(purge_topics=False)
        engine = DemoResetEngine(config)
        report = await engine.execute(dry_run=True)

        topic_actions = [a for a in report.actions if "topic" in a.resource.lower()]
        assert any(a.action == EnumResetAction.SKIPPED for a in topic_actions)
        assert any("not requested" in a.detail for a in topic_actions)


# =============================================================================
# Engine Tests -- Preserved Resources
# =============================================================================


class TestDemoResetPreservation:
    """Tests for explicit resource preservation."""

    @pytest.mark.asyncio
    async def test_preserved_resources_listed(self) -> None:
        """All preserved resources appear in the report."""
        config = DemoResetConfig()
        engine = DemoResetEngine(config)
        report = await engine.execute(dry_run=True)

        preserved_names = [
            a.resource for a in report.actions if a.action == EnumResetAction.PRESERVED
        ]

        for resource in PRESERVED_RESOURCES:
            assert resource in preserved_names, (
                f"Expected '{resource}' in preserved list"
            )

    @pytest.mark.asyncio
    async def test_preserved_count_minimum(self) -> None:
        """At least the static preserved resources are counted."""
        config = DemoResetConfig()
        engine = DemoResetEngine(config)
        report = await engine.execute(dry_run=True)
        assert report.preserved_count >= len(PRESERVED_RESOURCES)


# =============================================================================
# Engine Tests -- Idempotency
# =============================================================================


class TestDemoResetIdempotency:
    """Tests for idempotent behavior."""

    @pytest.mark.asyncio
    async def test_running_twice_same_result(self) -> None:
        """Running reset twice produces the same report structure."""
        config = DemoResetConfig()
        engine = DemoResetEngine(config)

        report1 = await engine.execute(dry_run=True)
        report2 = await engine.execute(dry_run=True)

        # Same number of actions and same action types
        assert len(report1.actions) == len(report2.actions)
        for a1, a2 in zip(report1.actions, report2.actions, strict=True):
            assert a1.resource == a2.resource
            assert a1.action == a2.action

    @pytest.mark.asyncio
    async def test_delete_zero_rows_is_not_error(self) -> None:
        """Deleting zero rows (already clean) is reported as reset, not error."""
        config = DemoResetConfig(postgres_dsn="postgresql://localhost/test")
        engine = DemoResetEngine(config)

        with patch.object(engine, "_delete_projection_rows", return_value=0):
            report = await engine.execute(dry_run=False)

        projector_actions = [a for a in report.actions if "Projector" in a.resource]
        assert projector_actions[0].action == EnumResetAction.RESET
        assert "0" in projector_actions[0].detail


# =============================================================================
# CLI Command Tests
# =============================================================================


class TestDemoResetCLI:
    """Tests for the Click CLI command interface."""

    def test_demo_group_exists(self) -> None:
        """Demo command group is registered."""
        from omnibase_infra.cli.commands import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["demo", "--help"])
        assert result.exit_code == 0
        assert "reset" in result.output.lower()

    def test_demo_reset_help(self) -> None:
        """Demo reset command has help text."""
        from omnibase_infra.cli.commands import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["demo", "reset", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--purge-topics" in result.output
        assert "--env-file" in result.output

    def test_demo_reset_dry_run_no_infra(self) -> None:
        """Dry run with no infrastructure configured succeeds."""
        import os

        from omnibase_infra.cli.commands import cli

        runner = CliRunner()
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OMNIBASE_INFRA_DB_URL", "KAFKA_BOOTSTRAP_SERVERS")
        }
        with patch.dict("os.environ", clean_env, clear=True):
            result = runner.invoke(cli, ["demo", "reset", "--dry-run"])

        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_existing_commands_not_broken(self) -> None:
        """Adding demo commands does not break existing CLI."""
        from omnibase_infra.cli.commands import cli

        runner = CliRunner()

        # Validate group still works
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0

        # Registry group still works
        result = runner.invoke(cli, ["registry", "--help"])
        assert result.exit_code == 0


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnumResetAction:
    """Tests for EnumResetAction values."""

    def test_all_values_exist(self) -> None:
        """All expected action types exist."""
        assert EnumResetAction.RESET == "reset"
        assert EnumResetAction.PRESERVED == "preserved"
        assert EnumResetAction.SKIPPED == "skipped"
        assert EnumResetAction.ERROR == "error"

    def test_string_representation(self) -> None:
        """Enum values are string-compatible."""
        assert EnumResetAction.RESET.value == "reset"
        assert EnumResetAction.PRESERVED.value == "preserved"
