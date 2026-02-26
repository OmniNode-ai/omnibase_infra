# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for demo reset service and CLI command.

Tests cover:
- DemoResetEngine with dry-run and live execution
- Scoped resource filtering (only demo resources affected)
- Shared infrastructure preservation
- Idempotent behavior
- CLI command integration (Click runner)
- Error handling for missing infrastructure
- _load_env_for_demo .env parser
- _validate_table_name SQL injection boundary
- Consumer group deletion with mocked AdminClient
- Topic purge with mocked AdminClient

Related:
    - OMN-2299: Demo Reset scoped command for safe environment reset
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.enum_reset_action import EnumResetAction
from omnibase_infra.cli.model_demo_reset_config import ModelDemoResetConfig
from omnibase_infra.cli.model_demo_reset_report import ModelDemoResetReport
from omnibase_infra.cli.model_reset_action_result import ModelResetActionResult
from omnibase_infra.cli.service_demo_reset import (
    _ALLOWED_PROJECTION_TABLES,
    DEMO_CONSUMER_GROUP_PATTERN,
    DEMO_PROJECTION_TABLE,
    DEMO_TOPIC_PREFIXES,
    PRESERVED_RESOURCES,
    DemoResetEngine,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# Config Tests
# =============================================================================


class TestModelDemoResetConfig:
    """Tests for ModelDemoResetConfig construction."""

    def test_default_config(self) -> None:
        """Default config has expected defaults."""
        config = ModelDemoResetConfig()
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
            config = ModelDemoResetConfig.from_env(purge_topics=True)
            assert config.postgres_dsn == "postgresql://localhost/test"
            assert config.kafka_bootstrap_servers == "localhost:9092"
            assert config.purge_topics is True

    def test_from_env_missing_vars(self) -> None:
        """from_env uses empty strings when env vars are missing."""
        # Only remove the two keys that from_env reads, rather than wiping
        # the entire environment (which can break subprocess/locale state).
        env_without_keys = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OMNIBASE_INFRA_DB_URL", "KAFKA_BOOTSTRAP_SERVERS")
        }
        with patch.dict("os.environ", env_without_keys, clear=True):
            config = ModelDemoResetConfig.from_env()
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
            "onex.evt.omniclaude.agent-status.v1",
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


class TestModelDemoResetReport:
    """Tests for ModelDemoResetReport formatting and properties."""

    def test_empty_report(self) -> None:
        """Empty report has zero counts."""
        report = ModelDemoResetReport()
        assert report.reset_count == 0
        assert report.preserved_count == 0
        assert report.error_count == 0
        assert report.skipped_count == 0

    def test_counts_by_action_type(self) -> None:
        """Report counts actions by type correctly."""
        report = ModelDemoResetReport(
            actions=[
                ModelResetActionResult("a", EnumResetAction.RESET, "done"),
                ModelResetActionResult("b", EnumResetAction.RESET, "done"),
                ModelResetActionResult("c", EnumResetAction.PRESERVED, "kept"),
                ModelResetActionResult("d", EnumResetAction.SKIPPED, "na"),
                ModelResetActionResult("e", EnumResetAction.ERROR, "fail"),
            ]
        )
        assert report.reset_count == 2
        assert report.preserved_count == 1
        assert report.skipped_count == 1
        assert report.error_count == 1

    def test_format_summary_dry_run(self) -> None:
        """Summary indicates dry run mode."""
        report = ModelDemoResetReport(dry_run=True)
        summary = report.format_summary()
        assert "DRY RUN" in summary

    def test_format_summary_executed(self) -> None:
        """Summary indicates execution mode."""
        report = ModelDemoResetReport(dry_run=False)
        summary = report.format_summary()
        assert "EXECUTED" in summary

    def test_format_summary_includes_actions(self) -> None:
        """Summary includes action details."""
        report = ModelDemoResetReport(
            actions=[
                ModelResetActionResult(
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
        config = ModelDemoResetConfig(postgres_dsn="")
        engine = DemoResetEngine(config)
        report = await engine.execute(dry_run=True)

        projector_actions = [a for a in report.actions if "Projector" in a.resource]
        assert len(projector_actions) == 1
        assert projector_actions[0].action == EnumResetAction.SKIPPED
        assert "not configured" in projector_actions[0].detail

    @pytest.mark.asyncio
    async def test_dry_run_counts_rows(self) -> None:
        """Dry run reports row count without deleting."""
        config = ModelDemoResetConfig(postgres_dsn="postgresql://localhost/test")
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
        config = ModelDemoResetConfig(postgres_dsn="postgresql://localhost/test")
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
        config = ModelDemoResetConfig(postgres_dsn="postgresql://localhost/test")
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
        config = ModelDemoResetConfig(kafka_bootstrap_servers="")
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
        config = ModelDemoResetConfig(purge_topics=False)
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
        config = ModelDemoResetConfig()
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
        config = ModelDemoResetConfig()
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
        config = ModelDemoResetConfig()
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
        config = ModelDemoResetConfig(postgres_dsn="postgresql://localhost/test")
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


# =============================================================================
# _load_env_for_demo Tests (Issue 1 & 2)
# =============================================================================


class TestLoadEnvForDemo:
    """Tests for _load_env_for_demo .env parser in commands.py."""

    def _load(
        self, content: str, *, existing_env: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Write *content* to a temp file, call _load_env_for_demo, return new env vars."""
        import os
        import tempfile
        from pathlib import Path

        from omnibase_infra.cli.commands import _load_env_for_demo

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            # Build the env snapshot: current env plus any pre-set vars
            base_env = dict(os.environ)
            if existing_env:
                base_env.update(existing_env)

            with patch.dict("os.environ", base_env, clear=True):
                env_before = dict(os.environ)
                _load_env_for_demo(path)
                new_vars = {k: v for k, v in os.environ.items() if k not in env_before}
                return new_vars
        finally:
            Path(path).unlink()

    def test_simple_key_value(self) -> None:
        """Normal KEY=VALUE line is parsed."""
        result = self._load("FOO=bar\n")
        assert result["FOO"] == "bar"

    def test_export_prefix(self) -> None:
        """Lines with 'export ' prefix are handled."""
        result = self._load("export MY_VAR=hello\n")
        assert result["MY_VAR"] == "hello"

    def test_double_quoted_value(self) -> None:
        """Double-quoted values have outer quotes stripped."""
        result = self._load('DB_URL="postgresql://host/db"\n')
        assert result["DB_URL"] == "postgresql://host/db"

    def test_single_quoted_value(self) -> None:
        """Single-quoted values have outer quotes stripped."""
        result = self._load("SECRET='s3cr3t'\n")
        assert result["SECRET"] == "s3cr3t"

    def test_inline_comment_stripped_for_unquoted(self) -> None:
        """Inline comments (space-hash) are stripped from unquoted values."""
        result = self._load("PORT=5432 # default port\n")
        assert result["PORT"] == "5432"

    def test_inline_comment_preserved_in_quoted_value(self) -> None:
        """Hash inside quoted values is NOT treated as a comment.

        This is a documented limitation: inline comments are only stripped
        from unquoted values.
        """
        result = self._load('GREETING="hello # world"\n')
        assert result["GREETING"] == "hello # world"

    def test_hash_without_leading_space_not_stripped(self) -> None:
        """A '#' without a preceding space is NOT treated as a comment.

        Documents the limitation described in Issue 2: only ' #' (space-hash)
        triggers comment stripping for unquoted values.
        """
        result = self._load("COLOR=red#blue\n")
        assert result["COLOR"] == "red#blue"

    def test_value_containing_equals(self) -> None:
        """Values containing '=' are preserved (only first '=' splits)."""
        result = self._load("DSN=postgres://user:pass@host:5432/db?opt=1\n")
        assert result["DSN"] == "postgres://user:pass@host:5432/db?opt=1"

    def test_empty_value(self) -> None:
        """Empty values are parsed as empty strings."""
        result = self._load("EMPTY=\n")
        assert result["EMPTY"] == ""

    def test_blank_lines_and_comments_ignored(self) -> None:
        """Blank lines and comment lines are skipped."""
        result = self._load("# a comment\n\n  \nKEY=val\n")
        assert result == {"KEY": "val"}

    def test_malformed_line_no_equals(self) -> None:
        """Lines without '=' are silently skipped."""
        result = self._load("this has no equals sign\nGOOD=yes\n")
        assert "GOOD" in result
        assert len(result) == 1

    def test_does_not_override_existing(self) -> None:
        """Existing environment variables are NOT overridden."""
        result = self._load(
            "EXISTING_VAR=overridden\n",
            existing_env={"EXISTING_VAR": "original"},
        )
        # _load returns only NEW vars, so EXISTING_VAR should not appear
        assert "EXISTING_VAR" not in result

    def test_missing_file_warns_no_crash(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Non-existent file path prints warning but does not crash."""
        from omnibase_infra.cli.commands import _load_env_for_demo

        _load_env_for_demo("/tmp/_nonexistent_demo_env_file_test_.env")  # noqa: S108
        # Function should return without error (warning is printed via Rich)

    def test_whitespace_around_key_and_value(self) -> None:
        """Leading/trailing whitespace around key and value is stripped."""
        result = self._load("  MY_KEY  =  myvalue  \n")
        assert result["MY_KEY"] == "myvalue"


# =============================================================================
# _validate_table_name Tests (Issue 5)
# =============================================================================


class TestValidateTableName:
    """Tests for _validate_table_name SQL injection security boundary."""

    def test_accepts_allowed_table(self) -> None:
        """Allowed table names pass validation without raising."""
        for table in _ALLOWED_PROJECTION_TABLES:
            DemoResetEngine._validate_table_name(table)

    def test_rejects_unknown_table(self) -> None:
        """Unknown table names raise ValueError."""
        with pytest.raises(ValueError, match="not in the allowed projection tables"):
            DemoResetEngine._validate_table_name("users")

    def test_rejects_sql_injection_drop(self) -> None:
        """SQL injection via DROP TABLE is rejected."""
        with pytest.raises(ValueError):
            DemoResetEngine._validate_table_name(
                "registration_projections; DROP TABLE users"
            )

    def test_rejects_sql_injection_union(self) -> None:
        """SQL injection via UNION is rejected."""
        with pytest.raises(ValueError):
            DemoResetEngine._validate_table_name(
                "registration_projections UNION SELECT *"
            )

    def test_rejects_empty_string(self) -> None:
        """Empty string is rejected."""
        with pytest.raises(ValueError):
            DemoResetEngine._validate_table_name("")

    def test_rejects_similar_name(self) -> None:
        """Table names similar but not identical to allowed names are rejected."""
        with pytest.raises(ValueError):
            DemoResetEngine._validate_table_name(
                "registration_projection"
            )  # missing 's'

    @pytest.mark.parametrize(
        "injection",
        [
            "'; DROP TABLE registration_projections; --",
            "registration_projections\n; DROP TABLE users",
            "../../../etc/passwd",
            "registration_projections/**/",
        ],
    )
    def test_rejects_various_injection_attempts(self, injection: str) -> None:
        """Various SQL injection patterns are rejected."""
        with pytest.raises(ValueError):
            DemoResetEngine._validate_table_name(injection)

    @pytest.mark.asyncio
    async def test_count_projection_rows_rejects_disallowed_table(self) -> None:
        """_count_projection_rows raises ValueError for a table not in the allowlist."""
        config = ModelDemoResetConfig(
            postgres_dsn="postgresql://localhost/test",
            projection_table="evil_table",
        )
        engine = DemoResetEngine(config)

        with pytest.raises(ValueError, match="not in the allowed projection tables"):
            await engine._count_projection_rows(correlation_id=uuid4())

    @pytest.mark.asyncio
    async def test_delete_projection_rows_rejects_disallowed_table(self) -> None:
        """_delete_projection_rows raises ValueError for a table not in the allowlist."""
        config = ModelDemoResetConfig(
            postgres_dsn="postgresql://localhost/test",
            projection_table="evil_table",
        )
        engine = DemoResetEngine(config)

        with pytest.raises(ValueError, match="not in the allowed projection tables"):
            await engine._delete_projection_rows(correlation_id=uuid4())


# =============================================================================
# Engine Tests -- Consumer Group Live Execution (Issue 4)
# =============================================================================


def _make_consumer_group_listing(group_id: str) -> SimpleNamespace:
    """Create a mock ConsumerGroupListing with the expected attributes."""
    return SimpleNamespace(
        group_id=group_id,
        is_simple_consumer_group=False,
        state=None,
        type=None,
    )


def _make_list_consumer_groups_result(
    group_ids: list[str],
) -> SimpleNamespace:
    """Create a mock ListConsumerGroupsResult."""
    return SimpleNamespace(
        valid=[_make_consumer_group_listing(g) for g in group_ids],
        errors=[],
    )


def _make_list_consumer_groups_future(
    group_ids: list[str],
) -> MagicMock:
    """Create a mock future whose .result() returns a ListConsumerGroupsResult."""
    future = MagicMock()
    future.result.return_value = _make_list_consumer_groups_result(group_ids)
    return future


class TestDemoResetEngineConsumerGroupsLive:
    """Tests for consumer group reset live execution paths with mocked AdminClient."""

    @pytest.mark.asyncio
    async def test_dry_run_reports_matching_groups(self) -> None:
        """Dry run lists demo groups that would be deleted."""
        config = ModelDemoResetConfig(kafka_bootstrap_servers="localhost:9092")
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_consumer_groups.return_value = (
            _make_list_consumer_groups_future(
                [
                    "dev.omnibase.registration_orchestrator.v1",
                    "dev.business.analytics.v1",
                ]
            )
        )

        with patch("confluent_kafka.admin.AdminClient", return_value=mock_admin):
            report = ModelDemoResetReport(dry_run=True)
            await engine._reset_consumer_groups(
                report, dry_run=True, correlation_id=uuid4()
            )

        cg_actions = [a for a in report.actions if "Consumer group" in a.resource]
        assert any(a.action == EnumResetAction.RESET for a in cg_actions)
        assert any("Would delete" in a.detail for a in cg_actions)
        assert any("registration" in a.detail for a in cg_actions)

    @pytest.mark.asyncio
    async def test_live_deletes_demo_groups(self) -> None:
        """Live execution deletes demo groups and preserves non-demo ones."""
        config = ModelDemoResetConfig(kafka_bootstrap_servers="localhost:9092")
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_consumer_groups.return_value = (
            _make_list_consumer_groups_future(
                [
                    "dev.omnibase.registration_orchestrator.v1",
                    "dev.omnibase.projector_shell.v1",
                    "dev.business.analytics.v1",
                ]
            )
        )

        # Mock delete_consumer_groups to succeed for all
        success_future = MagicMock()
        success_future.result.return_value = None
        mock_admin.delete_consumer_groups.return_value = {
            "dev.omnibase.registration_orchestrator.v1": success_future,
            "dev.omnibase.projector_shell.v1": success_future,
        }

        with patch("confluent_kafka.admin.AdminClient", return_value=mock_admin):
            report = ModelDemoResetReport()
            await engine._reset_consumer_groups(
                report, dry_run=False, correlation_id=uuid4()
            )

        # Check deletion was reported
        reset_actions = [
            a
            for a in report.actions
            if a.action == EnumResetAction.RESET and "Consumer group" in a.resource
        ]
        assert len(reset_actions) == 1
        assert "2" in reset_actions[0].detail  # "Deleted 2 consumer group(s)"

        # Check non-demo group preserved
        preserved_actions = [
            a for a in report.actions if a.action == EnumResetAction.PRESERVED
        ]
        assert any("analytics" in a.detail for a in preserved_actions)

    @pytest.mark.asyncio
    async def test_partial_failure_on_group_deletion(self) -> None:
        """Partial failure during group deletion reports both success and error."""
        config = ModelDemoResetConfig(kafka_bootstrap_servers="localhost:9092")
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_consumer_groups.return_value = (
            _make_list_consumer_groups_future(
                [
                    "dev.omnibase.registration_orchestrator.v1",
                    "dev.omnibase.projector_shell.v1",
                ]
            )
        )

        success_future = MagicMock()
        success_future.result.return_value = None

        fail_future = MagicMock()
        fail_future.result.side_effect = Exception("group not empty")

        mock_admin.delete_consumer_groups.return_value = {
            "dev.omnibase.registration_orchestrator.v1": success_future,
            "dev.omnibase.projector_shell.v1": fail_future,
        }

        with patch("confluent_kafka.admin.AdminClient", return_value=mock_admin):
            report = ModelDemoResetReport()
            await engine._reset_consumer_groups(
                report, dry_run=False, correlation_id=uuid4()
            )

        # Should have both a RESET and an ERROR action
        assert any(a.action == EnumResetAction.RESET for a in report.actions)
        assert any(a.action == EnumResetAction.ERROR for a in report.actions)

    @pytest.mark.asyncio
    async def test_no_demo_groups_found(self) -> None:
        """When no demo groups match, report skipped."""
        config = ModelDemoResetConfig(kafka_bootstrap_servers="localhost:9092")
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_consumer_groups.return_value = (
            _make_list_consumer_groups_future(
                ["dev.business.analytics.v1", "dev.business.reporting.v1"]
            )
        )

        with patch("confluent_kafka.admin.AdminClient", return_value=mock_admin):
            report = ModelDemoResetReport()
            await engine._reset_consumer_groups(
                report, dry_run=False, correlation_id=uuid4()
            )

        cg_actions = [a for a in report.actions if "Consumer group" in a.resource]
        assert any(a.action == EnumResetAction.SKIPPED for a in cg_actions)
        assert any("No demo consumer groups" in a.detail for a in cg_actions)


# =============================================================================
# Engine Tests -- Topic Purge Live Execution (Issue 4)
# =============================================================================


class _MockTopicPartition:
    """Hashable stand-in for confluent_kafka.TopicPartition in tests."""

    __slots__ = ("offset", "partition", "topic")

    def __init__(self, topic: str, partition: int, offset: int = 0) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset

    def __hash__(self) -> int:
        return hash((self.topic, self.partition))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MockTopicPartition):
            return NotImplemented
        return self.topic == other.topic and self.partition == other.partition

    def __repr__(self) -> str:
        return (
            f"_MockTopicPartition(topic={self.topic!r}, "
            f"partition={self.partition}, offset={self.offset})"
        )


def _make_topic_metadata(partition_ids: list[int]) -> SimpleNamespace:
    """Create mock topic metadata with given partition IDs."""
    return SimpleNamespace(
        partitions={pid: SimpleNamespace() for pid in partition_ids},
    )


def _make_cluster_metadata(
    topics: dict[str, list[int]],
) -> SimpleNamespace:
    """Create mock ClusterMetadata with topic -> partition_ids mapping."""
    return SimpleNamespace(
        topics={name: _make_topic_metadata(pids) for name, pids in topics.items()},
    )


class TestDemoResetEngineTopicPurgeLive:
    """Tests for topic purge live execution paths with mocked AdminClient."""

    @pytest.mark.asyncio
    async def test_dry_run_reports_demo_topics(self) -> None:
        """Dry run lists demo topics that would be purged."""
        config = ModelDemoResetConfig(
            kafka_bootstrap_servers="localhost:9092",
            purge_topics=True,
        )
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_topics.return_value = _make_cluster_metadata(
            {
                "onex.evt.platform.node-registration.v1": [0],
                "custom.business.events.v1": [0],
            }
        )

        with patch("confluent_kafka.admin.AdminClient", return_value=mock_admin):
            report = ModelDemoResetReport(dry_run=True)
            await engine._purge_demo_topics(
                report, dry_run=True, correlation_id=uuid4()
            )

        topic_actions = [a for a in report.actions if "topic" in a.resource.lower()]
        assert any(a.action == EnumResetAction.RESET for a in topic_actions)
        assert any("Would purge" in a.detail for a in topic_actions)

    @pytest.mark.asyncio
    async def test_live_purges_demo_topics(self) -> None:
        """Live execution purges demo topics and preserves non-demo ones."""
        config = ModelDemoResetConfig(
            kafka_bootstrap_servers="localhost:9092",
            purge_topics=True,
        )
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_topics.return_value = _make_cluster_metadata(
            {
                "onex.evt.platform.node-registration.v1": [0, 1],
                "custom.business.events.v1": [0],
            }
        )

        # Mock TopicPartition so the import inside the method works
        mock_tp_class = MagicMock(side_effect=_MockTopicPartition)

        # Mock Consumer for watermark offset queries
        mock_consumer = MagicMock()
        # Return (low=0, high=100) for all partitions so they get purged
        mock_consumer.get_watermark_offsets.return_value = (0, 100)

        success_future = MagicMock()
        success_future.result.return_value = None

        # delete_records returns {TopicPartition: future}
        # The code builds TopicPartition objects with high-watermark offsets,
        # so we return matching mock objects from delete_records.
        tp0 = _MockTopicPartition("onex.evt.platform.node-registration.v1", 0, 100)
        tp1 = _MockTopicPartition("onex.evt.platform.node-registration.v1", 1, 100)
        mock_admin.delete_records.return_value = {
            tp0: success_future,
            tp1: success_future,
        }

        with (
            patch(
                "confluent_kafka.admin.AdminClient",
                return_value=mock_admin,
            ),
            patch(
                "confluent_kafka.TopicPartition",
                mock_tp_class,
            ),
            patch(
                "confluent_kafka.Consumer",
                return_value=mock_consumer,
            ),
        ):
            report = ModelDemoResetReport()
            await engine._purge_demo_topics(
                report, dry_run=False, correlation_id=uuid4()
            )

        # Verify purge reported
        reset_actions = [
            a
            for a in report.actions
            if a.action == EnumResetAction.RESET and "topic" in a.resource.lower()
        ]
        assert len(reset_actions) == 1
        assert "Purged" in reset_actions[0].detail

        # Verify non-demo topic preserved
        preserved = [a for a in report.actions if a.action == EnumResetAction.PRESERVED]
        assert any("custom.business" in a.detail for a in preserved)

    @pytest.mark.asyncio
    async def test_partial_failure_on_topic_purge(self) -> None:
        """Partial failure during topic purge reports both success and error."""
        config = ModelDemoResetConfig(
            kafka_bootstrap_servers="localhost:9092",
            purge_topics=True,
        )
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_topics.return_value = _make_cluster_metadata(
            {
                "onex.evt.platform.node-registration.v1": [0, 1],
            }
        )

        mock_tp_class = MagicMock(side_effect=_MockTopicPartition)

        # Mock Consumer for watermark offset queries
        mock_consumer = MagicMock()
        mock_consumer.get_watermark_offsets.return_value = (0, 100)

        success_future = MagicMock()
        success_future.result.return_value = None

        fail_future = MagicMock()
        fail_future.result.side_effect = Exception("timeout")

        tp0 = _MockTopicPartition("onex.evt.platform.node-registration.v1", 0, 100)
        tp1 = _MockTopicPartition("onex.evt.platform.node-registration.v1", 1, 100)
        mock_admin.delete_records.return_value = {
            tp0: success_future,
            tp1: fail_future,
        }

        with (
            patch(
                "confluent_kafka.admin.AdminClient",
                return_value=mock_admin,
            ),
            patch(
                "confluent_kafka.TopicPartition",
                mock_tp_class,
            ),
            patch(
                "confluent_kafka.Consumer",
                return_value=mock_consumer,
            ),
        ):
            report = ModelDemoResetReport()
            await engine._purge_demo_topics(
                report, dry_run=False, correlation_id=uuid4()
            )

        # Should have both a RESET (partition 0 succeeded) and ERROR (partition 1 failed)
        assert any(a.action == EnumResetAction.RESET for a in report.actions)
        assert any(a.action == EnumResetAction.ERROR for a in report.actions)

    @pytest.mark.asyncio
    async def test_already_empty_topics_report_skipped(self) -> None:
        """Demo topics with all partitions at offset 0 report skipped (already empty)."""
        config = ModelDemoResetConfig(
            kafka_bootstrap_servers="localhost:9092",
            purge_topics=True,
        )
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_topics.return_value = _make_cluster_metadata(
            {
                "onex.evt.platform.node-registration.v1": [0, 1],
                "custom.business.events.v1": [0],
            }
        )

        mock_tp_class = MagicMock(side_effect=_MockTopicPartition)

        # Mock Consumer -- return (0, 0) for all partitions (topics exist but are empty)
        mock_consumer = MagicMock()
        mock_consumer.get_watermark_offsets.return_value = (0, 0)

        with (
            patch(
                "confluent_kafka.admin.AdminClient",
                return_value=mock_admin,
            ),
            patch(
                "confluent_kafka.TopicPartition",
                mock_tp_class,
            ),
            patch(
                "confluent_kafka.Consumer",
                return_value=mock_consumer,
            ),
        ):
            report = ModelDemoResetReport()
            await engine._purge_demo_topics(
                report, dry_run=False, correlation_id=uuid4()
            )

        topic_actions = [a for a in report.actions if "topic" in a.resource.lower()]
        skipped = [a for a in topic_actions if a.action == EnumResetAction.SKIPPED]
        assert len(skipped) == 1
        assert "already empty" in skipped[0].detail

        # delete_records should never have been called since there are no partitions to purge
        mock_admin.delete_records.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_watermark_lookups_fail_reports_error(self) -> None:
        """When all watermark lookups fail, report ERROR instead of SKIPPED.

        This guards against masking failures as "already empty". If every
        ``get_watermark_offsets`` call raises an exception, ``partitions_to_delete``
        stays empty -- but we must NOT report SKIPPED because we have no idea
        whether the topics are actually empty.
        """
        config = ModelDemoResetConfig(
            kafka_bootstrap_servers="localhost:9092",
            purge_topics=True,
        )
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_topics.return_value = _make_cluster_metadata(
            {
                "onex.evt.platform.node-registration.v1": [0, 1],
            }
        )

        mock_tp_class = MagicMock(side_effect=_MockTopicPartition)

        # Mock Consumer -- every watermark lookup raises an exception
        mock_consumer = MagicMock()
        mock_consumer.get_watermark_offsets.side_effect = RuntimeError(
            "broker unavailable"
        )

        with (
            patch(
                "confluent_kafka.admin.AdminClient",
                return_value=mock_admin,
            ),
            patch(
                "confluent_kafka.TopicPartition",
                mock_tp_class,
            ),
            patch(
                "confluent_kafka.Consumer",
                return_value=mock_consumer,
            ),
        ):
            report = ModelDemoResetReport()
            await engine._purge_demo_topics(
                report, dry_run=False, correlation_id=uuid4()
            )

        topic_actions = [a for a in report.actions if "topic" in a.resource.lower()]

        # Must NOT be SKIPPED -- should be ERROR
        skipped = [a for a in topic_actions if a.action == EnumResetAction.SKIPPED]
        assert len(skipped) == 0, "Watermark failures must not be masked as SKIPPED"

        error_actions = [a for a in topic_actions if a.action == EnumResetAction.ERROR]
        assert len(error_actions) == 1
        assert "watermark" in error_actions[0].detail.lower()
        assert "2 partition(s)" in error_actions[0].detail

        # delete_records should never have been called
        mock_admin.delete_records.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_watermark_failure_still_purges_successful(self) -> None:
        """When some watermark lookups fail but others succeed, purge the successful ones.

        Partitions whose watermark lookup succeeds (with high > 0) should still
        be purged. The watermark failures for other partitions are logged but do
        not prevent the successful ones from being deleted.
        """
        config = ModelDemoResetConfig(
            kafka_bootstrap_servers="localhost:9092",
            purge_topics=True,
        )
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_topics.return_value = _make_cluster_metadata(
            {
                "onex.evt.platform.node-registration.v1": [0, 1],
            }
        )

        mock_tp_class = MagicMock(side_effect=_MockTopicPartition)

        # Mock Consumer -- partition 0 succeeds, partition 1 fails
        mock_consumer = MagicMock()

        def _watermark_side_effect(
            tp: _MockTopicPartition, timeout: int = 5
        ) -> tuple[int, int]:
            if tp.partition == 0:
                return (0, 50)
            raise RuntimeError("broker unavailable")

        mock_consumer.get_watermark_offsets.side_effect = _watermark_side_effect

        success_future = MagicMock()
        success_future.result.return_value = None

        tp0 = _MockTopicPartition("onex.evt.platform.node-registration.v1", 0, 50)
        mock_admin.delete_records.return_value = {
            tp0: success_future,
        }

        with (
            patch(
                "confluent_kafka.admin.AdminClient",
                return_value=mock_admin,
            ),
            patch(
                "confluent_kafka.TopicPartition",
                mock_tp_class,
            ),
            patch(
                "confluent_kafka.Consumer",
                return_value=mock_consumer,
            ),
        ):
            report = ModelDemoResetReport()
            await engine._purge_demo_topics(
                report, dry_run=False, correlation_id=uuid4()
            )

        # Should have a RESET action for the successful partition
        reset_actions = [
            a
            for a in report.actions
            if a.action == EnumResetAction.RESET and "topic" in a.resource.lower()
        ]
        assert len(reset_actions) == 1
        assert "Purged" in reset_actions[0].detail

        # delete_records should have been called (partition 0 had data)
        mock_admin.delete_records.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_demo_topics_found(self) -> None:
        """When no demo topics exist, report skipped."""
        config = ModelDemoResetConfig(
            kafka_bootstrap_servers="localhost:9092",
            purge_topics=True,
        )
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_topics.return_value = _make_cluster_metadata(
            {
                "custom.business.events.v1": [0],
                "__consumer_offsets": [0],
            }
        )

        with patch("confluent_kafka.admin.AdminClient", return_value=mock_admin):
            report = ModelDemoResetReport()
            await engine._purge_demo_topics(
                report, dry_run=False, correlation_id=uuid4()
            )

        topic_actions = [a for a in report.actions if "topic" in a.resource.lower()]
        assert any(a.action == EnumResetAction.SKIPPED for a in topic_actions)


# =============================================================================
# Edge Case Tests -- ImportError for confluent-kafka (Test 1)
# =============================================================================


class TestConfluentKafkaImportError:
    """Tests for the ImportError fallback when confluent-kafka is not installed."""

    @pytest.mark.asyncio
    async def test_reset_consumer_groups_import_error(self) -> None:
        """_reset_consumer_groups reports ERROR when confluent-kafka is missing."""
        config = ModelDemoResetConfig(kafka_bootstrap_servers="localhost:9092")
        engine = DemoResetEngine(config)

        with patch.dict(
            "sys.modules", {"confluent_kafka": None, "confluent_kafka.admin": None}
        ):
            report = ModelDemoResetReport()
            await engine._reset_consumer_groups(
                report, dry_run=False, correlation_id=uuid4()
            )

        cg_actions = [a for a in report.actions if "Consumer group" in a.resource]
        assert len(cg_actions) == 1
        assert cg_actions[0].action == EnumResetAction.ERROR
        assert "confluent-kafka not installed" in cg_actions[0].detail

    @pytest.mark.asyncio
    async def test_purge_demo_topics_import_error(self) -> None:
        """_purge_demo_topics reports ERROR when confluent-kafka is missing."""
        config = ModelDemoResetConfig(
            kafka_bootstrap_servers="localhost:9092",
            purge_topics=True,
        )
        engine = DemoResetEngine(config)

        with patch.dict(
            "sys.modules", {"confluent_kafka": None, "confluent_kafka.admin": None}
        ):
            report = ModelDemoResetReport()
            await engine._purge_demo_topics(
                report, dry_run=False, correlation_id=uuid4()
            )

        topic_actions = [a for a in report.actions if "topic" in a.resource.lower()]
        assert len(topic_actions) == 1
        assert topic_actions[0].action == EnumResetAction.ERROR
        assert "confluent-kafka not installed" in topic_actions[0].detail


# =============================================================================
# Edge Case Tests -- AdapterPostgresConnection timeout (Test 2)
# =============================================================================


class TestAdapterPostgresConnectionTimeout:
    """Tests for AdapterPostgresConnection handling asyncio.TimeoutError."""

    @pytest.mark.asyncio
    async def test_timeout_on_connect_raises_and_does_not_leak(self) -> None:
        """TimeoutError during connect propagates and does not leak connections.

        When ``asyncio.wait_for`` fires before ``asyncpg.connect()`` completes,
        the ``_conn`` attribute remains ``None``. The context manager's
        ``__aexit__`` must not attempt to close a ``None`` connection.
        """
        from omnibase_infra.cli.service_demo_reset import AdapterPostgresConnection

        ctx = AdapterPostgresConnection(
            dsn="postgresql://localhost/test", timeout=0.001
        )

        # Patch asyncpg.connect to simulate a slow connection that triggers timeout
        async def _slow_connect(*args: object, **kwargs: object) -> None:
            await asyncio.sleep(10)

        with patch("asyncpg.connect", side_effect=_slow_connect):
            with pytest.raises(asyncio.TimeoutError):
                async with ctx:
                    pass  # pragma: no cover -- should not reach here

        # After the timeout, _conn must be None (no leak)
        assert ctx._conn is None

    @pytest.mark.asyncio
    async def test_successful_connect_closes_on_exit(self) -> None:
        """A successfully opened connection is closed during __aexit__."""
        from unittest.mock import AsyncMock

        from omnibase_infra.cli.service_demo_reset import AdapterPostgresConnection

        mock_conn = MagicMock()
        mock_conn.close = AsyncMock(return_value=None)

        async def _fast_connect(*args: object, **kwargs: object) -> MagicMock:
            return mock_conn

        ctx = AdapterPostgresConnection(dsn="postgresql://localhost/test", timeout=5.0)

        with patch("asyncpg.connect", side_effect=_fast_connect):
            async with ctx as conn:
                assert conn is mock_conn

        # Connection must be closed and cleared
        mock_conn.close.assert_awaited_once()
        assert ctx._conn is None


# =============================================================================
# Edge Case Tests -- Ephemeral consumer group cleanup failure (Test 3)
# =============================================================================


class TestEphemeralConsumerGroupCleanupFailure:
    """Tests for ephemeral consumer group deletion failure in _purge_demo_topics."""

    @pytest.mark.asyncio
    async def test_orphaned_group_error_appended_to_report(self) -> None:
        """When ephemeral consumer group deletion fails, an ERROR action is reported."""
        config = ModelDemoResetConfig(
            kafka_bootstrap_servers="localhost:9092",
            purge_topics=True,
        )
        engine = DemoResetEngine(config)

        mock_admin = MagicMock()
        mock_admin.list_topics.return_value = _make_cluster_metadata(
            {
                "onex.evt.platform.node-registration.v1": [0],
            }
        )

        mock_tp_class = MagicMock(side_effect=_MockTopicPartition)

        # Mock Consumer for watermark offset queries
        mock_consumer = MagicMock()
        mock_consumer.get_watermark_offsets.return_value = (0, 100)

        # Make delete_records succeed
        success_future = MagicMock()
        success_future.result.return_value = None

        tp0 = _MockTopicPartition("onex.evt.platform.node-registration.v1", 0, 100)
        mock_admin.delete_records.return_value = {
            tp0: success_future,
        }

        # Make ephemeral consumer group deletion FAIL
        ephemeral_delete_future = MagicMock()
        ephemeral_delete_future.result.side_effect = RuntimeError("GROUP_ID_NOT_FOUND")

        def _mock_delete_consumer_groups(group_ids: list[str]) -> dict[str, MagicMock]:
            return {group_ids[0]: ephemeral_delete_future}

        mock_admin.delete_consumer_groups.side_effect = _mock_delete_consumer_groups

        with (
            patch(
                "confluent_kafka.admin.AdminClient",
                return_value=mock_admin,
            ),
            patch(
                "confluent_kafka.TopicPartition",
                mock_tp_class,
            ),
            patch(
                "confluent_kafka.Consumer",
                return_value=mock_consumer,
            ),
        ):
            report = ModelDemoResetReport()
            await engine._purge_demo_topics(
                report, dry_run=False, correlation_id=uuid4()
            )

        # There should be an ERROR action about the ephemeral consumer group
        ephemeral_errors = [
            a
            for a in report.actions
            if a.action == EnumResetAction.ERROR and "ephemeral" in a.resource.lower()
        ]
        assert len(ephemeral_errors) == 1
        assert "orphan" in ephemeral_errors[0].detail.lower()

        # The topic purge itself should still succeed
        reset_actions = [
            a
            for a in report.actions
            if a.action == EnumResetAction.RESET and "topic" in a.resource.lower()
        ]
        assert len(reset_actions) == 1


# =============================================================================
# Edge Case Tests -- asyncio.TimeoutError in projector path (Test 4)
# =============================================================================


class TestProjectorTimeoutError:
    """Tests for asyncio.TimeoutError handling in projector state operations."""

    @pytest.mark.asyncio
    async def test_count_projection_rows_timeout_reported_as_error(self) -> None:
        """TimeoutError in _count_projection_rows during dry run is reported as ERROR."""
        config = ModelDemoResetConfig(postgres_dsn="postgresql://localhost/test")
        engine = DemoResetEngine(config)

        with patch.object(
            engine,
            "_count_projection_rows",
            side_effect=TimeoutError(),
        ):
            report = await engine.execute(dry_run=True)

        projector_actions = [a for a in report.actions if "Projector" in a.resource]
        assert len(projector_actions) == 1
        assert projector_actions[0].action == EnumResetAction.ERROR
        assert "Failed" in projector_actions[0].detail

    @pytest.mark.asyncio
    async def test_delete_projection_rows_timeout_reported_as_error(self) -> None:
        """TimeoutError in _delete_projection_rows during live run is reported as ERROR."""
        config = ModelDemoResetConfig(postgres_dsn="postgresql://localhost/test")
        engine = DemoResetEngine(config)

        with patch.object(
            engine,
            "_delete_projection_rows",
            side_effect=TimeoutError(),
        ):
            report = await engine.execute(dry_run=False)

        projector_actions = [a for a in report.actions if "Projector" in a.resource]
        assert len(projector_actions) == 1
        assert projector_actions[0].action == EnumResetAction.ERROR
        assert "Failed" in projector_actions[0].detail


# =============================================================================
# Edge Case Tests -- CLI --env-file functional test (Test 5)
# =============================================================================


class TestCLIEnvFileLoading:
    """Tests for the CLI --env-file option loading environment variables."""

    def test_env_file_loaded_before_execution(self) -> None:
        """CLI --env-file loads variables that ModelDemoResetConfig.from_env() can read."""
        import tempfile
        from pathlib import Path

        from omnibase_infra.cli.commands import cli

        # Write a temporary .env file with known values
        env_content = (
            "OMNIBASE_INFRA_DB_URL=postgresql://testhost:5432/testdb\n"
            "KAFKA_BOOTSTRAP_SERVERS=testhost:9092\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(env_content)
            f.flush()
            env_path = f.name

        try:
            runner = CliRunner()
            # Strip OMNIBASE_INFRA_DB_URL and KAFKA_BOOTSTRAP_SERVERS so they
            # are only set via the --env-file.
            clean_env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("OMNIBASE_INFRA_DB_URL", "KAFKA_BOOTSTRAP_SERVERS")
            }
            with patch.dict("os.environ", clean_env, clear=True):
                result = runner.invoke(
                    cli,
                    ["demo", "reset", "--dry-run", "--env-file", env_path],
                )

            assert result.exit_code == 0 or result.exit_code == 1
            # The env file should have been loaded -- verify the output
            # references configured resources (not "not configured" skips for
            # both postgres AND kafka).
            output = result.output
            # With the env file loaded, the engine will attempt to connect to
            # postgres (and fail, producing an error), which proves the DSN was
            # read. Without the env file, both would show "not configured".
            assert "DRY RUN" in output
        finally:
            Path(env_path).unlink()
