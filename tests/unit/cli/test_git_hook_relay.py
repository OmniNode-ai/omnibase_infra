# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for onex-git-hook-relay CLI.

Tests cover:
    - Inline gates (--gates-json)
    - File gates (--gates-file)
    - Kafka-unavailable spool path (mocked)
    - Repo format validation
    - Event payload structure

Related Tickets:
    - OMN-2656: Phase 2 — Effect Nodes & CLIs (omnibase_infra)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.git_hook_relay import (
    ModelGitHookEmitParams,
    _build_event,
    _validate_repo,
    cli,
)
from omnibase_infra.errors import ProtocolConfigurationError


class TestValidateRepo:
    @pytest.mark.unit
    def test_valid_owner_name_format(self) -> None:
        _validate_repo("OmniNode-ai/omniclaude")  # no raise

    @pytest.mark.unit
    def test_valid_with_dots_and_dashes(self) -> None:
        _validate_repo("my-org/my.repo")  # no raise

    @pytest.mark.unit
    def test_rejects_absolute_path(self) -> None:
        with pytest.raises(ProtocolConfigurationError, match="Invalid repo format"):
            _validate_repo("/home/user/repos/omniclaude")

    @pytest.mark.unit
    def test_rejects_bare_name(self) -> None:
        with pytest.raises(ProtocolConfigurationError, match="Invalid repo format"):
            _validate_repo("omniclaude")

    @pytest.mark.unit
    def test_rejects_three_segments(self) -> None:
        with pytest.raises(ProtocolConfigurationError, match="Invalid repo format"):
            _validate_repo("org/repo/extra")


class TestBuildEvent:
    @pytest.mark.unit
    def test_event_structure(self) -> None:
        params = ModelGitHookEmitParams(
            hook="pre-commit",
            repo="OmniNode-ai/omniclaude",
            branch="main",
            author="jsmith",
            outcome="pass",
            gates=["lint", "tests"],
        )
        event = _build_event(params)
        assert event["event_type"] == "onex.evt.git.hook.v1"
        assert event["hook"] == "pre-commit"
        assert event["repo"] == "OmniNode-ai/omniclaude"
        assert event["branch"] == "main"
        assert event["author"] == "jsmith"
        assert event["outcome"] == "pass"
        assert event["gates"] == ["lint", "tests"]
        assert "correlation_id" in event
        assert "emitted_at" in event

    @pytest.mark.unit
    def test_event_empty_gates(self) -> None:
        params = ModelGitHookEmitParams(
            hook="post-receive",
            repo="OmniNode-ai/omnibase_core",
            branch="jonah/feature",
            author="jdoe",
            outcome="pass",
            gates=[],
        )
        event = _build_event(params)
        assert event["gates"] == []


class TestEmitCommandGatesJson:
    @pytest.mark.unit
    def test_emit_inline_gates_success(self) -> None:
        runner = CliRunner()
        with patch(
            "omnibase_infra.cli.git_hook_relay._publish_event",
            new=AsyncMock(return_value=True),
        ):
            result = runner.invoke(
                cli,
                [
                    "emit",
                    "--hook",
                    "pre-commit",
                    "--repo",
                    "OmniNode-ai/omniclaude",
                    "--branch",
                    "main",
                    "--author",
                    "jsmith",
                    "--outcome",
                    "pass",
                    "--gates-json",
                    '["lint", "tests"]',
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "Published pre-commit event" in result.output

    @pytest.mark.unit
    def test_emit_empty_gates_json(self) -> None:
        runner = CliRunner()
        with patch(
            "omnibase_infra.cli.git_hook_relay._publish_event",
            new=AsyncMock(return_value=True),
        ):
            result = runner.invoke(
                cli,
                [
                    "emit",
                    "--hook",
                    "pre-commit",
                    "--repo",
                    "OmniNode-ai/omniclaude",
                    "--branch",
                    "main",
                    "--author",
                    "jsmith",
                    "--outcome",
                    "allowed",
                    "--gates-json",
                    "[]",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    @pytest.mark.unit
    def test_emit_invalid_gates_json(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "emit",
                "--hook",
                "pre-commit",
                "--repo",
                "OmniNode-ai/omniclaude",
                "--branch",
                "main",
                "--author",
                "jsmith",
                "--outcome",
                "pass",
                "--gates-json",
                "not-json",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid --gates-json" in result.output or "Invalid" in (
            result.output + str(result.stderr or "")
        )


class TestEmitCommandGatesFile:
    @pytest.mark.unit
    def test_emit_from_file(self, tmp_path: Path) -> None:
        gates_file = tmp_path / "gates.json"
        gates_file.write_text('["ruff", "mypy"]', encoding="utf-8")

        runner = CliRunner()
        with patch(
            "omnibase_infra.cli.git_hook_relay._publish_event",
            new=AsyncMock(return_value=True),
        ):
            result = runner.invoke(
                cli,
                [
                    "emit",
                    "--hook",
                    "post-receive",
                    "--repo",
                    "OmniNode-ai/omnibase_infra",
                    "--branch",
                    "jonah/feature",
                    "--author",
                    "jsmith",
                    "--outcome",
                    "pass",
                    "--gates-file",
                    str(gates_file),
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    @pytest.mark.unit
    def test_emit_gates_file_must_be_array(self, tmp_path: Path) -> None:
        gates_file = tmp_path / "gates.json"
        gates_file.write_text('{"key": "value"}', encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "emit",
                "--hook",
                "pre-commit",
                "--repo",
                "OmniNode-ai/omniclaude",
                "--branch",
                "main",
                "--author",
                "jsmith",
                "--outcome",
                "pass",
                "--gates-file",
                str(gates_file),
            ],
        )
        assert result.exit_code == 1


class TestEmitKafkaUnavailableSpool:
    @pytest.mark.unit
    def test_spools_when_kafka_unavailable(self, tmp_path: Path) -> None:
        """When Kafka publish fails, event is spooled and CLI exits 0."""
        runner = CliRunner()
        spool_file = tmp_path / "spool" / "git-hooks.jsonl"

        with (
            patch(
                "omnibase_infra.cli.git_hook_relay._publish_event",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "omnibase_infra.cli.git_hook_relay._SPOOL_FILE",
                spool_file,
            ),
            patch(
                "omnibase_infra.cli.git_hook_relay._SPOOL_DIR",
                tmp_path / "spool",
            ),
        ):
            result = runner.invoke(
                cli,
                [
                    "emit",
                    "--hook",
                    "pre-commit",
                    "--repo",
                    "OmniNode-ai/omniclaude",
                    "--branch",
                    "main",
                    "--author",
                    "jsmith",
                    "--outcome",
                    "pass",
                    "--gates-json",
                    "[]",
                ],
                catch_exceptions=False,
            )

        # Always exits 0 — non-blocking
        assert result.exit_code == 0
        # Spool file should contain the event
        assert spool_file.exists()
        lines = spool_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["hook"] == "pre-commit"
        assert event["repo"] == "OmniNode-ai/omniclaude"


class TestEmitRepoValidation:
    @pytest.mark.unit
    def test_rejects_absolute_path_repo(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "emit",
                "--hook",
                "pre-commit",
                "--repo",
                "/home/user/project",
                "--branch",
                "main",
                "--author",
                "jsmith",
                "--outcome",
                "pass",
                "--gates-json",
                "[]",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid repo format" in (result.output + str(result.stderr or ""))

    @pytest.mark.unit
    def test_accepts_valid_owner_name(self) -> None:
        runner = CliRunner()
        with patch(
            "omnibase_infra.cli.git_hook_relay._publish_event",
            new=AsyncMock(return_value=True),
        ):
            result = runner.invoke(
                cli,
                [
                    "emit",
                    "--hook",
                    "pre-commit",
                    "--repo",
                    "OmniNode-ai/omniclaude",
                    "--branch",
                    "main",
                    "--author",
                    "jsmith",
                    "--outcome",
                    "allowed",
                    "--gates-json",
                    "[]",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
