# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for onex-infra-test CLI scaffold and command structure.

Verifies that all CLI commands are properly wired, help text renders,
and the command hierarchy is correct.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.infra_test.cli import cli


@pytest.mark.unit
class TestCLIScaffold:
    """Test CLI command structure and help output."""

    def test_cli_top_level_help(self) -> None:
        """CLI root shows help with all subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ONEX Infrastructure Integration Test CLI" in result.output
        assert "env" in result.output
        assert "introspect" in result.output
        assert "verify" in result.output
        assert "run" in result.output

    def test_env_group_help(self) -> None:
        """env group shows up/down subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["env", "--help"])
        assert result.exit_code == 0
        assert "up" in result.output
        assert "down" in result.output

    def test_env_up_help(self) -> None:
        """env up command shows profile and wait options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["env", "up", "--help"])
        assert result.exit_code == 0
        assert "--profile" in result.output
        assert "--wait" in result.output
        assert "--timeout" in result.output

    def test_env_down_help(self) -> None:
        """env down command shows volumes option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["env", "down", "--help"])
        assert result.exit_code == 0
        assert "--volumes" in result.output

    def test_introspect_help(self) -> None:
        """introspect command shows node options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["introspect", "--help"])
        assert result.exit_code == 0
        assert "--node-id" in result.output
        assert "--node-type" in result.output
        assert "--topic" in result.output
        assert "--broker" in result.output

    def test_verify_group_help(self) -> None:
        """verify group shows all subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "--help"])
        assert result.exit_code == 0
        assert "registry" in result.output
        assert "topics" in result.output
        assert "snapshots" in result.output
        assert "idempotency" in result.output

    def test_verify_registry_help(self) -> None:
        """verify registry shows node-id filter option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "registry", "--help"])
        assert result.exit_code == 0
        assert "--node-id" in result.output

    def test_verify_topics_help(self) -> None:
        """verify topics help renders."""
        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "topics", "--help"])
        assert result.exit_code == 0
        assert "ONEX topic naming" in result.output

    def test_verify_snapshots_help(self) -> None:
        """verify snapshots shows topic option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "snapshots", "--help"])
        assert result.exit_code == 0
        assert "--topic" in result.output

    def test_verify_idempotency_help(self) -> None:
        """verify idempotency shows repetitions option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "idempotency", "--help"])
        assert result.exit_code == 0
        assert "--repetitions" in result.output

    def test_run_help(self) -> None:
        """run command shows suite option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--suite" in result.output
        assert "smoke" in result.output
        assert "failure" in result.output
        assert "idempotency" in result.output

    def test_compose_file_option(self) -> None:
        """Top-level --compose-file option is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--compose-file", "custom.yml", "--help"])
        assert result.exit_code == 0

    def test_project_name_option(self) -> None:
        """Top-level --project-name option is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--project-name", "test-project", "--help"])
        assert result.exit_code == 0
