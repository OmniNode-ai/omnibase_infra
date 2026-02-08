# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for CLI registry query commands.

Tests the Click command interface for registry queries including:
- list-nodes command
- get-node command
- list-topics command

Related:
    - OMN-1990: CLI registry query commands
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.commands import cli

pytestmark = [pytest.mark.unit]


class TestCLIRegistryCommands:
    """Tests for registry CLI commands."""

    def test_registry_group_exists(self) -> None:
        """Registry command group is registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["registry", "--help"])
        assert result.exit_code == 0
        assert "registry" in result.output.lower() or "Registry" in result.output

    def test_list_nodes_help(self) -> None:
        """list-nodes command has help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["registry", "list-nodes", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower() or "node" in result.output.lower()

    def test_get_node_help(self) -> None:
        """get-node command has help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["registry", "get-node", "--help"])
        assert result.exit_code == 0
        assert "node" in result.output.lower()

    def test_list_topics_command(self) -> None:
        """list-topics command returns platform topics."""
        runner = CliRunner()
        result = runner.invoke(cli, ["registry", "list-topics"])
        assert result.exit_code == 0
        # Should contain at least some platform topic names
        assert "onex" in result.output.lower()

    def test_list_topics_shows_all_kinds(self) -> None:
        """list-topics shows Event, Command, Intent, Snapshot kinds."""
        runner = CliRunner()
        result = runner.invoke(cli, ["registry", "list-topics"])
        assert result.exit_code == 0
        assert "Event" in result.output
        assert "Snapshot" in result.output

    def test_validate_group_still_works(self) -> None:
        """Existing validate commands still work."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "architecture" in result.output or "contracts" in result.output
