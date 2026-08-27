# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for `onex run <workflow>` CLI harness (OMN-9260).

`onex run` is the local workflow harness in released omnibase-core. These tests
verify:
  - `onex run` exists (not "No such command 'run'")
  - `onex run <unknown>` is rejected against the known workflow set
  - `onex run --help` lists the harness flags
  - legacy `onex node` flags are not silently accepted by `onex run`
"""

from __future__ import annotations

from click.testing import CliRunner

from omnibase_core.cli.cli_commands import cli


def test_onex_run_command_exists() -> None:
    """``onex run`` must not return 'No such command'."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert "No such command" not in result.output
    assert result.exit_code == 0


def test_onex_run_help_lists_expected_flags() -> None:
    """``onex run --help`` must expose the local workflow harness flags."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    output = result.output
    assert "--prompt" in output
    assert "--correlation-id" in output
    assert "--task-type" in output
    assert "--max-tokens" in output
    assert "--inference" in output
    assert "--fixture-completion" in output
    assert "--sqlite-path" in output
    assert "--runtime-sha" in output


def test_onex_run_unknown_workflow_reports_known_choices() -> None:
    """``onex run <unknown>`` must fail at click's workflow choice boundary."""
    runner = CliRunner()
    run_result = runner.invoke(cli, ["run", "definitely_not_a_real_node"])

    assert run_result.exit_code != 0

    run_combined = run_result.output + str(run_result.exception or "")
    assert "Invalid value for" in run_combined
    assert "definitely_not_a_real_node" in run_combined
    assert "delegation" in run_combined
    assert "sea" in run_combined


def test_onex_run_and_node_have_distinct_unknown_error_boundaries() -> None:
    """``run`` is workflow-scoped while ``node`` remains node-name scoped."""
    runner = CliRunner()
    run_result = runner.invoke(cli, ["run", "no_such_node_xyz"])
    node_result = runner.invoke(cli, ["node", "no_such_node_xyz"])

    run_err = run_result.output + str(run_result.exception or "")
    node_err = node_result.output + str(node_result.exception or "")

    assert "Invalid value for" in run_err
    assert "no_such_node_xyz" in run_err
    assert "Unknown node 'no_such_node_xyz'" in node_err


def test_onex_run_rejects_legacy_node_input_option() -> None:
    """``onex run`` must not silently accept the node-only ``--input`` flag."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "delegation",
            "--input",
            "does_not_exist.json",
        ],
    )
    assert result.exit_code != 0
    assert "No such option '--input'" in result.output


def test_onex_run_rejects_legacy_node_contract_option() -> None:
    """``onex run`` must not silently accept the node-only ``--contract`` flag."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "delegation",
            "--contract",
            "custom_contract.yaml",
        ],
    )
    assert result.exit_code != 0
    assert "No such option '--contract'" in result.output
