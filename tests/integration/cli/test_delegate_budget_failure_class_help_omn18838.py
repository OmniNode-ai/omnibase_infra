# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for the handler-budget failure class help (OMN-18838).

Drives the real ``delegate`` command through click so the help a triage lane
reaches from a command line names the handler-budget timeout.
"""

from __future__ import annotations

from click.testing import CliRunner

from omnibase_infra.cli.cli_delegate import delegate_command


def test_onex_delegate_help_names_handler_budget_failure_class() -> None:
    result = CliRunner().invoke(delegate_command, ["--help"])
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert (
        "delegation exceeded the handler execution budget of Ns and was cancelled"
        in text
    )
    assert "240s" in text
