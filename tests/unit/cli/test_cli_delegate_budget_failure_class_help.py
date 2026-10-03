# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex delegate --help`` names the handler-budget timeout as a failure class (OMN-18838 AC3).

A triage lane that meets ``delegation exceeded the handler execution budget of
Ns and was cancelled`` should be able to tell it from a broker or network fault
without re-deriving the 2026-10-02 investigation (seven reasoning runs, wall
16.2-22.9s against the 240s budget, so the budget was not the present cause).
The string used to live only in receipt JSON; a lane searching before it
assumes a transport fault searches the command's help, so the help names it.
"""

from __future__ import annotations

import time

from click.testing import CliRunner

from omnibase_infra.cli.cli_delegate import delegate_command

FAILURE_CLASS_STRING = (
    "delegation exceeded the handler execution budget of Ns and was cancelled"
)


def _help_text() -> str:
    result = CliRunner().invoke(delegate_command, ["--help"])
    assert result.exit_code == 0, result.output
    # click wraps help to the terminal width: compare on whitespace-normalised text
    return " ".join(result.output.split())


def test_help_names_the_handler_budget_failure_class_string() -> None:
    assert FAILURE_CLASS_STRING in _help_text()


def test_help_states_the_240s_ceiling() -> None:
    assert "240s" in _help_text()


def test_help_distinguishes_the_class_from_a_broker_or_network_fault() -> None:
    help_text = _help_text().lower()
    assert "not a broker or network fault" in help_text
    assert "transport-class terminal" in help_text


def test_help_says_what_to_do_about_it() -> None:
    help_text = _help_text().lower()
    assert "queue wait" in help_text
    assert "rung latency" in help_text
    assert "split the prompt" in help_text


def test_help_renders_in_under_one_second() -> None:
    # Warm render: the first call pays the one-off import and contract read, which
    # is cold-start cost and not what a help paragraph can change (OMN-19444).
    _help_text()
    started = time.monotonic()
    _help_text()
    assert time.monotonic() - started < 1.0
