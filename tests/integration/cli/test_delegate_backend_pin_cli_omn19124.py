# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for the rung pin (OMN-19124).

The unit module beside this one drives ``_write_payload`` and the receipt
helpers directly. This one goes through ``click`` -- the real command, the
real option parsing, the real refusal path -- because the defect this ticket
closes was reachable ONLY from a command line, and its RED proof is a parser
fact rather than a function fact.

Measured on the .201 dev lane, 2026-09-22, against the released CLI:

    Error: No such option '--backend-id'.   (exit 2)

That is the whole defect. ``backend_id`` was a declared optional input on the
delegate node contract, the handler threaded it and the local dispatch port
honoured it -- and no caller could say it. A payload-writer test cannot
observe that, because the payload writer was never the thing that was
missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration


def _invoke(args: list[str]) -> object:
    """Run the real command with option parsing, stopping before dispatch.

    Dispatch needs a co-installed omnimarket and a live model endpoint,
    neither of which belongs in this gate. Everything asserted here happens
    in FRONT of dispatch: option parsing and pin validation.
    """
    return CliRunner().invoke(delegate_command, args, catch_exceptions=False)


class TestTheRungPinIsReachableFromACommandLine:
    """The lane RED, as a test: the flag must PARSE."""

    def test_the_pin_flag_parses_rather_than_being_an_unknown_option(
        self, tmp_path: Path
    ) -> None:
        result = _invoke(
            [
                "summarise this",
                "--task-type",
                "summarization",
                "--backend-id",
                "cloud-glm",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert "No such option" not in result.output, (
            "the released CLI answered exactly this on the dev lane; a pin "
            "nobody can type is the whole of OMN-19124"
        )
        assert "--backend-id" not in result.output or "Usage:" not in result.output

    def test_the_flag_is_listed_in_the_commands_own_help(self) -> None:
        """A caller finds a flag by reading --help, not by reading source."""
        result = _invoke(["--help"])
        assert result.exit_code == 0
        assert "--backend-id" in result.output

    def test_an_empty_pin_is_refused_at_the_flag_and_names_it(
        self, tmp_path: Path
    ) -> None:
        """A pin silently dropped would walk the ladder and answer anyway.

        The refusal must fire in FRONT of the omnimarket drift guard, which
        is the next thing on this command's path and which reports something
        entirely unrelated to the caller's flag value.
        """
        result = _invoke(
            [
                "summarise this",
                "--task-type",
                "summarization",
                "--backend-id",
                "   ",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert result.exit_code != 0
        assert "--backend-id was given an empty value" in result.output
        assert "omnimarket is NOT INSTALLED" not in result.output, (
            "the pin refusal must precede the drift guard, or the caller is "
            "told about a co-install when their flag value is the problem"
        )

    def test_the_unpinned_command_line_is_unchanged(self, tmp_path: Path) -> None:
        """AC2 at the parser. An absent flag must not become a usage error."""
        result = _invoke(
            [
                "summarise this",
                "--task-type",
                "summarization",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert "No such option" not in result.output
        assert "--backend-id was given an empty value" not in result.output
