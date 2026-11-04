# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for the ``onex`` command group after ``onex occ`` left (OMN-20074).

``tests/ci/test_no_occ_cli_omn20074.py`` reads ``pyproject.toml`` and the tree as
data. This module goes through the real ``onex`` command group that
``omnibase_core`` builds from the installed ``onex.cli`` entry points: rendering
the root listing resolves every advertised extension (a dangling entry point
raises there), and ``onex occ`` must be an unknown command rather than a
half-removed one.

Falsifier: re-register ``occ`` under ``onex.cli`` (or leave a dangling
``cli_occ`` entry point) and the listing test or the unknown-command test fails.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from omnibase_core.cli.cli_commands import cli

pytestmark = pytest.mark.integration

# Extensions this distribution contributes to the ``onex`` group that survive the
# removal; a listing that renders them proves the group still resolves each one.
_SURVIVING_INFRA_COMMANDS = ("auth", "delegate", "kafka", "ledger", "node", "profile")


def test_root_listing_resolves_every_extension_and_names_no_occ() -> None:
    result = CliRunner().invoke(cli, ["--help"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    listed = {
        line.split()[0] for line in result.output.splitlines() if line.startswith("  ")
    }
    assert "occ" not in listed
    for name in _SURVIVING_INFRA_COMMANDS:
        assert name in listed, f"{name} missing from the onex root listing"


def test_occ_is_an_unknown_command() -> None:
    result = CliRunner().invoke(cli, ["occ", "--help"], catch_exceptions=False)

    assert result.exit_code == 2
    assert "No such command 'occ'" in result.output


def test_positive_control_a_surviving_extension_resolves_by_name() -> None:
    """The unknown-command zero above is a lookup that can succeed."""
    result = CliRunner().invoke(cli, ["profile", "--help"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
