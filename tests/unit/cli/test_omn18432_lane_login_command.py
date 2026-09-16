# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18432 AC3: a first-class command stores the lane identity, from stdin.

The alternative this closes is not a worse command; it is no command, which in
practice means an operator or an agent hand-editing ``~/.onex/config.yaml`` and
``~/.onex/credentials.json``. A hand-edited credential file is how the 0600
mode and the reference-only rule stop being enforced at all: both are checks
the store performs, and nothing performs them on a file written by a text
editor.

Stdin, never an argv flag, for the reason ``onex auth login`` already records
and OMN-18296 re-proved on a different surface: a value on argv is visible in
the process table to every other process on the host, and lands in shell
history and exec logs -- three durable copies that outlive the session.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from omnibase_infra.cli.cli_auth import auth_group

pytestmark = pytest.mark.unit


@pytest.fixture
def onex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the commands at a real directory instead of the operator's own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home / ".onex"


def test_lane_login_writes_the_reference_and_the_owner_only_secret(
    onex_home: Path,
) -> None:
    result = CliRunner().invoke(
        auth_group,
        [
            "lane-login",
            "--lane",
            "dev",
            "--sasl-username",
            "dev-cli-host",
            "--sasl-password-stdin",
        ],
        input="s3kr3t-value\n",
    )

    assert result.exit_code == 0, result.output

    document = yaml.safe_load((onex_home / "config.yaml").read_text())
    assert document["lanes"]["dev"]["sasl_username"] == "dev-cli-host"
    assert document["lanes"]["dev"]["sasl_password_ref"]
    assert "sasl_password" not in document["lanes"]["dev"]

    credentials_path = onex_home / "credentials.json"
    assert stat.S_IMODE(credentials_path.stat().st_mode) == 0o600
    assert "s3kr3t-value" in json.loads(credentials_path.read_text()).values()


def test_lane_login_never_echoes_the_value(onex_home: Path) -> None:
    result = CliRunner().invoke(
        auth_group,
        [
            "lane-login",
            "--lane",
            "dev",
            "--sasl-username",
            "dev-cli-host",
            "--sasl-password-stdin",
        ],
        input="s3kr3t-value\n",
    )

    assert "s3kr3t-value" not in result.output


def test_lane_login_refuses_an_empty_stdin(onex_home: Path) -> None:
    result = CliRunner().invoke(
        auth_group,
        [
            "lane-login",
            "--lane",
            "dev",
            "--sasl-username",
            "dev-cli-host",
            "--sasl-password-stdin",
        ],
        input="",
    )

    assert result.exit_code != 0
    assert not (onex_home / "credentials.json").exists()


def test_lane_login_requires_the_stdin_flag_rather_than_taking_a_value(
    onex_home: Path,
) -> None:
    """There is deliberately no ``--sasl-password <value>`` form to fall back to."""
    result = CliRunner().invoke(
        auth_group,
        ["lane-login", "--lane", "dev", "--sasl-username", "dev-cli-host"],
    )

    assert result.exit_code != 0
    assert "--sasl-password-stdin" in result.output


def test_no_option_on_the_command_accepts_the_password_as_a_value() -> None:
    """The argv form must not exist at all, not merely be discouraged."""
    lane_login = auth_group.get_command(None, "lane-login)".rstrip(")"))
    assert lane_login is not None
    value_taking = [
        option
        for option in lane_login.params
        for name in option.opts
        if "password" in name and not getattr(option, "is_flag", False)
    ]
    assert value_taking == []


def test_lane_logout_removes_both_halves(onex_home: Path) -> None:
    runner = CliRunner()
    runner.invoke(
        auth_group,
        [
            "lane-login",
            "--lane",
            "dev",
            "--sasl-username",
            "dev-cli-host",
            "--sasl-password-stdin",
        ],
        input="s3kr3t-value\n",
    )

    result = runner.invoke(auth_group, ["lane-logout", "--lane", "dev"])

    assert result.exit_code == 0, result.output
    document = yaml.safe_load((onex_home / "config.yaml").read_text())
    assert "dev" not in document.get("lanes", {})
    assert json.loads((onex_home / "credentials.json").read_text()) == {}
