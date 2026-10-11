# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression checks for the local bring-up instructions (OMN-19477)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.docker.catalog import cli

_ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "relative_path", ["scripts/generate-local-env.sh", ".env.example"]
)
def test_bringup_instructions_name_the_make_entrypoint(relative_path: str) -> None:
    """Neither the generator nor the template may direct users to a missing CLI."""
    text = (_ROOT / relative_path).read_text(encoding="utf-8")
    assert "onex up" not in text
    assert "make up" in text


def test_generator_prints_a_runnable_next_step(tmp_path: Path) -> None:
    """Exercise the closing instruction and the generated file without logging secrets."""
    env_file = tmp_path / "local.env"
    result = subprocess.run(
        ["bash", str(_ROOT / "scripts/generate-local-env.sh"), str(env_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Next: make up" in result.stdout
    assert "infra-up" not in result.stdout
    assert "onex up" not in env_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("target", "bundle"),
    [
        ("down", "core"),
        ("down-auth", "auth"),
        ("down-runtime", "runtime"),
        ("down-all", "runtime"),
    ],
)
def test_teardown_help_describes_project_wide_behavior(
    target: str, bundle: str
) -> None:
    """Bundle arguments never select services in the existing cmd_down implementation."""
    with (
        patch.object(cli, "_load_stack_env", return_value=0),
        patch.object(cli, "_load_stack_env_file", return_value=None),
        patch.object(cli.subprocess, "run") as run,
    ):
        run.return_value.returncode = 0
        assert cli.cmd_down([bundle]) == 0
        run.assert_called_once_with(
            ["docker", "compose", "-f", cli._DEFAULT_OUTPUT, "down"],
            cwd=str(cli._REPO_ROOT),
            check=False,
        )

    result = subprocess.run(
        ["make", "help"], cwd=_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    description = next(
        line.strip().split(maxsplit=1)[1]
        for line in result.stdout.splitlines()
        if line.strip().split(maxsplit=1)[0] == target
    )
    assert "last generated compose project" in description
    assert "keeps volumes" in description
    for relative_path in ("Makefile", "README.md"):
        assert "core bundle ONLY" not in (_ROOT / relative_path).read_text(
            encoding="utf-8"
        )
