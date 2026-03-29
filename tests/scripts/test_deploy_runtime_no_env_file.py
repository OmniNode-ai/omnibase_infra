# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression guard: deploy-runtime.sh must not use --env-file.

F65 / OMN-6910: The old setup_env() approach copied ~/.omnibase/.env into a
stale snapshot and then passed --env-file to docker compose. This caused env
var changes to be silently ignored until the next full redeploy.

The fix sources ~/.omnibase/.env at script top and lets docker compose
resolve ${VAR} from the shell environment directly -- no --env-file needed.

These tests ensure the anti-pattern is never reintroduced.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DEPLOY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy-runtime.sh"


def _read_script_lines() -> list[str]:
    """Read deploy-runtime.sh, stripping comment-only lines."""
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


@pytest.mark.unit
def test_no_env_file_flag_in_active_code() -> None:
    """deploy-runtime.sh must not pass --env-file to docker compose."""
    lines = _read_script_lines()
    violations = [(i + 1, line) for i, line in enumerate(lines) if "--env-file" in line]
    assert violations == [], (
        f"Found --env-file in non-comment lines of deploy-runtime.sh: {violations}"
    )


@pytest.mark.unit
def test_no_env_file_args_variable() -> None:
    """deploy-runtime.sh must not declare env_file_args arrays."""
    lines = _read_script_lines()
    violations = [
        (i + 1, line) for i, line in enumerate(lines) if "env_file_args" in line
    ]
    assert violations == [], (
        f"Found env_file_args in non-comment lines of deploy-runtime.sh: {violations}"
    )


@pytest.mark.unit
def test_no_setup_env_function() -> None:
    """deploy-runtime.sh must not define a setup_env() function."""
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    # Match function definition: setup_env() { (with possible whitespace)
    assert not re.search(r"^setup_env\s*\(\)", text, re.MULTILINE), (
        "setup_env() function definition found in deploy-runtime.sh -- must not exist (F65)"
    )


@pytest.mark.unit
def test_sources_omnibase_env_at_top() -> None:
    """deploy-runtime.sh must source ~/.omnibase/.env early in the script."""
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    # Find the source line (not in a comment)
    match = re.search(
        r'^source\s+["\']?\$\{?HOME\}?/\.omnibase/\.env["\']?',
        text,
        re.MULTILINE,
    )
    assert match is not None, (
        "deploy-runtime.sh must source ~/.omnibase/.env "
        "(expected 'source ${HOME}/.omnibase/.env' or similar near top of script)"
    )
    # Ensure it appears in the first 50 lines
    line_number = text[: match.start()].count("\n") + 1
    assert line_number <= 50, (
        f"source ~/.omnibase/.env found at line {line_number}, "
        "expected within first 50 lines of script"
    )
