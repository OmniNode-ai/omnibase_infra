# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""scripts/fire-build-loop.sh runs the core CLI as a module, not the onex launcher (OMN-25070 AC3).

omnibase-core stops declaring the onex console script once the customer package omninode-onex carries it, so a script
that runs `uv run onex` would find no command. The script runs `uv run python -m omnibase_core.cli.cli_commands run
build_loop_workflow.yaml` with the same arguments.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "fire-build-loop.sh"
_LAUNCHER = re.compile(r"(^|[\s(\"'=])onex(\s|$)")


def _code_lines(text: str) -> list[str]:
    """Return non-comment, non-blank lines from *text*."""
    return [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_the_script_runs_no_onex_launcher() -> None:
    offenders = [
        line for line in _code_lines(SCRIPT.read_text()) if _LAUNCHER.search(line)
    ]
    assert offenders == []


def test_the_script_runs_the_build_loop_through_the_core_module() -> None:
    assert (
        "env ONEX_CLI_SHOW_ALL_COMMANDS=1 uv run python -m omnibase_core.cli.cli_commands run build_loop_workflow.yaml"
        in SCRIPT.read_text()
    )


def test_the_launcher_pattern_matches_a_launcher_line() -> None:
    assert _LAUNCHER.search(
        'CMD=(uv run onex run build_loop_workflow.yaml "${ARGS[@]}")'
    )
    assert _LAUNCHER.search("docker/onex-api") is None
