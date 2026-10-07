# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for bounded reconcile-host execution (OMN-19709)."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_lane_hook_arming_omn18260 import (
    _teach_dispatch_python_to_run_programs,
)
from tests.scripts.test_reconcile_workspace_venvs import _Workspace

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENV_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"

EXIT_DECLINED = 4
EXIT_STEP_TIMEOUT = 5
EXIT_STALE_LIVE_HOLDER = 6


def test_command_output_is_never_fed_to_read_through_a_here_string() -> None:
    for script in (_VENV_SCRIPT,):
        source = script.read_text(encoding="utf-8")
        assert "<<<" not in source, (
            f"{script.name} still contains a here-string. Bash may fill the "
            "here-string pipe before starting its reader and deadlock when "
            "captured command output exceeds the pipe capacity."
        )


def test_lane_hook_report_handles_200_kib_status_output_within_30_seconds(
    tmp_path: Path,
) -> None:
    ws = _Workspace(tmp_path / "omni_home")
    _teach_dispatch_python_to_run_programs(ws)
    lane_identity = ws.root / "omniclaude" / "scripts" / "lane_identity.py"
    lane_identity.parent.mkdir(parents=True, exist_ok=True)
    lane_identity.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if sys.argv[1:] == ['status']:\n"
        "    sys.stdout.write(('x' * 100 + '\\n') * 2048)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    lane_identity.chmod(0o755)
    fake_home = ws.root / "fakehome"
    fake_home.mkdir()
    env = ws.env()
    env["HOME"] = str(fake_home)
    env["ONEX_LANE_IDENTITY_SCRIPT"] = str(lane_identity)

    started = time.monotonic()
    result = subprocess.run(
        ["bash", str(_VENV_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )

    assert time.monotonic() - started < 30
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("x" * 100) == 2048
