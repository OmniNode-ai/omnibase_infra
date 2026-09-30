# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""pull-all.sh fails fast when OMNI_HOME is unset (OMN-18971, rules 6 and 8).

The script used to default OMNI_HOME to a machine-specific volume path. On any
other host that silently pointed the whole sync at a directory that does not
exist there, and reported the repos as absent instead of failing. This test is
its own file so the git-env-scrub guard judges only what it adds.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

PULL_ALL = Path(__file__).resolve().parents[3] / "scripts" / "pull-all.sh"


@pytest.mark.unit
def test_unset_omni_home_fails_fast_and_names_the_variable(tmp_path: Path) -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "OMNI_HOME" and not k.startswith("GIT_")
    }
    env["HOME"] = str(tmp_path)
    proc = subprocess.run(
        ["bash", str(PULL_ALL), "omniclaude"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "OMNI_HOME" in proc.stderr, proc.stderr


@pytest.mark.unit
def test_script_carries_no_machine_specific_default() -> None:
    text = PULL_ALL.read_text(encoding="utf-8")
    assert "/Volumes" + "/" not in text
    assert "/Users" + "/" not in text
