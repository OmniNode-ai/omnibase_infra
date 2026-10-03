# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18675 AC2: the co-install fails when an ``onex.cli`` entry point cannot load.

Step 4 of ``install-node-skill-package.sh`` loads every advertised ``onex.cli``
entry point and exits non-zero naming the ones that raise. The 2026-09-18
breakage was exactly this shape: an entry point whose module had vanished from
a downgraded dependency. These tests run the script's own step-4 program,
extracted from the script text, against a synthetic distribution carrying a
healthy and a broken entry point. Nothing is installed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "install-node-skill-package.sh"
)


def _step4_program() -> str:
    text = _SCRIPT.read_text(encoding="utf-8")
    marker = "== step 4: verify every advertised onex.cli entry point imports =="
    tail = text[text.index(marker) :]
    match = re.search(r"<<'PYEOF'\n(.*?)\nPYEOF\n", tail, re.DOTALL)
    assert match, "step 4 heredoc not found in install-node-skill-package.sh"
    return match.group(1)


def _site(root: Path, entry_points: dict[str, str]) -> Path:
    site = root / "site"
    dist = site / "fake_cli-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fake-cli\nVersion: 1.0\n", encoding="utf-8"
    )
    lines = ["[onex.cli]"] + [f"{k} = {v}" for k, v in entry_points.items()]
    (dist / "entry_points.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (site / "healthy_mod.py").write_text("def cmd():\n    return 1\n", encoding="utf-8")
    return site


def _run(site: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(site)
    return subprocess.run(
        [sys.executable, "-S", "-c", _step4_program()],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_a_broken_cli_entry_point_fails_the_install_naming_it(tmp_path: Path) -> None:
    site = _site(
        tmp_path,
        {
            "good": "healthy_mod:cmd",
            "occ": "vanished_module.contracts.pr_occ_stamp:occ",
        },
    )
    result = _run(site)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "the onex CLI does not load" in result.stderr
    assert "occ: ModuleNotFoundError" in result.stderr


def test_healthy_cli_entry_points_pass(tmp_path: Path) -> None:
    site = _site(tmp_path, {"good": "healthy_mod:cmd"})
    result = _run(site)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "onex.cli entry points import cleanly" in result.stdout


def test_step4_runs_before_the_success_readback() -> None:
    """The check must gate success: it precedes the step-5 readback in the script."""
    text = _SCRIPT.read_text(encoding="utf-8")
    assert text.index("== step 4:") < text.index("== step 5:")
