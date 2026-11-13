# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pool's build phase names why the slot script exited non-zero (OMN-20413).

prepr_verify_lane.sh exits 10 when the workspace build fails or times out. The build
phase used to print only ``build rc=10`` for that exit, so a lab-proof-receipt FAIL read
as an unexplained stack build failure. The ``build rc`` line now carries the slot
script's REFUSED/FAILED reason and the end of its log follows it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

PROVE_SH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "runtime_build"
    / "prepr_pool_prove.sh"
)


def _arm_source() -> str:
    text = PROVE_SH.read_text(encoding="utf-8")
    arm = re.search(r"^\s*\*\) echo \"build rc=\$src .*$", text, re.M)
    assert arm is not None, "prepr_pool_prove.sh has no detailed build rc arm"
    return arm.group(0).strip().removeprefix("*)").removesuffix(";;")


def _run(work: Path, src: int) -> str:
    return subprocess.run(
        ["/bin/bash", "-c", f"W={work}; src={src}; {_arm_source()}"],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    ).stdout


@pytest.mark.unit
def test_build_failure_names_reason_and_log_tail(tmp_path: Path) -> None:
    (tmp_path / "build.log").write_text(
        "#12 ERROR: pip install failed\n"
        "[prepr-verify-lane] REFUSED/FAILED: the workspace build failed or timed out.\n",
        encoding="utf-8",
    )
    out = _run(tmp_path, 10)
    first = out.splitlines()[0]
    assert first.startswith("build rc=10 ")
    assert "the workspace build failed or timed out." in first
    assert "#12 ERROR: pip install failed" in out


@pytest.mark.unit
def test_build_failure_without_reason_line_still_reports_rc(tmp_path: Path) -> None:
    (tmp_path / "build.log").write_text("some output\n", encoding="utf-8")
    out = _run(tmp_path, 10)
    assert out.splitlines()[0].startswith("build rc=10")
    assert "some output" in out
