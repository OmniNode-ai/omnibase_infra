# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pool's build phase stops at the local-env step that failed and names it (OMN-19998).

prepr_pool_prove.sh used to pipe ``make local-env`` through ``tail -2`` and go on. On a
host without make (or when the recipe failed) the compose build then died on a missing
env file and the readback carried only ``build rc=1``, so a lab-proof-receipt red on an
omnimarket PR read as an unexplained stack build failure. the build step now stops at
the step and prints the cause on a ``build rc=1`` line, the line the receipt keeps.
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


def _step_source() -> str:
    text = PROVE_SH.read_text(encoding="utf-8")
    step = re.search(r"^\s*lenv=\$\(make local-env .*$", text, re.M)
    assert step is not None, (
        "prepr_pool_prove.sh has no local-env step that checks make"
    )
    return step.group(0)


def _run(work: Path, bin_dir: Path | None) -> subprocess.CompletedProcess[str]:
    path = f"{bin_dir}:/usr/bin:/bin" if bin_dir else "/nonexistent-bin"
    # `bash` is named by absolute path so an empty PATH still finds the shell
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            f"set -u; W={work}; {_step_source()}",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": path},
    )


def _fake_make(bin_dir: Path, body: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    make = bin_dir / "make"
    make.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    make.chmod(0o755)


@pytest.mark.unit
def test_a_host_without_make_stops_at_local_env_and_says_so(tmp_path: Path) -> None:
    # the failing control: before the fix the step went on, so the build phase
    # reached the compose build with no env file and reported only "build rc=1"
    done = _run(tmp_path, bin_dir=None)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "build rc=1 make local-env failed (exit 127)" in done.stdout, done.stdout


@pytest.mark.unit
def test_a_failing_recipe_stops_at_local_env_with_its_last_lines(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    _fake_make(bin_dir, "echo 'ERROR: openssl is required'; exit 1")
    done = _run(tmp_path, bin_dir)
    assert done.returncode == 1, done.stdout + done.stderr
    assert (
        "build rc=1 make local-env failed (exit 1): ERROR: openssl is required"
        in done.stdout
    ), done.stdout


@pytest.mark.unit
def test_a_recipe_that_exits_zero_without_writing_the_files_still_stops(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    _fake_make(bin_dir, "exit 0")
    done = _run(tmp_path, bin_dir)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "build rc=1 make local-env failed (exit 0)" in done.stdout, done.stdout


@pytest.mark.unit
def test_a_recipe_that_writes_both_files_passes_silently(tmp_path: Path) -> None:
    # the positive control: the step still passes when the files are written
    bin_dir = tmp_path / "bin"
    _fake_make(
        bin_dir,
        f"echo x > {tmp_path}/local.env; echo y > {tmp_path}/local.bifrost.yaml",
    )
    done = _run(tmp_path, bin_dir)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "build rc=" not in done.stdout


@pytest.mark.unit
def test_the_build_phase_no_longer_pipes_make_through_tail() -> None:
    text = PROVE_SH.read_text(encoding="utf-8")
    assert not re.search(r"^\s*make local-env .*\| tail", text, re.M)
