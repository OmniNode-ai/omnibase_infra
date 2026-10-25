# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""runner-image-multiarch-smoke retries a QEMU emulator crash once, and nothing else (OMN-20413).

On 2026-10-25 omnibase_infra#5312, which changed one line of pyproject.toml,
went red in runner-image-multiarch-smoke: the arm64 libc-bin trigger died with
``qemu: uncaught target signal 11`` on the same binfmt digest and runner image
that built green for #5230 eleven hours earlier. The cross-build step now
retries once when its log carries that QEMU signature. Any other failure must
still fail on the first attempt, and a crash on both attempts must still fail.

These tests run the step's own ``run:`` script in bash against a stub
``docker`` that records each call, so they exercise the shipped script.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runner-image-build-smoke.yml"
JOB = "runner-image-multiarch-smoke"
STEP = "Cross-build the architecture-sensitive stage for amd64 + arm64"
QEMU_CRASH = (
    "#28 38.78 qemu: uncaught target signal 11 (Segmentation fault) - core dumped"
)
OTHER_FAILURE = (
    "#12 ERROR: failed to fetch https://example.invalid/runner.tar.gz: 404 Not Found"
)


def _step_script() -> str:
    document: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for step in document["jobs"][JOB]["steps"]:
        if step.get("name") == STEP:
            return str(step["run"])
    raise AssertionError(f"{JOB} has no step named {STEP!r}")


def _run(tmp_path: Path, outcomes: list[tuple[int, str]]) -> tuple[int, str, int]:
    """Run the step with a stub docker that plays ``outcomes`` in order.

    Returns the step's exit code, its output, and how many times docker ran.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    calls.write_text("", encoding="utf-8")
    for index, (code, line) in enumerate(outcomes, 1):
        (tmp_path / f"outcome-{index}").write_text(
            f"{code}\n{line}\n", encoding="utf-8"
        )
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{calls}"\n'
        f'n=$(wc -l < "{calls}" | tr -d " ")\n'
        f'outcome="{tmp_path}/outcome-${{n}}"\n'
        '[[ -f "${outcome}" ]] || { echo "stub docker: no outcome ${n}" >&2; exit 99; }\n'
        'code=$(sed -n 1p "${outcome}")\n'
        'sed -n 2p "${outcome}" >&2\n'
        'exit "${code}"\n',
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(runner_temp),
    }
    result = subprocess.run(
        ["bash", "-c", _step_script()],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    attempts = len(calls.read_text(encoding="utf-8").splitlines())
    return result.returncode, result.stdout + result.stderr, attempts


def test_a_clean_build_passes_on_the_first_attempt(tmp_path: Path) -> None:
    code, output, attempts = _run(tmp_path, [(0, "#30 DONE")])
    assert (code, attempts) == (0, 1), output
    assert "multi-arch smoke PASSED" in output


def test_a_qemu_crash_is_retried_once_and_a_clean_retry_passes(tmp_path: Path) -> None:
    code, output, attempts = _run(tmp_path, [(1, QEMU_CRASH), (0, "#30 DONE")])
    assert (code, attempts) == (0, 2), output
    assert "the QEMU emulator crashed" in output


def test_a_qemu_crash_on_both_attempts_still_fails(tmp_path: Path) -> None:
    code, output, attempts = _run(tmp_path, [(1, QEMU_CRASH), (1, QEMU_CRASH)])
    assert (code, attempts) == (1, 2), output
    assert "crashed on both attempts" in output


def test_a_failure_that_is_not_a_qemu_crash_is_not_retried(tmp_path: Path) -> None:
    code, output, attempts = _run(tmp_path, [(1, OTHER_FAILURE), (0, "#30 DONE")])
    assert (code, attempts) == (1, 1), output
    assert "not a QEMU emulator crash" in output


def test_every_attempt_builds_both_platforms_into_one_discarded_manifest(
    tmp_path: Path,
) -> None:
    _run(tmp_path, [(1, QEMU_CRASH), (0, "#30 DONE")])
    calls = (tmp_path / "calls").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    for call in calls:
        assert call.startswith("buildx build ")
        assert "--platform linux/amd64,linux/arm64" in call
        assert "--target toolchain" in call
        assert "--output type=cacheonly" in call
