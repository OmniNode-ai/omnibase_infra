# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pre-job hook reaps the runner's TMPDIR and warns on its headroom (OMN-17427).

THE INCIDENT. On 2026-10-21 one lab host's CI runners' TMPDIR (a per-runner bind
mount on the host's scratch disk) held over 120,000 test sandboxes that nothing
ever removed, about 1.7 million inodes per runner. The scratch disk reached 100
percent of its inodes at 77 percent of its bytes, every job that wrote a temp
file failed with ENOSPC, and a private repository's main went red, which stopped
its merge queue for five hours. Nothing warned: the admission gate reads bytes
on the workspace mount only.

THE MECHANISM UNDER TEST. Before each job (one runner runs one job at a time,
and each runner has a TMPDIR of its own), the hook removes the TMPDIR entries
older than RUNNER_TMP_REAP_MINUTES, bounded in time and fail-open, and never
touches a TMPDIR that is not a dedicated one (/tmp, /, a home directory). It
then reads byte and inode use of the TMPDIR mount and emits a ::warning
annotation naming the runner and both numbers at or past 85 percent, so every
job on a filling runner says so on its own page.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-job-started.sh"
TEN_GB_KB = 10 * 5 * 1024 * 1024


def _has_gnu_realpath_m() -> bool:
    try:
        result = subprocess.run(
            ["realpath", "-m", "."],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not _has_gnu_realpath_m(),
        reason="runner-job-started.sh requires GNU `realpath -m` (the runner image and Linux CI have it)",
    ),
]


def _age(path: Path, minutes: int) -> None:
    then = time.time() - minutes * 60
    os.utime(path, (then, then))


def _run_hook(
    tmp_path: Path,
    tmpdir: Path | None,
    *,
    reap_minutes: int = 60,
    df_shim: str | None = None,
) -> subprocess.CompletedProcess[str]:
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "some-repo" / "some-repo"
    workspace.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "RUNNER_HOME": str(runner_home),
            "GITHUB_WORKSPACE": str(workspace),
            "RUNNER_NAME": "ci-runner-3",
            "DISK_ADMISSION_DF_OVERRIDE_KB": str(TEN_GB_KB),
            "OMNI_GIT_MIRROR_RUNNERS": "NONE",
            "RUNNER_TMP_REAP_MINUTES": str(reap_minutes),
        }
    )
    if tmpdir is None:
        env.pop("TMPDIR", None)
    else:
        env["TMPDIR"] = str(tmpdir)
    if df_shim is not None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        df = bin_dir / "df"
        df.write_text(df_shim)
        df.chmod(0o755)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(HOOK_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_stale_tmpdir_entries_are_removed_and_fresh_ones_kept(tmp_path: Path) -> None:
    tmpdir = tmp_path / "runner-tmp"
    tmpdir.mkdir()
    stale = tmpdir / "prh-old123"
    (stale / "docs" / "tracking").mkdir(parents=True)
    (stale / "docs" / "tracking" / "ROLLING_WORK_LEDGER.md").write_text("x")
    _age(stale, 120)
    stale_file = tmpdir / "omni-ledger-write-args-abc"
    stale_file.write_text("x")
    _age(stale_file, 120)
    fresh = tmpdir / "prh-new456"
    fresh.mkdir()
    pytest_root = tmpdir / "pytest-of-runner"
    pytest_root.mkdir()
    _age(pytest_root, 600)

    result = _run_hook(tmp_path, tmpdir)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not stale.exists()
    assert not stale_file.exists()
    assert fresh.is_dir(), "an entry younger than the reap age is kept"
    assert pytest_root.is_dir(), (
        "pytest prunes its own numbered dirs; the reaper leaves them"
    )
    assert "[runner-tmp] reaped 2 of 4 entries" in result.stdout


def test_a_shared_tmp_is_never_reaped(tmp_path: Path) -> None:
    """TMPDIR=/tmp (the image bakes omni-ci-bin there) or any non-dedicated dir is refused."""
    shared = Path("/") / "tmp"  # the shared /tmp is exactly what must be refused
    result = _run_hook(tmp_path, shared)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "[runner-tmp] /tmp is not a dedicated runner TMPDIR; not reaping"
        in result.stdout
    )


def test_an_unset_tmpdir_is_a_quiet_no_op(tmp_path: Path) -> None:
    result = _run_hook(tmp_path, None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reaped" not in result.stdout


def _df_shim(byte_pct: int, inode_pct: int) -> str:
    return (
        "#!/usr/bin/env bash\n"
        'if [[ " $* " == *" -Pi "* ]]; then\n'
        "  echo 'Filesystem Inodes IUsed IFree IUse% Mounted on'\n"
        f"  echo '/dev/sda1 1000 {inode_pct * 10} {1000 - inode_pct * 10} {inode_pct}% /scratch'\n"
        "  exit 0\n"
        "fi\n"
        "echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
        f"echo '/dev/sda1 1000 {byte_pct * 10} {1000 - byte_pct * 10} {byte_pct}% /scratch'\n"
    )


def test_inodes_past_85_percent_raise_a_warning_annotation(tmp_path: Path) -> None:
    tmpdir = tmp_path / "runner-tmp"
    tmpdir.mkdir()
    result = _run_hook(tmp_path, tmpdir, df_shim=_df_shim(77, 100))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning title=RUNNER-TMP-HEADROOM::" in result.stdout
    assert "ci-runner-3" in result.stdout
    assert "inodes 100% used" in result.stdout
    assert "bytes 77% used" in result.stdout


def test_bytes_past_85_percent_raise_a_warning_annotation(tmp_path: Path) -> None:
    tmpdir = tmp_path / "runner-tmp"
    tmpdir.mkdir()
    result = _run_hook(tmp_path, tmpdir, df_shim=_df_shim(90, 10))
    assert "::warning title=RUNNER-TMP-HEADROOM::" in result.stdout


def test_headroom_under_85_percent_is_quiet(tmp_path: Path) -> None:
    tmpdir = tmp_path / "runner-tmp"
    tmpdir.mkdir()
    result = _run_hook(tmp_path, tmpdir, df_shim=_df_shim(77, 84))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RUNNER-TMP-HEADROOM" not in result.stdout
