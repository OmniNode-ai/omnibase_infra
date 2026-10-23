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


@pytest.mark.parametrize(("byte_pct", "inode_pct"), [(85, 10), (77, 85)])
def test_headroom_at_exactly_85_percent_warns(
    tmp_path: Path, byte_pct: int, inode_pct: int
) -> None:
    tmpdir = tmp_path / "runner-tmp"
    tmpdir.mkdir()
    result = _run_hook(tmp_path, tmpdir, df_shim=_df_shim(byte_pct, inode_pct))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning title=RUNNER-TMP-HEADROOM::" in result.stdout


@pytest.mark.parametrize("inode_only", [False, True])
def test_failed_df_is_reported_without_stopping_the_job(
    tmp_path: Path, inode_only: bool
) -> None:
    tmpdir = tmp_path / "runner-tmp"
    tmpdir.mkdir()
    shim = "#!/usr/bin/env bash\nexit 1\n"
    if inode_only:
        shim = _df_shim(90, 10).replace(
            'if [[ " $* " == *" -Pi "* ]]; then\n',
            'if [[ " $* " == *" -Pi "* ]]; then\nexit 1\n',
        )
    result = _run_hook(tmp_path, tmpdir, df_shim=shim)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "headroom unreadable:" in result.stdout
    assert "inodes unknown%" in result.stdout
    assert ("RUNNER-TMP-HEADROOM" in result.stdout) == inode_only


def test_unset_tmpdir_emits_no_headroom_warning(tmp_path: Path) -> None:
    result = _run_hook(tmp_path, None, df_shim=_df_shim(100, 100))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RUNNER-TMP-HEADROOM" not in result.stdout
    assert "[runner-tmp]" not in result.stdout


def test_home_and_symlinked_shared_tmpdir_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    stale = home_dir / "keep"
    stale.write_text("keep")
    _age(stale, 120)
    monkeypatch.setenv("HOME", str(home_dir))
    home_result = _run_hook(tmp_path, home_dir)
    assert home_result.returncode == 0, home_result.stdout + home_result.stderr
    assert "not a dedicated runner TMPDIR" in home_result.stdout
    assert stale.read_text() == "keep"

    link = tmp_path / "shared-link"
    link.symlink_to(Path("/") / "tmp", target_is_directory=True)
    result = _run_hook(tmp_path, link, reap_minutes=99999999)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[runner-tmp] /tmp is not a dedicated runner TMPDIR" in result.stdout


def test_reap_dir_override_and_symlink_target_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = tmp_path / "default-tmp"
    selected = tmp_path / "selected-tmp"
    default.mkdir()
    selected.mkdir()
    keep = default / "keep"
    keep.write_text("keep")
    _age(keep, 120)
    stale_link = selected / "stale-link"
    stale_link.symlink_to(keep)
    then = time.time() - 120 * 60
    os.utime(stale_link, (then, then), follow_symlinks=False)
    monkeypatch.setenv("RUNNER_TMP_REAP_DIR", str(selected))
    result = _run_hook(tmp_path, default)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not stale_link.is_symlink()
    assert keep.read_text() == "keep"
    assert "reaped 1 of 1 entries" in result.stdout


def test_entries_owned_by_another_uid_are_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmpdir = tmp_path / "runner-tmp"
    tmpdir.mkdir()
    stale = tmpdir / "other-owner"
    stale.write_text("keep")
    _age(stale, 120)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Present the reaper as a different uid without needing privileged chown.
    uid = bin_dir / "id"
    uid.write_text("#!/usr/bin/env bash\necho 2147483646\n")
    uid.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    result = _run_hook(tmp_path, tmpdir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert stale.read_text() == "keep"
    assert "reaped 0 of 1 entries" in result.stdout


def test_spent_budget_skips_reaping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmpdir = tmp_path / "runner-tmp"
    tmpdir.mkdir()
    stale = tmpdir / "keep"
    stale.write_text("keep")
    _age(stale, 120)
    monkeypatch.setenv("OMNI_GIT_MIRROR_HOOK_BUDGET", "0")
    result = _run_hook(tmp_path, tmpdir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hook budget spent; reap" in result.stdout
    assert stale.read_text() == "keep"


def test_slow_counting_is_inside_the_reap_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmpdir = tmp_path / "runner-tmp"
    tmpdir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    find = bin_dir / "find"
    find.write_text("#!/usr/bin/env bash\nexec sleep 10\n")
    find.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("RUNNER_TMP_REAP_SECONDS", "1")
    started = time.monotonic()
    result = _run_hook(tmp_path, tmpdir, df_shim=_df_shim(10, 10))
    assert result.returncode == 0, result.stdout + result.stderr
    assert time.monotonic() - started < 5, result.stdout + result.stderr
