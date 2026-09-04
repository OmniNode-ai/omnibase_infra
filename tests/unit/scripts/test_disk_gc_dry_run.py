# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Dry-run safety smoke tests for the GC shell scripts (OMN-13008).

The single most important property: in DEFAULT (dry-run) mode the scripts must
NEVER call a destructive docker/git subcommand. We prove this by shimming `docker`,
`git`, and `rpk` on PATH with a recorder that logs every invocation, then asserting
no destructive verb (rmi, rm, prune, produce, 'worktree remove') was issued.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"

DESTRUCTIVE_TOKENS = ("rmi", "prune", "produce", "remove", "rm ")


def _make_shim(bin_dir: Path, name: str, calllog: Path) -> None:
    """Create an executable shim that records its argv and exits 0 with empty output."""
    shim = bin_dir / name
    shim.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "{name} $*" >> "{calllog}"\n'
        # image ls / ps queries must return empty so planner produces an empty plan.
        "exit 0\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _make_df_shim(bin_dir: Path, *, avail_gb: int) -> None:
    """Create a `df` shim reporting an exact free-space figure, in 1 KiB blocks.

    OMN-17872 made free space the halt criterion, so a test that wants to force a
    breach shims the MEASUREMENT rather than passing a throwaway threshold. The
    thresholds themselves are declared in scripts/disk-watermark-thresholds.json
    and are deliberately not settable per invocation.
    """
    total_kb = 1024 * 1024 * 1024  # 1 TiB
    avail_kb = avail_gb * 1024 * 1024
    used_kb = total_kb - avail_kb
    used_pct = round(used_kb * 100 / total_kb)
    shim = bin_dir / "df"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "cat <<'POSIXEOF'\n"
        "Filesystem     1024-blocks      Used Available Capacity Mounted on\n"
        f"/dev/disk1   {total_kb}  {used_kb}  {avail_kb}  {used_pct}% /\n"
        "POSIXEOF\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    script: str,
    args: list[str],
    tmp_path: Path,
    *,
    df_avail_gb: int | None = None,
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calllog = tmp_path / "calls.log"
    calllog.write_text("")
    for tool in ("docker", "git", "rpk"):
        _make_shim(bin_dir, tool, calllog)
    if df_avail_gb is not None:
        _make_df_shim(bin_dir, avail_gb=df_avail_gb)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)  # redirect log files into tmp
    # Ensure no ambient broker leaks in — fail-fast publish path must stay quiet.
    env.pop("KAFKA_BOOTSTRAP_SERVERS", None)
    proc = subprocess.run(
        ["bash", str(_SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    return proc, calllog.read_text()


@pytest.mark.unit
class TestGcDryRunSafety:
    def test_disk_gc_default_is_dry_run_no_destructive_calls(
        self, tmp_path: Path
    ) -> None:
        proc, calls = _run("disk-gc.sh", [], tmp_path)
        assert proc.returncode == 0, proc.stderr
        for tok in DESTRUCTIVE_TOKENS:
            assert tok not in calls, (
                f"dry-run issued destructive docker/git op: {tok!r}\n{calls}"
            )

    def test_watermark_dry_run_does_not_produce_to_bus(self, tmp_path: Path) -> None:
        # Force a breach by shimming the MEASUREMENT (1 GiB free, under the
        # declared 50 GiB floor) so the event path runs, but --dry-run must NOT
        # call `rpk ... produce`.
        proc, calls = _run(
            "disk-watermark-check.sh",
            ["--mount", "/", "--dry-run"],
            tmp_path,
            df_avail_gb=1,
        )
        # Below the free-space floor → exit 20, but dry-run means no publish.
        assert proc.returncode == 20, proc.stderr
        assert "produce" not in calls, f"dry-run published to bus:\n{calls}"
        assert '"severity": "critical"' in proc.stdout

    def test_watermark_breach_without_broker_does_not_publish(
        self, tmp_path: Path
    ) -> None:
        # Real (non-dry-run) breach but KAFKA_BOOTSTRAP_SERVERS unset → fail-fast:
        # the script must NOT fall back to a default broker and must NOT call produce.
        proc, calls = _run(
            "disk-watermark-check.sh",
            ["--mount", "/"],
            tmp_path,
            df_avail_gb=1,
        )
        assert proc.returncode == 20, proc.stderr
        assert "produce" not in calls, f"published with no broker configured:\n{calls}"

    def test_watermark_under_threshold_is_quiet(self, tmp_path: Path) -> None:
        # 900 GiB free on a 1 TiB volume: above both floors, 12% used, so quiet.
        proc, calls = _run(
            "disk-watermark-check.sh",
            ["--mount", "/"],
            tmp_path,
            df_avail_gb=900,
        )
        assert proc.returncode == 0, proc.stderr
        assert "produce" not in calls

    def test_worktree_gc_default_dry_run_passes_no_execute(
        self, tmp_path: Path
    ) -> None:
        # Point at an empty worktrees root and a stub prune script that records args.
        wt_root = tmp_path / "wt"
        wt_root.mkdir()
        prune = tmp_path / "prune-worktrees.sh"
        prune.write_text(
            "#!/usr/bin/env bash\n"
            f'echo "prune $*" >> "{tmp_path / "prune-calls.log"}"\n'
            "exit 0\n"
        )
        prune.chmod(0o755)
        proc, _ = _run(
            "worktree-gc.sh",
            ["--worktrees-root", str(wt_root), "--prune-script", str(prune)],
            tmp_path,
        )
        assert proc.returncode == 0, proc.stderr
        prune_calls = (tmp_path / "prune-calls.log").read_text()
        # Default mode must NOT pass --execute to the prune script.
        assert "--execute" not in prune_calls, prune_calls

    def test_worktree_gc_execute_forwards_execute_flag(self, tmp_path: Path) -> None:
        wt_root = tmp_path / "wt"
        wt_root.mkdir()
        prune = tmp_path / "prune-worktrees.sh"
        prune.write_text(
            "#!/usr/bin/env bash\n"
            f'echo "prune $*" >> "{tmp_path / "prune-calls.log"}"\n'
            "exit 0\n"
        )
        prune.chmod(0o755)
        proc, _ = _run(
            "worktree-gc.sh",
            [
                "--execute",
                "--worktrees-root",
                str(wt_root),
                "--prune-script",
                str(prune),
            ],
            tmp_path,
        )
        assert proc.returncode == 0, proc.stderr
        prune_calls = (tmp_path / "prune-calls.log").read_text()
        assert "--execute" in prune_calls, prune_calls
