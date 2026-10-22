# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The disk watermark guard also reads inode use (OMN-17427).

WHY THIS EXISTS. On 2026-10-21 a lab host's CI runner scratch filesystem
(``/scratch``, which holds every runner's TMPDIR and job workspace) reported
77 percent of its bytes used and 100 percent of its inodes used. Every write
failed with ENOSPC ("No space left on device") while ``df -h`` looked healthy,
so nothing alarmed: the guard read bytes only, and it was never pointed at
``/scratch`` at all. The pr-handoff sandbox tests failed on the runners and
a private repository's main went red, which stopped its merge queue.

The guard now warns when inode use reaches the same declared advisory
percentage (``warn_pct``, 85) as byte use, and ``--mount-optional`` lets one
unit check a mount that exists on some hosts only (``/scratch`` on the CI runner host).
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
_KIB_PER_GIB = 1024 * 1024
_TOTAL_KB = 1_900_000_000
_TOTAL_INODES = 122_101_760


def _shim(bin_dir: Path, *, avail_gb: int, inode_pct: int | None) -> None:
    used_kb = _TOTAL_KB - avail_gb * _KIB_PER_GIB
    used_pct = round(used_kb * 100 / _TOTAL_KB)
    inode_block = ""
    if inode_pct is not None:
        iused = _TOTAL_INODES * inode_pct // 100
        inode_block = (
            'if [[ " $* " == *" -Pi "* ]]; then\n'
            "cat <<'POSIXEOF'\n"
            "Filesystem        Inodes     IUsed    IFree IUse% Mounted on\n"
            f"/dev/sda1      {_TOTAL_INODES} {iused} {_TOTAL_INODES - iused} {inode_pct}% /scratch\n"
            "POSIXEOF\n"
            "exit 0\n"
            "fi\n"
        )
    df = bin_dir / "df"
    df.write_text(
        "#!/usr/bin/env bash\n"
        + inode_block
        + 'if [[ " $* " != *" -Pk "* ]]; then echo "df shim: refused $*" >&2; exit 64; fi\n'
        "cat <<'POSIXEOF'\n"
        "Filesystem     1024-blocks      Used Available Capacity Mounted on\n"
        f"/dev/sda1   {_TOTAL_KB}  {used_kb}  {avail_gb * _KIB_PER_GIB}  {used_pct}% /scratch\n"
        "POSIXEOF\n"
    )
    df.chmod(df.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    host = bin_dir / "hostname"
    host.write_text("#!/usr/bin/env bash\necho ci-host\n")
    host.chmod(0o755)


def _run(
    tmp_path: Path,
    *,
    avail_gb: int,
    inode_pct: int | None,
    mount: str,
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _shim(bin_dir, avail_gb=avail_gb, inode_pct=inode_pct)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env.pop("KAFKA_BOOTSTRAP_SERVERS", None)
    env.pop("ONEX_BUS_PUBLISH_URL", None)
    return subprocess.run(
        [
            "bash",
            str(_SCRIPTS / "disk-watermark-check.sh"),
            "--mount",
            mount,
            "--dry-run",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )


def _event(proc: subprocess.CompletedProcess[str]) -> dict[str, object]:
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert lines, proc.stdout + proc.stderr
    return json.loads(lines[-1])


@pytest.mark.unit
class TestDiskWatermarkInodes:
    def test_inodes_at_100_percent_warn_while_bytes_look_healthy(
        self, tmp_path: Path
    ) -> None:
        """The 2026-10-21 shape: 422 GiB free, 77% of bytes, 100% of inodes."""
        proc = _run(tmp_path, avail_gb=422, inode_pct=100, mount=str(tmp_path))
        assert proc.returncode == 10, proc.stderr
        event = _event(proc)
        assert event["severity"] == "warning"
        assert event["halt_reason"] == "inode_used_pct_advisory"
        assert event["inode_used_pct"] == 100
        assert "inodes 100% used" in str(event["message"])

    def test_inodes_at_the_declared_85_percent_warn(self, tmp_path: Path) -> None:
        proc = _run(tmp_path, avail_gb=422, inode_pct=85, mount=str(tmp_path))
        assert proc.returncode == 10, proc.stderr
        assert _event(proc)["halt_reason"] == "inode_used_pct_advisory"

    def test_inodes_under_85_percent_stay_quiet(self, tmp_path: Path) -> None:
        proc = _run(tmp_path, avail_gb=422, inode_pct=84, mount=str(tmp_path))
        assert proc.returncode == 0, proc.stderr
        assert "inode_used_pct=84%" in proc.stderr

    def test_free_space_below_the_floor_still_outranks_inodes(
        self, tmp_path: Path
    ) -> None:
        proc = _run(tmp_path, avail_gb=20, inode_pct=100, mount=str(tmp_path))
        assert proc.returncode == 20, proc.stderr
        event = _event(proc)
        assert event["severity"] == "critical"
        assert event["halt_reason"] == "free_space_below_crit_floor"
        assert event["inode_used_pct"] == 100

    def test_unreadable_inode_use_is_logged_and_bytes_still_decide(
        self, tmp_path: Path
    ) -> None:
        """A df without inode support (the shim refuses -Pi) must not abort the guard."""
        proc = _run(tmp_path, avail_gb=422, inode_pct=None, mount=str(tmp_path))
        assert proc.returncode == 0, proc.stderr
        assert "inode use unreadable" in proc.stderr

    def test_an_optional_mount_that_is_absent_is_a_quiet_no_op(
        self, tmp_path: Path
    ) -> None:
        """The hourly unit checks /scratch on every host; a host without it must not fall back to /."""
        absent = tmp_path / "no-such-mount"
        proc = _run(
            tmp_path,
            avail_gb=20,
            inode_pct=100,
            mount=str(absent),
            extra=("--mount-optional",),
        )
        assert proc.returncode == 0, proc.stderr
        assert "not present on this host" in proc.stderr
        assert "{" not in proc.stdout

    def test_the_hourly_unit_checks_scratch_as_an_optional_mount(self) -> None:
        unit = (_REPO / "deploy" / "disk-gc" / "onex-disk-gc.service").read_text(
            encoding="utf-8"
        )
        assert 'disk-watermark-check.sh" --mount /scratch --mount-optional' in unit
