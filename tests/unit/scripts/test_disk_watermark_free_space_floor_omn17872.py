# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Absolute free-space floor for the disk watermark guard (OMN-17872).

WHY THIS EXISTS. Before OMN-17872 the guard halted on percent-used alone
(``WARN_PCT=85`` / ``CRIT_PCT=90``). ``AVAIL_KB`` was measured and shipped in the
event payload but never compared against anything. On a 3.6 TiB volume that made
"critical" fire with 276 GiB still free, and it stopped a real lane at
2026-09-04T00:52:44Z with 87 GiB free.

Percent-used is a ratio; admission is an absolute question — does the next unit
of work fit. The halt criterion is therefore free space in GiB. The percentage
survives only as a secondary warning and can never, on its own, produce the
critical exit code.

The numbers are declared in ``scripts/disk-watermark-thresholds.json`` — the
guard's own config, not an environment variable (Rule 8).
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
_THRESHOLDS = _SCRIPTS / "disk-watermark-thresholds.json"

_KIB_PER_GIB = 1024 * 1024

# The live volume this guard runs against on the Mac, measured 2026-09-04T10:35Z:
#   df -h /System/Volumes/Data -> 3.6Ti size, 3.3Ti used, 276Gi avail, 93% capacity
_MAC_DATA_TOTAL_KB = 3_906_988_032  # ~3.64 TiB
_ADMITTED_AVAIL_GB = 276
_REFUSED_AVAIL_GB = 20


def _make_df_shim(bin_dir: Path, *, total_kb: int, avail_kb: int) -> int:
    """Create a `df` shim reporting an exact total/avail pair. Returns used pct.

    The shim REFUSES to answer unless `-k` is passed. That is deliberate: POSIX
    `df -P` alone means 512-byte blocks on BSD/macOS and 1024-byte blocks on
    GNU/Linux, and the guard now halts on that number. Measured on the Mac
    2026-09-04, the same 276 GiB reads 579145608 under `df -P` and 289572804
    under `df -Pk` — a 2x error that would admit at half the declared floor.
    Pinning it here means a future edit that drops the `-k` fails loudly rather
    than silently doubling every reading on one of the two host families.
    """
    used_kb = total_kb - avail_kb
    used_pct = round(used_kb * 100 / total_kb)
    shim = bin_dir / "df"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ " $* " != *" -Pk "* && " $* " != *"-k"* ]]; then\n'
        '  echo "df shim: refusing a call without -k (block size must be 1 KiB): $*" >&2\n'
        "  exit 64\n"
        "fi\n"
        "cat <<'POSIXEOF'\n"
        "Filesystem     1024-blocks      Used Available Capacity Mounted on\n"
        f"/dev/disk3s5   {total_kb}  {used_kb}  {avail_kb}  {used_pct}% /System/Volumes/Data\n"
        "POSIXEOF\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return used_pct


def _run(
    tmp_path: Path, *, avail_gb: int, total_kb: int = _MAC_DATA_TOTAL_KB
) -> tuple[subprocess.CompletedProcess[str], int]:
    """Run the guard --dry-run against a synthetic volume. Returns (proc, used_pct)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    used_pct = _make_df_shim(
        bin_dir, total_kb=total_kb, avail_kb=avail_gb * _KIB_PER_GIB
    )
    hostname_shim = bin_dir / "hostname"
    hostname_shim.write_text("#!/usr/bin/env bash\necho mac-test\n")
    hostname_shim.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env.pop("KAFKA_BOOTSTRAP_SERVERS", None)
    env.pop("ONEX_BUS_PUBLISH_URL", None)

    proc = subprocess.run(
        [
            "bash",
            str(_SCRIPTS / "disk-watermark-check.sh"),
            "--mount",
            "/System/Volumes/Data",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )
    return proc, used_pct


@pytest.mark.unit
class TestDiskWatermarkFreeSpaceFloor:
    """The halt criterion is absolute free space, never the percentage."""

    def test_thresholds_are_declared_in_the_guard_config_not_env(self) -> None:
        """Both floors live in the guard's own config file (Rule 8: no env var)."""
        assert _THRESHOLDS.is_file(), (
            f"{_THRESHOLDS} must exist — the guard declares its own thresholds"
        )
        declared = json.loads(_THRESHOLDS.read_text())
        assert declared["crit_free_gb"] == 50, declared
        assert declared["warn_free_gb"] == 100, declared
        assert declared["warn_pct"] == 85, declared

        script = (_SCRIPTS / "disk-watermark-check.sh").read_text()
        for banned in (
            "DISK_WATERMARK_CRIT_FREE_GB",
            "DISK_WATERMARK_WARN_FREE_GB",
            "DISK_WATERMARK_WARN_PCT",
        ):
            assert banned not in script, (
                f"{banned}: thresholds must come from {_THRESHOLDS.name}, not env"
            )

    def test_276gb_free_on_a_36tb_volume_is_admitted(self, tmp_path: Path) -> None:
        """The live 2026-09-04 reading: 93% used but 276 GiB free — never a halt."""
        proc, used_pct = _run(tmp_path, avail_gb=_ADMITTED_AVAIL_GB)
        assert used_pct >= 90, (
            f"fixture must sit above the OLD crit_pct=90 line; got {used_pct}%"
        )
        assert proc.returncode != 20, (
            f"276 GiB free must never produce the critical halt code; "
            f"got {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr}"
        )
        assert '"severity": "critical"' not in proc.stdout, proc.stdout
        # It is still above warn_pct, so it warns — loudly, but it does not halt.
        assert proc.returncode == 10, (
            f"expected the secondary percentage warning (exit 10); "
            f"got {proc.returncode}; stderr={proc.stderr}"
        )
        assert '"severity": "warning"' in proc.stdout, proc.stdout

    def test_20gb_free_on_the_same_volume_is_refused(self, tmp_path: Path) -> None:
        """Below the 50 GiB floor the guard halts, whatever the percentage says."""
        proc, _ = _run(tmp_path, avail_gb=_REFUSED_AVAIL_GB)
        assert proc.returncode == 20, (
            f"20 GiB free is below the 50 GiB floor and must halt; "
            f"got {proc.returncode}; stderr={proc.stderr}"
        )
        assert '"severity": "critical"' in proc.stdout, proc.stdout

    def test_percentage_alone_can_never_escalate_to_critical(
        self, tmp_path: Path
    ) -> None:
        """99% used with 400 GiB free on a huge volume: warn, never halt."""
        huge_total_kb = 40 * 1024 * _KIB_PER_GIB  # 40 TiB
        proc, used_pct = _run(tmp_path, avail_gb=400, total_kb=huge_total_kb)
        assert used_pct >= 99, used_pct
        assert proc.returncode == 10, (
            f"percentage must be advisory only; got {proc.returncode}; "
            f"stderr={proc.stderr}"
        )
        assert '"severity": "critical"' not in proc.stdout, proc.stdout

    def test_free_space_below_warn_floor_warns_even_when_pct_is_quiet(
        self, tmp_path: Path
    ) -> None:
        """A small volume 60% full but with only 60 GiB free still warns."""
        small_total_kb = 150 * _KIB_PER_GIB  # 150 GiB
        proc, used_pct = _run(tmp_path, avail_gb=60, total_kb=small_total_kb)
        assert used_pct < 85, f"fixture must be quiet on percentage; got {used_pct}%"
        assert proc.returncode == 10, (
            f"60 GiB free is under the 100 GiB warn floor; got {proc.returncode}; "
            f"stderr={proc.stderr}"
        )
        assert '"severity": "warning"' in proc.stdout, proc.stdout

    def test_healthy_volume_is_quiet(self, tmp_path: Path) -> None:
        """Above both floors and under warn_pct: exit 0, no event."""
        proc, used_pct = _run(tmp_path, avail_gb=2000)
        assert used_pct < 85, used_pct
        assert proc.returncode == 0, f"expected quiet; stderr={proc.stderr}"
        assert proc.stdout.strip() == "", proc.stdout

    def test_free_space_is_read_in_kib_not_512_byte_blocks(
        self, tmp_path: Path
    ) -> None:
        """The guard must ask df for 1 KiB blocks explicitly (`-k`).

        Without `-k`, macOS returns 512-byte blocks and every free-space figure
        doubles, so the 50 GiB floor would in practice admit at 25 GiB. The df
        shim above refuses a call that omits `-k`; if the script ever drops it,
        the shim exits 64, the script cannot parse a reading, and this fails.
        """
        source = (_SCRIPTS / "disk-watermark-check.sh").read_text()
        assert "df -Pk" in source, "the guard must call df with -k"
        assert "df -P " not in source, (
            "bare `df -P` is 512-byte blocks on macOS — every reading doubles"
        )
        proc, _ = _run(tmp_path, avail_gb=_REFUSED_AVAIL_GB)
        assert proc.returncode == 20, (
            f"the shim rejects a df call without -k; got {proc.returncode}; "
            f"stderr={proc.stderr}"
        )

    def test_receipt_names_both_numbers(self, tmp_path: Path) -> None:
        """The event must carry free-GB vs floor AND used-pct vs pct line."""
        proc, _ = _run(tmp_path, avail_gb=_REFUSED_AVAIL_GB)
        event = json.loads(proc.stdout.strip().splitlines()[-1])
        assert event["avail_gb"] == _REFUSED_AVAIL_GB, event
        assert event["crit_free_gb"] == 50, event
        assert event["warn_free_gb"] == 100, event
        assert event["warn_pct"] == 85, event
        assert event["halt_reason"] == "free_space_below_crit_floor", event
        assert "20 GiB free" in event["message"], event["message"]
        assert "crit floor 50 GiB" in event["message"], event["message"]
        assert "% used" in event["message"], event["message"]
