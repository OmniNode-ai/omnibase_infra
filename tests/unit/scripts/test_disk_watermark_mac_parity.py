# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Mac disk-watermark parity tests (OMN-13229).

Verifies that disk-watermark-check.sh behaves correctly when targeting the Mac
Data volume (/System/Volumes/Data). Tests mock `df` to inject synthetic usage
readings and mock `rpk`/`curl` to intercept publish calls — no live side effects.

Threshold contract:
  >85% used  => warning event emitted (ticket-path signal to sweep consumer)
  >90% used  => critical event emitted (loud bus event)
  <85% used  => quiet (exit 0, no event, no publish call)

The Mac publish path uses `curl` (ONEX_BUS_PUBLISH_URL) because rpk is not
present on Mac hosts. Both paths are tested here.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"

_FAKE_DF_TEMPLATE = """\
#!/usr/bin/env bash
# Shim df — returns synthetic POSIX output for any path argument.
# Usage percent is hard-wired to FAKE_USED_PCT env var.
cat <<'EOF'
Filesystem     1024-blocks      Used Available Use% Mounted on
/dev/disk3s1   976762584  {used_blocks}  {avail_blocks}  {pct}% /System/Volumes/Data
EOF
"""


def _make_df_shim(bin_dir: Path, used_pct: int) -> None:
    """Create a `df` shim that reports a fixed usage percentage."""
    total_blocks = 976762584
    used_blocks = int(total_blocks * used_pct / 100)
    avail_blocks = total_blocks - used_blocks
    shim = bin_dir / "df"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "cat <<'POSIXEOF'\n"
        "Filesystem     1024-blocks      Used Available Use% Mounted on\n"
        f"/dev/disk3s1   {total_blocks}  {used_blocks}  {avail_blocks}  {used_pct}% /System/Volumes/Data\n"
        "POSIXEOF\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _make_call_recorder(
    bin_dir: Path, name: str, calllog: Path, exit_code: int = 0
) -> None:
    """Create a shim that records its invocation and exits with the given code."""
    shim = bin_dir / name
    shim.write_text(
        f'#!/usr/bin/env bash\necho "{name} $*" >> "{calllog}"\nexit {exit_code}\n'
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _make_curl_shim(bin_dir: Path, calllog: Path, http_status: str = "200") -> None:
    """Create a curl shim that records the POST call and returns the given HTTP status."""
    shim = bin_dir / "curl"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "curl $*" >> "{calllog}"\n'
        # `-w %{http_code}` causes the real curl to print the status code on stdout.
        # Our shim must do the same so the script can read it.
        f'echo -n "{http_status}"\n'
        "exit 0\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_watermark(
    tmp_path: Path,
    *,
    used_pct: int,
    mount: str = "/System/Volumes/Data",
    warn_pct: int = 85,
    crit_pct: int = 90,
    extra_env: dict[str, str] | None = None,
    dry_run: bool = False,
    curl_http_status: str = "200",
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run disk-watermark-check.sh with a synthetic df and return (proc, calllog)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calllog = tmp_path / "calls.log"
    calllog.write_text("")

    _make_df_shim(bin_dir, used_pct)
    _make_call_recorder(
        bin_dir, "rpk", calllog
    )  # rpk present but unused (no KAFKA_BOOTSTRAP_SERVERS)
    _make_curl_shim(bin_dir, calllog, http_status=curl_http_status)
    # hostname shim so HOSTNAME_TAG is deterministic
    hostname_shim = bin_dir / "hostname"
    hostname_shim.write_text("#!/usr/bin/env bash\necho mac-test\n")
    hostname_shim.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env.pop("KAFKA_BOOTSTRAP_SERVERS", None)
    env.pop("ONEX_BUS_PUBLISH_URL", None)
    if extra_env:
        env.update(extra_env)

    args: list[str] = [
        "bash",
        str(_SCRIPTS / "disk-watermark-check.sh"),
        "--mount",
        mount,
        "--warn",
        str(warn_pct),
        "--crit",
        str(crit_pct),
    ]
    if dry_run:
        args.append("--dry-run")

    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )
    return proc, calllog.read_text()


@pytest.mark.unit
class TestDiskWatermarkMacParity:
    """Mac /System/Volumes/Data watermark threshold behaviour."""

    def test_under_threshold_is_quiet_no_publish(self, tmp_path: Path) -> None:
        """84% usage must exit 0 with no publish call and no event on stdout."""
        proc, calls = _run_watermark(tmp_path, used_pct=84)
        assert proc.returncode == 0, f"expected quiet exit; stderr={proc.stderr}"
        # Must not call rpk or curl
        assert "rpk" not in calls, f"rpk called unexpectedly:\n{calls}"
        assert "curl" not in calls, f"curl called unexpectedly:\n{calls}"
        # stdout silent
        assert proc.stdout.strip() == "", f"unexpected stdout: {proc.stdout!r}"

    def test_warn_threshold_triggers_ticket_path_event(self, tmp_path: Path) -> None:
        """87% usage crosses warn=85 — must emit a warning event (ticket-path signal)."""
        proc, _ = _run_watermark(
            tmp_path,
            used_pct=87,
            dry_run=True,
        )
        # Exit code 10 = warn breach
        assert proc.returncode == 10, (
            f"expected exit 10; got {proc.returncode}; stderr={proc.stderr}"
        )
        # Dry-run prints the event JSON to stdout
        assert '"severity": "warning"' in proc.stdout, proc.stdout
        assert '"event_type": "disk-watermark"' in proc.stdout, proc.stdout

    def test_crit_threshold_triggers_loud_bus_event(self, tmp_path: Path) -> None:
        """91% usage crosses crit=90 — must emit a critical (loud) bus event."""
        proc, _ = _run_watermark(
            tmp_path,
            used_pct=91,
            dry_run=True,
        )
        # Exit code 20 = crit breach
        assert proc.returncode == 20, (
            f"expected exit 20; got {proc.returncode}; stderr={proc.stderr}"
        )
        assert '"severity": "critical"' in proc.stdout, proc.stdout

    def test_warn_breach_publishes_via_http_when_url_set(self, tmp_path: Path) -> None:
        """87% + ONEX_BUS_PUBLISH_URL set → curl POST invoked (non-dry-run)."""
        proc, calls = _run_watermark(
            tmp_path,
            used_pct=87,
            extra_env={
                "ONEX_BUS_PUBLISH_URL": "http://bus.example.internal/api/events"
            },
        )
        assert proc.returncode == 10, f"expected exit 10; stderr={proc.stderr}"
        assert "curl" in calls, f"curl not invoked for HTTP publish:\n{calls}"
        # Must POST to the configured URL, not a hardcoded address
        assert "bus.example.internal" in calls, f"wrong URL in curl call:\n{calls}"

    def test_crit_breach_publishes_via_http_when_url_set(self, tmp_path: Path) -> None:
        """91% + ONEX_BUS_PUBLISH_URL set → curl POST with critical event."""
        proc, calls = _run_watermark(
            tmp_path,
            used_pct=91,
            extra_env={
                "ONEX_BUS_PUBLISH_URL": "http://bus.example.internal/api/events"
            },
        )
        assert proc.returncode == 20, f"expected exit 20; stderr={proc.stderr}"
        assert "curl" in calls, f"curl not invoked:\n{calls}"

    def test_dry_run_does_not_call_curl_or_rpk(self, tmp_path: Path) -> None:
        """--dry-run must never invoke curl or rpk regardless of env vars."""
        proc, calls = _run_watermark(
            tmp_path,
            used_pct=95,
            dry_run=True,
            extra_env={
                "ONEX_BUS_PUBLISH_URL": "http://bus.example.internal/api/events"
            },
        )
        assert proc.returncode == 20, proc.stderr
        assert "curl" not in calls, f"dry-run invoked curl:\n{calls}"
        assert "rpk produce" not in calls, f"dry-run invoked rpk produce:\n{calls}"

    def test_no_broker_and_no_url_logs_but_does_not_crash(self, tmp_path: Path) -> None:
        """Breach with neither KAFKA_BOOTSTRAP_SERVERS nor ONEX_BUS_PUBLISH_URL:
        script must log the event and exit with breach code, never crash."""
        proc, calls = _run_watermark(tmp_path, used_pct=91)
        assert proc.returncode == 20, f"expected exit 20; stderr={proc.stderr}"
        # Neither publisher should have been invoked
        assert "produce" not in calls, calls
        assert "curl" not in calls, calls

    def test_http_publish_failure_is_logged_not_fatal(self, tmp_path: Path) -> None:
        """HTTP 500 from the bus endpoint must not change the exit code or crash."""
        proc, calls = _run_watermark(
            tmp_path,
            used_pct=91,
            extra_env={
                "ONEX_BUS_PUBLISH_URL": "http://bus.example.internal/api/events"
            },
            curl_http_status="500",
        )
        # Breach exit code must still be returned (not 1 from a failed publish)
        assert proc.returncode == 20, f"expected exit 20; stderr={proc.stderr}"
        assert "curl" in calls, calls

    def test_event_mount_field_reflects_data_volume(self, tmp_path: Path) -> None:
        """The emitted event must reference /System/Volumes/Data, not /."""
        proc, _ = _run_watermark(
            tmp_path,
            used_pct=87,
            mount="/System/Volumes/Data",
            dry_run=True,
        )
        assert proc.returncode == 10, proc.stderr
        assert (
            "/System/Volumes/Data" in proc.stdout or "/data" in proc.stdout.lower()
        ), f"mount not in event: {proc.stdout!r}"
