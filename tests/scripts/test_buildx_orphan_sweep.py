# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression/simulation coverage for `scripts/buildx-orphan-sweep.sh`
(OMN-16406, "Layer 0b" of the disk-guard architecture).

Drives the real script against a stub `docker` binary -- no real Docker
daemon, no real buildx builders required. Each test asserts on the
cumulative `rm`/`volume rm` call log plus stdout log lines, proving the
three safe-removal criteria are each independently enforced:

  1. Registered in `docker buildx ls` -> never touched, regardless of age
     or activity.
  2. Age below the floor -> never touched, even if unregistered.
  3. Running AND not idle across two samples (CPU or block I/O changed) ->
     never touched -- the mid-build protection.

And that a container passing all three criteria is removed together with
its own `buildx_buildkit_`-prefixed state volume, in `--execute` mode only
(dry-run never mutates).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "buildx-orphan-sweep.sh"

pytestmark = [pytest.mark.unit]

# A fixed "now" so age-in-minutes math in the script is deterministic.
NOW_EPOCH = 2_000_000_000  # 2033-05-18T03:33:20Z -- arbitrary, just fixed.
OLD_ISO = "2020-01-01T00:00:00.000000000Z"  # far more than 60min before NOW_EPOCH


def _young_iso(minutes_before_now: int) -> str:
    """An RFC3339 timestamp `minutes_before_now` minutes before NOW_EPOCH."""
    import datetime

    dt = datetime.datetime.fromtimestamp(
        NOW_EPOCH - minutes_before_now * 60, tz=datetime.UTC
    )
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def _write_docker_stub(
    bin_dir: Path,
    *,
    registered_nodes: list[str],
    containers: dict[str, dict[str, Any]],
    calls_log: Path,
) -> None:
    """Write a bash `docker` stub covering exactly the subcommands the
    sweep script issues: `buildx ls`, `ps -a --filter ... --format ...`,
    `inspect --format <fmt> <name>` (three distinct format strings),
    `stats --no-stream --format ... <name>` (two calls per running
    candidate, tracked via a per-name counter file), `rm -f`, `volume rm`.

    containers: {name: {"created": iso, "status": "...", "mounts": [...],
                          "stats": [(cpu, io), (cpu, io)]}}
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    counter_dir = bin_dir / "_stats_counters"
    counter_dir.mkdir(parents=True, exist_ok=True)

    ls_lines = ["NAME/NODE       DRIVER/ENDPOINT       STATUS  BUILDKIT PLATFORMS"]
    for node in registered_nodes:
        ls_lines.append(f"{node}-builder      docker-container")
        ls_lines.append(
            f"  {node} unix:///var/run/docker.sock running v0.12.0 linux/amd64"
        )
    ls_output = "\n".join(ls_lines)

    ps_output = "\n".join(containers.keys())

    case_blocks = []
    for name, spec in containers.items():
        mounts_out = "\\n".join(spec.get("mounts", []))
        case_blocks.append(
            f"""
    {name})
      case "$fmt" in
        '{{{{.Created}}}}') echo "{spec["created"]}" ;;
        '{{{{.State.Status}}}}') echo "{spec["status"]}" ;;
        '{{{{range .Mounts}}}}{{{{if eq .Type "volume"}}}}{{{{.Name}}}}{{{{"\\n"}}}}{{{{end}}}}{{{{end}}}}') printf '{mounts_out}\\n' ;;
      esac
      ;;"""
        )
    inspect_cases = "".join(case_blocks)

    stats_blocks = []
    for name, spec in containers.items():
        samples = spec.get("stats", [])
        if not samples:
            continue
        counter_file = counter_dir / name
        sample_lines = "\n".join(
            f'      {i}) echo "{cpu}|{io}" ;;' for i, (cpu, io) in enumerate(samples)
        )
        stats_blocks.append(
            f"""
    {name})
      cf="{counter_file}"
      idx=0
      [[ -f "$cf" ]] && idx=$(cat "$cf")
      case "$idx" in
{sample_lines}
      esac
      echo $((idx + 1)) > "$cf"
      ;;"""
        )
    stats_cases = "".join(stats_blocks)

    script = f"""#!/usr/bin/env bash
echo "CALL: $*" >> "{calls_log}"
case "$1" in
  buildx)
    if [[ "$2" == "ls" ]]; then
      cat <<'LSEOF'
{ls_output}
LSEOF
    fi
    ;;
  ps)
    cat <<'PSEOF'
{ps_output}
PSEOF
    ;;
  inspect)
    fmt="$3"
    name="$4"
    case "$name" in{inspect_cases}
    esac
    ;;
  stats)
    name="${{@: -1}}"
    case "$name" in{stats_cases}
    esac
    ;;
  rm)
    echo "rm ${{@: -1}}" >> "{calls_log}"
    ;;
  volume)
    if [[ "$2" == "rm" ]]; then
      echo "volrm $3" >> "{calls_log}"
    fi
    ;;
esac
"""
    stub = bin_dir / "docker"
    stub.write_text(script)
    stub.chmod(0o755)


def _run(
    bin_dir: Path,
    *,
    execute: bool = False,
    min_age_minutes: int = 60,
    sample_gap_seconds: int = 1,
) -> subprocess.CompletedProcess[str]:
    import os

    env = dict(os.environ)
    env["BUILDX_ORPHAN_SWEEP_DOCKER_BIN"] = str(bin_dir / "docker")
    env["BUILDX_ORPHAN_SWEEP_NOW_EPOCH_OVERRIDE"] = str(NOW_EPOCH)
    args = [
        "bash",
        str(SWEEP_SCRIPT),
        "--json",
        "--min-age-minutes",
        str(min_age_minutes),
        "--sample-gap-seconds",
        str(sample_gap_seconds),
    ]
    if execute:
        args.append("--execute")
    return subprocess.run(
        args, env=env, capture_output=True, text=True, timeout=30, check=False
    )


def test_script_exists_and_is_executable() -> None:
    assert SWEEP_SCRIPT.is_file(), f"sweep script missing: {SWEEP_SCRIPT}"
    mode = SWEEP_SCRIPT.stat().st_mode
    assert mode & 0o111, "sweep script must be executable"


def test_no_buildx_containers_is_a_clean_noop(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    calls_log = tmp_path / "calls.log"
    _write_docker_stub(bin_dir, registered_nodes=[], containers={}, calls_log=calls_log)

    result = _run(bin_dir, execute=True)

    assert result.returncode == 0, result.stderr
    assert '"candidates": []' in result.stdout
    assert not calls_log.exists() or "rm " not in calls_log.read_text()


def test_registered_builder_is_never_touched_regardless_of_age(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    calls_log = tmp_path / "calls.log"
    name = "buildx_buildkit_onex-shared-buildkit0"
    _write_docker_stub(
        bin_dir,
        registered_nodes=["onex-shared-buildkit0"],
        containers={
            name: {"created": OLD_ISO, "status": "running", "mounts": [f"{name}_state"]}
        },
        calls_log=calls_log,
    )

    result = _run(bin_dir, execute=True)

    assert result.returncode == 0, result.stderr
    assert 'candidates": []' in result.stdout
    assert "registered" in result.stderr
    calls = calls_log.read_text() if calls_log.exists() else ""
    assert f"rm {name}" not in calls
    assert "stats" not in calls  # never even sampled -- filtered before idle check


def test_unregistered_but_too_young_is_skipped(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    calls_log = tmp_path / "calls.log"
    name = "buildx_buildkit_builder-freshuuid0"
    _write_docker_stub(
        bin_dir,
        registered_nodes=[],
        containers={name: {"created": _young_iso(5), "status": "exited", "mounts": []}},
        calls_log=calls_log,
    )

    result = _run(bin_dir, execute=True, min_age_minutes=60)

    assert result.returncode == 0, result.stderr
    assert 'candidates": []' in result.stdout
    assert "60min floor" in result.stderr
    calls = calls_log.read_text() if calls_log.exists() else ""
    assert f"rm {name}" not in calls


def test_exited_old_orphan_removed_with_its_volume_in_execute_mode(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    calls_log = tmp_path / "calls.log"
    name = "buildx_buildkit_builder-aaaa0"
    volume = f"{name}_state"
    _write_docker_stub(
        bin_dir,
        registered_nodes=[],
        containers={name: {"created": OLD_ISO, "status": "exited", "mounts": [volume]}},
        calls_log=calls_log,
    )

    dry = _run(bin_dir, execute=False)
    assert dry.returncode == 0, dry.stderr
    assert name in dry.stdout
    dry_calls = calls_log.read_text() if calls_log.exists() else ""
    assert f"rm {name}" not in dry_calls, "dry-run must never mutate"

    execute = _run(bin_dir, execute=True)
    assert execute.returncode == 0, execute.stderr
    calls = calls_log.read_text()
    assert f"rm {name}" in calls
    assert f"volrm {volume}" in calls
    assert name in execute.stdout and '"removed"' in execute.stdout


def test_running_orphan_with_changing_block_io_is_never_removed(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    calls_log = tmp_path / "calls.log"
    name = "buildx_buildkit_builder-midbuild0"
    _write_docker_stub(
        bin_dir,
        registered_nodes=[],
        containers={
            name: {
                "created": OLD_ISO,
                "status": "running",
                "mounts": [f"{name}_state"],
                # Block I/O advances between the two samples -> mid-build, must survive.
                "stats": [("42.00%", "676MB / 1.54GB"), ("38.00%", "812MB / 1.60GB")],
            }
        },
        calls_log=calls_log,
    )

    result = _run(bin_dir, execute=True, sample_gap_seconds=1)

    assert result.returncode == 0, result.stderr
    assert 'candidates": []' in result.stdout
    assert "NOT idle" in result.stderr
    calls = calls_log.read_text() if calls_log.exists() else ""
    assert f"rm {name}" not in calls


def test_running_orphan_idle_across_two_samples_is_removed(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    calls_log = tmp_path / "calls.log"
    name = "buildx_buildkit_builder-idle0"
    volume = f"{name}_state"
    _write_docker_stub(
        bin_dir,
        registered_nodes=[],
        containers={
            name: {
                "created": OLD_ISO,
                "status": "running",
                "mounts": [volume],
                # Identical block I/O both samples, low CPU both times -> idle.
                "stats": [("0.01%", "676MB / 1.54GB"), ("0.00%", "676MB / 1.54GB")],
            }
        },
        calls_log=calls_log,
    )

    result = _run(bin_dir, execute=True, sample_gap_seconds=1)

    assert result.returncode == 0, result.stderr
    calls = calls_log.read_text()
    assert f"rm {name}" in calls
    assert f"volrm {volume}" in calls


def test_mixed_fleet_only_the_qualifying_orphan_is_removed(tmp_path: Path) -> None:
    """One registered, one too-young, one mid-build, one genuine idle orphan --
    only the genuine orphan is touched. Regression guard against a filter
    stage accidentally leaking a non-qualifying container into removal."""
    bin_dir = tmp_path / "bin"
    calls_log = tmp_path / "calls.log"
    registered = "buildx_buildkit_onex-shared-buildkit0"
    young = "buildx_buildkit_builder-young0"
    midbuild = "buildx_buildkit_builder-midbuild0"
    orphan = "buildx_buildkit_builder-orphan0"
    orphan_volume = f"{orphan}_state"

    _write_docker_stub(
        bin_dir,
        registered_nodes=["onex-shared-buildkit0"],
        containers={
            registered: {"created": OLD_ISO, "status": "running", "mounts": []},
            young: {"created": _young_iso(2), "status": "exited", "mounts": []},
            midbuild: {
                "created": OLD_ISO,
                "status": "running",
                "mounts": [],
                "stats": [("55.00%", "1MB / 1MB"), ("60.00%", "2MB / 2MB")],
            },
            orphan: {
                "created": OLD_ISO,
                "status": "exited",
                "mounts": [orphan_volume],
            },
        },
        calls_log=calls_log,
    )

    result = _run(bin_dir, execute=True, sample_gap_seconds=1)

    assert result.returncode == 0, result.stderr
    calls = calls_log.read_text()
    assert f"rm {orphan}" in calls
    assert f"volrm {orphan_volume}" in calls
    for untouched in (registered, young, midbuild):
        assert f"rm {untouched}" not in calls
