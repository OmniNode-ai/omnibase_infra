# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner containers tell tools their real CPU count (OMN-19960).

A runner container runs under a CFS quota of 2 CPUs (``cpus: "2.0"``), but
``nproc`` and ``os.sched_getaffinity`` inside it report every host core (32 on
.201 and .202). pre-commit sizes its fan-out from that count, and on .202 the
``validate-spdx-headers`` hook's 32 partitions were memcg OOM-killed at the
runner's 6 GiB limit. ``pytest -n auto`` makes the same mistake, which is why
``ci.yml`` pins ``PYTEST_XDIST_AUTO_NUM_WORKERS`` per job (OMN-19209).

The fix is one image change: ``docker/runners/cgroup-cpu-env.sh`` defines
``omni_cgroup_cpu_env``, which reads the container's own ``cpu.max``, computes
``N = ceil(quota / period)``, and exports the count for the tools that fan out.
``entrypoint.sh`` sources it before it spawns ``run.sh``, so every job step
inherits the values.

These tests pin the function's behaviour over five ``cpu.max`` fixtures and the
wiring facts that make it reach a job: the entrypoint sources and calls it
before the ``run.sh`` spawn, the Dockerfile copies it into the image, the host
sync carries it into the image build context, and the image version moved past
10 so the fleet-drift check sees a new image.

``entrypoint.sh`` is bind-mounted into every runner from the host's staged
copy, which converges ahead of any image rebuild, so a runner still on an older
image can restart with the new entrypoint and no ``cgroup-cpu-env.sh``. The
entrypoint must skip a missing file rather than die under ``set -e``; one test
runs the real entrypoint block against an absent file to prove it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS_DIR = REPO_ROOT / "docker" / "runners"
CPU_ENV_SCRIPT = RUNNERS_DIR / "cgroup-cpu-env.sh"
ENTRYPOINT = RUNNERS_DIR / "entrypoint.sh"
DOCKERFILE = RUNNERS_DIR / "Dockerfile"
LOCK_FILE = RUNNERS_DIR / "runner-image.lock.json"
DEPLOY_RUNNERS = REPO_ROOT / "scripts" / "deploy-runners.sh"

EXPORTED = (
    "OMNI_CGROUP_CPUS",
    "PYTEST_XDIST_AUTO_NUM_WORKERS",
    "OMP_NUM_THREADS",
    "PRE_COMMIT_NO_CONCURRENCY",
)

BASH = shutil.which("bash")


def _run_function(
    tmp_path: Path, cpu_max: str | None, preset: dict[str, str] | None = None
) -> dict[str, str]:
    """Source the script, call the function on a fixture cpu.max, return the
    subset of the resulting environment this ticket cares about."""
    assert BASH is not None, "bash is required to exercise the runner script"
    assert CPU_ENV_SCRIPT.exists(), f"{CPU_ENV_SCRIPT} does not exist"
    if cpu_max is None:
        fixture = tmp_path / "absent-cpu.max"
    else:
        fixture = tmp_path / "cpu.max"
        fixture.write_text(cpu_max + "\n", encoding="utf-8")
    # A clean environment, so a value the test host happens to carry (for
    # example a CI job's own PYTEST_XDIST_AUTO_NUM_WORKERS) cannot mask the
    # function's behaviour.
    env = {"PATH": "/usr/bin:/bin"}
    env.update(preset or {})
    result = subprocess.run(
        [
            BASH,
            "-c",
            'set -euo pipefail; source "$1"; omni_cgroup_cpu_env "$2"; env',
            "bash",
            str(CPU_ENV_SCRIPT),
            str(fixture),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"function failed under set -euo pipefail: rc={result.returncode} "
        f"stderr={result.stderr!r}"
    )
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in EXPORTED:
            out[key] = value
    return out


def test_two_cpu_quota_exports_two_and_serial_precommit(tmp_path: Path) -> None:
    assert _run_function(tmp_path, "200000 100000") == {
        "OMNI_CGROUP_CPUS": "2",
        "PYTEST_XDIST_AUTO_NUM_WORKERS": "2",
        "OMP_NUM_THREADS": "2",
        "PRE_COMMIT_NO_CONCURRENCY": "1",
    }


def test_fractional_quota_rounds_up(tmp_path: Path) -> None:
    assert _run_function(tmp_path, "150000 100000") == {
        "OMNI_CGROUP_CPUS": "2",
        "PYTEST_XDIST_AUTO_NUM_WORKERS": "2",
        "OMP_NUM_THREADS": "2",
        "PRE_COMMIT_NO_CONCURRENCY": "1",
    }


def test_eight_cpu_quota_leaves_precommit_concurrent(tmp_path: Path) -> None:
    assert _run_function(tmp_path, "800000 100000") == {
        "OMNI_CGROUP_CPUS": "8",
        "PYTEST_XDIST_AUTO_NUM_WORKERS": "8",
        "OMP_NUM_THREADS": "8",
    }


def test_unlimited_quota_exports_nothing(tmp_path: Path) -> None:
    assert _run_function(tmp_path, "max 100000") == {}


def test_missing_cpu_max_exports_nothing(tmp_path: Path) -> None:
    # cgroup v1 hosts, or a sandbox without the cgroup mount: no guess.
    assert _run_function(tmp_path, None) == {}


def test_preset_value_is_left_alone(tmp_path: Path) -> None:
    got = _run_function(
        tmp_path, "200000 100000", preset={"PYTEST_XDIST_AUTO_NUM_WORKERS": "4"}
    )
    assert got["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "4"
    assert got["OMNI_CGROUP_CPUS"] == "2"
    assert got["OMP_NUM_THREADS"] == "2"


def test_entrypoint_calls_the_function_before_spawning_run_sh() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    source_at = text.find("source /usr/local/bin/cgroup-cpu-env.sh")
    call = re.search(r"^\s*omni_cgroup_cpu_env\s*$", text, flags=re.MULTILINE)
    spawn_at = text.find('_as_runner "${RUNNER_HOME}/run.sh"')
    assert source_at != -1, "entrypoint.sh must source cgroup-cpu-env.sh"
    assert call is not None, "entrypoint.sh must call omni_cgroup_cpu_env"
    assert spawn_at != -1, "entrypoint.sh no longer spawns run.sh as expected"
    assert source_at < call.start() < spawn_at, (
        "the CPU count must be exported before run.sh is spawned, "
        "or job steps do not inherit it"
    )


def _entrypoint_cpu_block() -> str:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    start = text.index("if [[ -r /usr/local/bin/cgroup-cpu-env.sh ]]; then")
    end = text.index("\nfi\n", start) + len("\nfi\n")
    return text[start:end]


@pytest.mark.parametrize(
    ("script_present", "expected_cpus"), [(False, ""), (True, "2")]
)
def test_entrypoint_block_survives_an_image_without_the_script(
    tmp_path: Path, script_present: bool, expected_cpus: str
) -> None:
    """Run the entrypoint's own block, with the installed path rewritten to a
    scratch location, under the entrypoint's ``set -euo pipefail``."""
    assert BASH is not None
    installed = tmp_path / "usr-local-bin" / "cgroup-cpu-env.sh"
    installed.parent.mkdir()
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("200000 100000\n", encoding="utf-8")
    if script_present:
        # Point the function's default path at the fixture, as the kernel's
        # /sys/fs/cgroup/cpu.max would be inside a runner.
        installed.write_text(
            CPU_ENV_SCRIPT.read_text(encoding="utf-8").replace(
                "/sys/fs/cgroup/cpu.max", str(cpu_max)
            ),
            encoding="utf-8",
        )
    block = _entrypoint_cpu_block().replace(
        "/usr/local/bin/cgroup-cpu-env.sh", str(installed)
    )
    result = subprocess.run(
        [
            BASH,
            "-c",
            "set -euo pipefail\n" + block + 'echo "CPUS=${OMNI_CGROUP_CPUS:-}"',
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"CPUS={expected_cpus}" in result.stdout.splitlines()


def test_host_sync_carries_the_script_into_the_build_context() -> None:
    text = DEPLOY_RUNNERS.read_text(encoding="utf-8")
    assert '    "docker/runners/cgroup-cpu-env.sh"\n' in text, (
        "deploy-runners.sh SYNC_PATHS must list cgroup-cpu-env.sh"
    )
    assert '"${REPO_ROOT}/docker/runners/cgroup-cpu-env.sh" \\\n' in text, (
        "deploy-runners.sh must rsync cgroup-cpu-env.sh into the host's "
        "docker/runners build context, or the image build fails at its COPY"
    )


def test_dockerfile_copies_the_script_into_the_image() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(
        r"^COPY cgroup-cpu-env\.sh /usr/local/bin/cgroup-cpu-env\.sh$",
        text,
        flags=re.MULTILINE,
    ), "Dockerfile must COPY cgroup-cpu-env.sh to /usr/local/bin"


def test_image_version_moved_past_ten() -> None:
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    assert data["image_version"] >= 11, (
        "the runner image content changed; bump image_version so the fleet "
        "drift check sees a new image"
    )


@pytest.mark.parametrize("cpu_max", ["200000 100000"])
def test_sourcing_twice_changes_nothing(tmp_path: Path, cpu_max: str) -> None:
    assert BASH is not None
    fixture = tmp_path / "cpu.max"
    fixture.write_text(cpu_max + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            BASH,
            "-c",
            'set -euo pipefail; source "$1"; omni_cgroup_cpu_env "$2"; '
            'source "$1"; omni_cgroup_cpu_env "$2"; echo "$OMNI_CGROUP_CPUS"',
            "bash",
            str(CPU_ENV_SCRIPT),
            str(fixture),
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2"
