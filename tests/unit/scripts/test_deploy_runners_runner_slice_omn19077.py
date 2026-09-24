# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19077: the .201 general runner pool is capped at 40 and runs in one
aggregate cgroup (``omnirunners.slice``), and ``deploy-runners.sh`` can scale
the pool DOWN one idle runner at a time.

Why these properties and not others:

- The slice is only a protection if it EXISTS with its limits before a runner
  is created in it. With the systemd cgroup driver, systemd silently creates a
  missing slice with no limits, so a runner placed in an uninstalled slice looks
  exactly like a correct one. The install must therefore run before every path
  that creates a runner, and must read the live values back and fail closed.
- The scale-down must never take a job down with it. The pre-existing way to
  drop runners was a default deploy's ``--remove-orphans``, which removes every
  surplus container in one call with no busy check.
- A removed container's named creds volume is data and is kept.

Where a property can be executed, it is executed: the functions are extracted
from the real script and run against stubbed ``ssh``/``rsync``/``gh``.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"
SLICE_FILE = REPO_ROOT / "docker" / "runners" / "systemd" / "omnirunners.slice"
GIB = 1024 * 1024 * 1024


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        _script_text(),
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"could not extract {name}() from deploy-runners.sh"
    return match.group(0)


def _function_body(name: str) -> str:
    return _extract_function(name)


def _run(harness: list[str], stubs: dict[str, str]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="omn19077-stub-") as stub_dir_name:
        stub_dir = Path(stub_dir_name)
        for tool, body in stubs.items():
            path = stub_dir / tool
            path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
            path.chmod(0o755)
        return subprocess.run(
            ["bash", "-c", "\n".join(harness)],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": f"{stub_dir}:/usr/bin:/bin"},
        )


_PRELUDE = [
    "set -euo pipefail",
    f'REPO_ROOT="{REPO_ROOT}"',
    'RUNNER_HOST="dummy-host"',
    'RUNNER_HOST_DIR="/dummy/runners"',
    'RUNNER_NAME_PREFIX="omninode-runner"',
    'RUNNER_ORG="OmniNode-ai"',
    "DRY_RUN=false",
    'TARGET_HOST=""',
    'log() { echo "LOG $*"; }',
    'warn() { echo "WARN $*" >&2; }',
    'err() { echo "ERR $*" >&2; exit 1; }',
    'RUNNER_SLICE_NAME="omnirunners.slice"',
    f'RUNNER_SLICE_SOURCE="{SLICE_FILE}"',
]


# ---------------------------------------------------------------------------
# The slice unit itself
# ---------------------------------------------------------------------------


def _slice_values() -> dict[str, str]:
    values: dict[str, str] = {}
    section = ""
    for raw in SLICE_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if section == "[Slice]" and "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key] = value
    return values


def _gib(value: str) -> int:
    assert value.endswith("G"), value
    return int(value[:-1])


def test_slice_bounds_the_pool_in_aggregate() -> None:
    values = _slice_values()
    for key in ("MemoryHigh", "MemoryMax", "MemorySwapMax", "CPUWeight"):
        assert key in values, f"omnirunners.slice must declare {key}"
    # Soft ceiling below the hard one, or MemoryHigh never throttles first.
    assert _gib(values["MemoryHigh"]) < _gib(values["MemoryMax"])
    # The host has 91 GiB; the pool must leave the lab lanes most of it.
    assert _gib(values["MemoryMax"]) <= 45
    # Below system.slice's default weight of 100, or CI does not yield CPU.
    assert int(values["CPUWeight"]) < 100


def test_slice_does_not_declare_an_inert_io_weight() -> None:
    """Both NVMe devices on .201 run the `none` scheduler with no io.cost, so
    an IOWeight line would be accepted by systemd and do nothing."""
    assert "IOWeight" not in _slice_values()


def test_runner_slice_expected_parses_the_unit_to_systemctl_form() -> None:
    result = _run(
        [
            *_PRELUDE,
            _extract_function("runner_slice_expected"),
            "runner_slice_expected",
        ],
        {},
    )
    assert result.returncode == 0, result.stderr
    got = dict(line.split("=", 1) for line in result.stdout.splitlines())
    values = _slice_values()
    assert got == {
        "MemoryHigh": str(_gib(values["MemoryHigh"]) * GIB),
        "MemoryMax": str(_gib(values["MemoryMax"]) * GIB),
        "MemorySwapMax": str(_gib(values["MemorySwapMax"]) * GIB),
        "CPUWeight": values["CPUWeight"],
    }


def _install_harness() -> list[str]:
    return [
        *_PRELUDE,
        "runner_config_field() { echo dummy-host; }",
        _extract_function("runner_slice_expected"),
        _extract_function("install_runner_slice"),
        "install_runner_slice",
    ]


def _systemctl_show_stub(memory_high_bytes: int) -> str:
    values = _slice_values()
    return (
        'case "$*" in\n'
        '  *"systemctl show"*)\n'
        f"    printf 'MemoryHigh={memory_high_bytes}\\n'\n"
        f"    printf 'MemoryMax={_gib(values['MemoryMax']) * GIB}\\n'\n"
        f"    printf 'MemorySwapMax={_gib(values['MemorySwapMax']) * GIB}\\n'\n"
        f"    printf 'CPUWeight={values['CPUWeight']}\\n' ;;\n"
        "  *) exit 0 ;;\n"
        "esac"
    )


def test_install_runner_slice_accepts_a_matching_readback() -> None:
    high = _gib(_slice_values()["MemoryHigh"]) * GIB
    result = _run(_install_harness(), {"ssh": _systemctl_show_stub(high)})
    assert result.returncode == 0, result.stderr
    assert "Runner slice read back" in result.stdout


def test_install_runner_slice_fails_closed_on_a_drifted_readback() -> None:
    """Positive control for the test above: the same harness with one live
    value wrong must refuse, or the readback is decoration."""
    result = _run(_install_harness(), {"ssh": _systemctl_show_stub(1)})
    assert result.returncode != 0
    assert "does not match" in result.stderr


def test_install_runner_slice_fails_closed_when_install_fails() -> None:
    result = _run(_install_harness(), {"ssh": "exit 1"})
    assert result.returncode != 0
    assert "could not install" in result.stderr


def test_slice_file_is_synced_to_the_host() -> None:
    text = _script_text()
    sync_block = re.search(r"SYNC_PATHS=\((.*?)\n\)", text, re.DOTALL)
    assert sync_block is not None
    assert '"docker/runners/systemd/omnirunners.slice"' in sync_block.group(1)

    result = _run(
        [
            *_PRELUDE,
            f'RUNNER_FLEET_CONFIG="{REPO_ROOT}/config/runner_fleet.yaml"',
            "run_ssh() { :; }",
            _extract_function("rsync_artifacts"),
            "rsync_artifacts",
        ],
        {"rsync": "printf '%s\\n' \"$@\""},
    )
    assert result.returncode == 0, result.stderr
    assert str(SLICE_FILE) in result.stdout


@pytest.mark.parametrize(
    ("function", "before"),
    [
        ("deploy_with_retry", "deploy_runners "),
        ("rolling_deploy", "fleet_services)"),
        ("retire_surplus", "host_surplus_runners)"),
    ],
)
def test_slice_is_installed_after_rsync_and_before_any_runner_is_touched(
    function: str, before: str
) -> None:
    body = _function_body(function)
    assert "rsync_artifacts" in body
    assert "install_runner_slice" in body
    assert body.index("rsync_artifacts") < body.index("install_runner_slice")
    assert body.index("install_runner_slice") < body.index(before)


# ---------------------------------------------------------------------------
# --retire-surplus
# ---------------------------------------------------------------------------


def _surplus(ssh_body: str, count: int = 40) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            *_PRELUDE,
            f"RUNNER_COUNT={count}",
            _extract_function("host_surplus_runners"),
            "host_surplus_runners",
        ],
        {"ssh": ssh_body},
    )


def test_host_surplus_runners_selects_only_general_pool_above_the_count() -> None:
    names = [f"omninode-runner-{i}" for i in range(1, 61)] + [
        "omninode-deploy-runner",
        "omninode-verify-runner-1",
        "omninode-customer-plane-runner-2",
        "omninode-prod-deploy-runner-1",
        "omnibase-infra-redpanda",
    ]
    listing = "\\n".join(reversed(names))
    result = _surplus(f"printf '{listing}\\n'")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [f"omninode-runner-{i}" for i in range(41, 61)]


def test_host_surplus_runners_is_empty_when_the_pool_is_already_capped() -> None:
    listing = "\\n".join(f"omninode-runner-{i}" for i in range(1, 41))
    result = _surplus(f"printf '{listing}\\n'")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_host_surplus_runners_fails_closed_when_the_host_is_unreadable() -> None:
    """An empty answer from a failed ssh must not read as 'nothing surplus'."""
    result = _surplus("exit 255")
    assert result.returncode != 0
    assert "refusing to decide" in result.stderr


def _idle(
    inspect_answer: str, runner_is_idle_rc: int
) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            *_PRELUDE,
            f'runner_is_idle() {{ echo "TWO-SIGNAL-CHECK"; return {runner_is_idle_rc}; }}',
            _extract_function("container_is_running"),
            _extract_function("surplus_runner_is_idle"),
            "surplus_runner_is_idle omninode-runner-41",
        ],
        {"ssh": inspect_answer},
    )


def test_a_stopped_surplus_runner_is_idle_without_asking_github() -> None:
    result = _idle("echo false", runner_is_idle_rc=1)
    assert result.returncode == 0
    assert "TWO-SIGNAL-CHECK" not in result.stdout


def test_a_running_surplus_runner_needs_the_two_signal_check() -> None:
    busy = _idle("echo true", runner_is_idle_rc=1)
    assert busy.returncode != 0
    assert "TWO-SIGNAL-CHECK" in busy.stdout
    idle = _idle("echo true", runner_is_idle_rc=0)
    assert idle.returncode == 0


def test_an_uninspectable_surplus_runner_is_treated_as_busy() -> None:
    assert _idle("exit 1", runner_is_idle_rc=0).returncode != 0
    assert _idle("echo garbage", runner_is_idle_rc=0).returncode != 0


def test_retire_one_runner_rechecks_before_stopping_and_keeps_volumes() -> None:
    body = _function_body("retire_one_runner")
    checks = [m.start() for m in re.finditer(r"surplus_runner_is_idle", body)]
    stop = body.index('ssh "${RUNNER_HOST}" "docker stop')
    assert len(checks) >= 2 and checks[-1] < stop, (
        "the idle check must be repeated immediately before the stop"
    )
    assert "docker rm ${name}" in body
    assert not re.search(r"docker rm\s+-\w*[vf]", body), "never rm -v / rm -f"


def test_retire_path_never_sweeps_orphans_or_deletes_volumes() -> None:
    block = "".join(
        _function_body(name)
        for name in (
            "host_surplus_runners",
            "container_is_running",
            "surplus_runner_is_idle",
            "deregister_retired_runner",
            "retire_one_runner",
            "retire_surplus",
        )
    )
    assert "--remove-orphans" not in block
    assert "volume rm" not in block
    assert "prune" not in block


def test_deregistration_waits_for_offline_and_names_one_runner() -> None:
    body = _function_body("deregister_retired_runner")
    assert "offline) break" in body
    assert body.index("offline) break") < body.index("-X DELETE")
    assert 'actions/runners/${id}"' in body


def test_retire_surplus_is_refused_with_other_modes() -> None:
    for other in ("--rolling", "--soft", "--add=omninode-deploy-runner"):
        result = subprocess.run(
            ["bash", str(DEPLOY_SCRIPT), "--retire-surplus", other, "--dry-run"],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )
        assert result.returncode != 0, other
        assert "mutually exclusive" in result.stderr, (other, result.stderr)
