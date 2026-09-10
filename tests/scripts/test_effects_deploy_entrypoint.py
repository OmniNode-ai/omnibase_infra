# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the real shell entrypoint/argument seam with external phases replaced."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy-runtime.sh"
pytestmark = pytest.mark.unit


def function(name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{.*?^\}}", SCRIPT.read_text(), re.M | re.S)
    assert match is not None
    return match[0]


def invoke(
    tmp_path: Path, args: list[str], overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    shell = tmp_path / "entrypoint.sh"
    shell.write_text(
        "\n".join(
            [
                "set -euo pipefail",
                "MODE=dry-run; EFFECTS_PLAN=; FORCE=false; RESTART=false; COLD_FULL_BRINGUP=false; PROD_LANE=false; PRINT_COMPOSE_CMD=false; COMPOSE_PROFILE=runtime",
                "DEPLOY_INVOCATION_ARGS=()",
                'log_error() { echo "$*" >&2; }',
                'run_scoped_effects_deploy() { printf \'SCOPED:%s:%s\\n\' "$MODE" "$EFFECTS_PLAN"; }',
                "validate_prerequisites() { echo FORBIDDEN_FULL_LANE_PATH >&2; exit 99; }",
                function("parse_args"),
                function("main"),
                'main "$@"',
            ]
        )
    )
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME"}}
    env.update(overrides or {})
    return subprocess.run(
        ["bash", str(shell), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize("execute", [False, True])
def test_scoped_entrypoint_never_enters_generic_phases(
    tmp_path: Path, execute: bool
) -> None:
    args = ["--effects-plan", str(tmp_path / "plan.json")]
    if execute:
        args.append("--execute")
    result = invoke(tmp_path, args)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("SCOPED:execute:" if execute else "SCOPED:dry-run:")
    assert "FORBIDDEN_FULL_LANE_PATH" not in result.stderr


@pytest.mark.parametrize(
    "flag",
    ["--restart", "--cold", "--force", "--prod", "--print-compose-cmd", "--profile"],
)
def test_conflicting_flags_refuse_before_any_phase(tmp_path: Path, flag: str) -> None:
    args = ["--effects-plan", "plan.json", "--execute", flag]
    if flag == "--profile":
        args.append("runtime")
    result = invoke(tmp_path, args)
    assert result.returncode == 64
    assert "SCOPED" not in result.stdout
    assert "FORBIDDEN" not in result.stderr


@pytest.mark.parametrize(
    "name",
    [
        "DEPLOY_REF",
        "BUILD_SOURCE",
        "EXPECTED_BUILD_SOURCE",
        "RUNTIME_BUILD_SERVICES_OVERRIDE",
        "DEPLOY_HOTPATCH",
        "ALLOW_SIBLING_PIN_DRIFT",
        "ALLOW_UNPINNED_DEPLOY_SOURCE",
        "HOTPATCH_PREFLIGHT_BYPASS",
    ],
)
def test_conflicting_environment_refuses_before_any_phase(
    tmp_path: Path, name: str
) -> None:
    result = invoke(tmp_path, ["--effects-plan", "plan.json"], {name: "1"})
    assert result.returncode == 64
    assert "SCOPED" not in result.stdout


def test_unscoped_invocation_retains_generic_path(tmp_path: Path) -> None:
    result = invoke(tmp_path, [])
    assert result.returncode == 99
    assert "FORBIDDEN_FULL_LANE_PATH" in result.stderr


@pytest.mark.parametrize("pid", [None, "", "corrupt", "2147483647", "1"])
def test_generic_lock_never_reclaims_existing_owner(
    tmp_path: Path, pid: str | None
) -> None:
    lock = tmp_path / ".deploy.lock"
    lock.mkdir()
    if pid is not None:
        (lock / "pid").write_text(pid)
    identity = lock.stat().st_ino
    script = "\n".join(
        [
            "set -euo pipefail",
            "DEPLOY_ROOT=" + shlex.quote(str(tmp_path)),
            "LOCK_DIR=" + shlex.quote(str(lock)),
            'log_error() { echo "$*" >&2; }',
            'log_info() { echo "$*"; }',
            "cleanup_on_exit() { echo FORBIDDEN_CLEANUP; }",
            function("acquire_lock"),
            "acquire_lock",
        ]
    )
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "Refusing automatic lock removal" in result.stderr
    assert "FORBIDDEN_CLEANUP" not in result.stdout
    assert lock.stat().st_ino == identity
    if pid is not None:
        assert (lock / "pid").read_text() == pid


def test_scoped_launcher_uses_shared_attribution_and_candidate_executor(
    tmp_path: Path,
) -> None:
    """Execute the real launcher, not a substitute implementation."""
    shell = tmp_path / "launcher.sh"
    shell.write_text(
        "\n".join(
            [
                "set -euo pipefail",
                "MODE=execute; EFFECTS_PLAN='plan with spaces.json'",
                "resolve_repo_root() { echo " + shlex.quote(str(tmp_path)) + "; }",
                "resolve_compose_project() { echo omnibase-infra; }",
                "check_command() { :; }",
                'log_error() { echo "$*" >&2; }',
                'guard_lane_deploy_attribution() { test "$MODE" = dry-run; echo ATTRIBUTION; }',
                # The generic host-HEAD hotpatch gate must not be used for an image.
                "guard_hotpatch_ledger() { echo WRONG_HOST_REFS >&2; exit 99; }",
                "uv() { printf 'ARG:%s\\n' \"$@\"; }",
                function("run_scoped_effects_deploy"),
                "run_scoped_effects_deploy",
            ]
        )
    )
    result = subprocess.run(
        ["bash", str(shell)],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ATTRIBUTION" in result.stdout
    assert "ARG:plan with spaces.json\n" in result.stdout
    assert "ARG:--execute\n" in result.stdout
    assert "scoped_effects_deploy.py" in result.stdout
    assert "WRONG_HOST_REFS" not in result.stderr
