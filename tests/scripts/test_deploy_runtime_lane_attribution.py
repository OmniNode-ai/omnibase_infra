# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""deploy-runtime.sh must enforce lane-deploy attribution + the grant interlock (OMN-15218).

Before OMN-15218 the sanctioned deploy path recorded WHAT was deployed
(registry.json) but nothing recorded WHO deployed it or WHY, and nothing checked
whether live prod-promotion grants were pinned to the proof the deploy was about
to replace. Two stability-lane rebuilds in two days (2026-07-26T21:45Z,
2026-07-27T10:05-10:09Z) were consequently unattributable.

Two kinds of test here:

  * wiring assertions over the script text (the repo's existing idiom for
    deploy-runtime.sh gates), and
  * an EXECUTED harness that extracts the guard function and runs it in bash
    against a stub preflight, proving the seam actually hard-fails and actually
    captures the record — not merely that the tokens appear in the file.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh"
PREFLIGHT = REPO_ROOT / "scripts" / "preflight_lane_deploy_attribution.py"


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\s*\(\)\s*\{{.*?^\}}", text, re.DOTALL | re.MULTILINE)
    assert match is not None, f"{name}() not found in deploy-runtime.sh"
    return match.group(0)


# --- wiring ------------------------------------------------------------------


@pytest.mark.unit
def test_preflight_script_exists_and_is_the_single_source_of_truth() -> None:
    assert PREFLIGHT.is_file(), "the attribution/interlock preflight must exist"
    assert "scripts/preflight_lane_deploy_attribution.py" in _script_text()


@pytest.mark.unit
def test_defines_and_calls_the_attribution_guard() -> None:
    text = _script_text()
    assert re.search(r"^guard_lane_deploy_attribution\s*\(\)", text, re.MULTILINE)
    # Defined AND called (definition + main() invocation).
    assert text.count("guard_lane_deploy_attribution") >= 2


@pytest.mark.unit
def test_guard_hard_fails_rather_than_warning() -> None:
    body = _extract_function(_script_text(), "guard_lane_deploy_attribution")
    assert "exit 1" in body, "the attribution guard must hard-fail, never warn"
    assert "log_warn" not in body, (
        "a warning-only attribution guard is the old behavior"
    )


@pytest.mark.unit
def test_guard_runs_before_any_mutation() -> None:
    """The guard must precede build/sync/restart, not follow them."""
    text = _script_text()
    main_body = text[text.index("\nmain() {") :]
    guard_at = main_body.index("guard_lane_deploy_attribution ")
    for later in (
        "sync_files ",
        "build_images ",
        "restart_services ",
        "bringup_full_stack ",
        "write_registry ",
    ):
        assert guard_at < main_body.index(later), (
            f"{later.strip()} must run AFTER the attribution guard"
        )


@pytest.mark.unit
def test_registry_carries_the_attribution_record() -> None:
    body = _extract_function(_script_text(), "write_registry")
    assert "attribution: $attribution" in body, (
        "registry.json must carry who/why, not just what"
    )


@pytest.mark.unit
def test_refresh_stability_lane_runs_the_preflight_before_it_mutates_anything() -> None:
    text = REFRESH_SCRIPT.read_text(encoding="utf-8")
    assert "preflight_lane_deploy_attribution.py" in text
    preflight_at = text.index("ATTRIBUTION_PREFLIGHT=")
    # docker tag (rollback anchor) and the ambient-clone checkout are this
    # script's own mutations; both happen after deploy-runtime.sh is chosen but
    # BEFORE it is invoked, so the preflight has to precede them here too.
    assert preflight_at < text.index('    docker tag "')
    assert preflight_at < text.index("checkout --force --detach")
    assert "attribution: $attribution" in text, (
        "the refresh receipt must carry the attribution record"
    )


@pytest.mark.unit
def test_lane_derivation_is_shared_not_duplicated() -> None:
    text = _script_text()
    assert re.search(r"^resolve_lane_name\s*\(\)", text, re.MULTILINE)
    hotpatch = _extract_function(text, "guard_hotpatch_ledger")
    assert "resolve_lane_name" in hotpatch, (
        "hot-patch guard must reuse the shared lane derivation"
    )


# --- executed harness --------------------------------------------------------


HARNESS_PRELUDE = """
set -uo pipefail
log_step() { printf '[step] %s\\n' "$*" >&2; }
log_info() { printf '[info] %s\\n' "$*" >&2; }
log_warn() { printf '[warn] %s\\n' "$*" >&2; }
log_error() { printf '[error] %s\\n' "$*" >&2; }
log_cmd()  { printf '[cmd] %s\\n' "$*" >&2; }
SCRIPT_NAME="deploy-runtime.sh"
DEPLOY_INVOCATION_ARGS=(--execute --restart)
LANE_ATTRIBUTION_RECORD_JSON=""
ONEX_DEPLOY_REASON_VAR="ONEX_DEPLOY_REASON"
ONEX_DEPLOY_GRANT_ACK_VAR="ONEX_DEPLOY_GRANT_ACK"
MODE="execute"
"""

STUB_PREFLIGHT = """#!/usr/bin/env python3
import json, os, sys
print(json.dumps({"result": "REFUSE" if os.environ.get("STUB_REFUSE") else "ALLOW",
                  "lane": "stability-test", "argv": sys.argv[1:]}))
sys.exit(1 if os.environ.get("STUB_REFUSE") else 0)
"""


def _harness(
    tmp_path: Path, *, refuse: bool, drop_preflight: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run the real guard function from deploy-runtime.sh against a stub preflight."""
    fake_repo = tmp_path / "repo"
    (fake_repo / "scripts").mkdir(parents=True)
    (fake_repo / ".venv" / "bin").mkdir(parents=True)
    # Pin python_bin resolution to this interpreter so the harness never depends
    # on uv/system python being present or on a pyproject in the fake repo.
    venv_python = fake_repo / ".venv" / "bin" / "python"
    venv_python.symlink_to(sys.executable)

    if not drop_preflight:
        stub = fake_repo / "scripts" / "preflight_lane_deploy_attribution.py"
        stub.write_text(STUB_PREFLIGHT, encoding="utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    text = _script_text()
    script = "\n".join(
        [
            HARNESS_PRELUDE,
            _extract_function(text, "resolve_lane_name"),
            _extract_function(text, "guard_lane_deploy_attribution"),
            'guard_lane_deploy_attribution "$1" "$2"',
            'printf "RECORD:%s\\n" "${LANE_ATTRIBUTION_RECORD_JSON}"',
        ]
    )
    harness = tmp_path / "harness.sh"
    harness.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    if refuse:
        env["STUB_REFUSE"] = "1"
    else:
        env.pop("STUB_REFUSE", None)
    return subprocess.run(
        ["bash", str(harness), str(fake_repo), "omnibase-infra-stability-test"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_guard_aborts_the_deploy_when_the_preflight_refuses(tmp_path: Path) -> None:
    """RED on the old behavior: the deploy continued regardless."""
    result = _harness(tmp_path, refuse=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "RECORD:" not in result.stdout, (
        "execution must stop at the refusal, not continue"
    )
    assert "REFUSED this deploy" in result.stderr


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_guard_captures_the_record_and_passes_lane_identity(tmp_path: Path) -> None:
    result = _harness(tmp_path, refuse=False)
    assert result.returncode == 0, result.stdout + result.stderr
    captured = result.stdout.split("RECORD:", 1)[1].strip()
    record = json.loads(captured)
    argv = record["argv"]
    assert "--lane" in argv and argv[argv.index("--lane") + 1] == "stability-test"
    assert "--compose-project" in argv
    assert argv[argv.index("--compose-project") + 1] == "omnibase-infra-stability-test"
    assert (
        "--source" in argv and argv[argv.index("--source") + 1] == "deploy-runtime.sh"
    )
    assert "--check-only" not in argv, "execute mode must write the durable record"


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_guard_refuses_when_the_preflight_is_missing(tmp_path: Path) -> None:
    """Deleting the mechanism must not silently restore the old behavior."""
    result = _harness(tmp_path, refuse=False, drop_preflight=True)
    assert result.returncode == 1
    assert "attribution preflight not found" in result.stderr
