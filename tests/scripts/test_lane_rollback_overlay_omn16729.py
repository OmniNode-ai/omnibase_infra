# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Lane rollback keeps the lane overlay, and never fires on a healthy lane [OMN-16729].

Three defects, all measured on the .201 dev lane on 2026-09-08, all recorded in
receipt ``20260908T184841Z-5773ffb03a82.json``:

1. ``refresh_dev_lane.sh``'s failure ROLLBACK recreated all four core services
   with a single ``-f docker-compose.infra.yml``. The dev lane's overlay,
   ``docker-compose.dev-lane.yml``, is the sole declaration of the runtime
   family's ``KAFKA_SECURITY_PROTOCOL`` / ``KAFKA_SASL_*`` environment, and the
   dev broker had required SASL since 18:16:19Z. The recreated runtime
   crash-looped on ``KafkaConnectionError``.
2. The rollback fired on ``revision_readback_ok=false`` ALONE, on a receipt
   carrying ``health_ok=true``, ``manifest_ok=true`` and ``errors=[]`` for a
   no-op rebuild (``digest_changed=false``). A container recreate cannot repair
   a revision-label mismatch; it can only take a serving lane down, which is
   what it did.
3. ``runtime-effects`` and ``runtime-worker`` were left at ``State=created``
   behind an unmet ``depends_on``. A created container emits zero log lines, so
   it was invisible to every log-grep triage, and the gate reported only an
   anonymous revision mismatch for 53 minutes.

These tests drive the real scripts and the real gate module -- no surrogate.
The overlay test carries its own RED CONTROL: the same assertion, run against a
reconstruction of the pre-fix single-``-f`` command, must fail.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REFRESH_DEV = REPO_ROOT / "scripts" / "runtime_build" / "refresh_dev_lane.sh"
REFRESH_STABILITY = (
    REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh"
)
COMPOSE_FILES_SH = REPO_ROOT / "scripts" / "runtime_build" / "compose_files.sh"
DEPLOY_RUNTIME = REPO_ROOT / "scripts" / "deploy-runtime.sh"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "runtime_build"))

INFRA_YML = "docker-compose.infra.yml"
DEV_LANE_YML = "docker-compose.dev-lane.yml"
STABILITY_YML = "docker-compose.stability-test.yml"


def _f_paths(argv: list[str]) -> list[str]:
    """Every value that follows a ``-f`` token, basename only."""
    return [
        Path(argv[i + 1]).name
        for i, tok in enumerate(argv)
        if tok == "-f" and i + 1 < len(argv)
    ]


# ---------------------------------------------------------------------------
# 1. The rollback recreate carries BOTH compose files
# ---------------------------------------------------------------------------


def _print_rollback_cmd(tmp_path: Path) -> list[str]:
    """Run the real script's ``--print-rollback-cmd`` and return its argv.

    ``--print-rollback-cmd`` invokes no docker and takes no lane lock, so this
    runs anywhere. ``OMNI_HOME`` is a scratch dir: the printed paths are derived
    from it, never read.
    """
    env = dict(os.environ)
    env["OMNI_HOME"] = str(tmp_path)
    env.pop("OMNIBASE_INFRA_COMPOSE_PROJECT", None)
    # The script requires docker/git/curl/jq on PATH before it will print.
    result = subprocess.run(
        ["bash", str(REFRESH_DEV), "--print-rollback-cmd"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"--print-rollback-cmd exited {result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    return result.stdout.strip().split()


@pytest.mark.unit
def test_dev_rollback_command_carries_both_compose_files(tmp_path: Path) -> None:
    """The dev rollback recreate names infra.yml AND the dev-lane overlay."""
    argv = _print_rollback_cmd(tmp_path)

    assert argv[:2] == ["docker", "compose"], argv
    files = _f_paths(argv)
    assert files == [INFRA_YML, DEV_LANE_YML], (
        "the rollback recreate must layer the dev-lane overlay -- it is the sole "
        f"declaration of the runtime family's KAFKA_SASL_* env. Got {files} from {argv}"
    )
    assert "--force-recreate" in argv, argv


@pytest.mark.unit
def test_red_control_pre_fix_rollback_command_carries_only_one_compose_file(
    tmp_path: Path,
) -> None:
    """RED CONTROL: the pre-fix command shape FAILS the assertion above.

    Without this, a test that passes proves nothing -- an assertion that both
    files appear would also pass against a command that named the overlay twice,
    or against a script that printed nothing. This reconstructs the exact
    2026-09-08 argv and shows the assertion rejects it.
    """
    infra_clone = tmp_path / "omnibase_infra"
    pre_fix_argv = [
        "docker",
        "compose",
        "-p",
        "omnibase-infra",
        "-f",
        str(infra_clone / "docker" / INFRA_YML),
        "--profile",
        "runtime",
        "up",
        "-d",
        "--no-deps",
        "--no-build",
        "--force-recreate",
        "omninode-runtime",
        "runtime-effects",
        "runtime-worker",
        "projection-api",
    ]
    assert _f_paths(pre_fix_argv) == [INFRA_YML]
    with pytest.raises(AssertionError):
        assert _f_paths(pre_fix_argv) == [INFRA_YML, DEV_LANE_YML]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("compose_project", "expected"),
    [
        ("omnibase-infra", [INFRA_YML, DEV_LANE_YML]),
        ("omnibase-infra-stability-test", [INFRA_YML, STABILITY_YML]),
        ("omnibase-infra-prod", [INFRA_YML, "docker-compose.prod.yml"]),
        ("omnibase-infra-judge", [INFRA_YML, "docker-compose.judge.yml"]),
    ],
)
def test_shared_resolver_always_emits_two_files(
    compose_project: str, expected: list[str]
) -> None:
    """Every lane resolves to exactly two ``-f`` paths -- base plus its overlay."""
    script = "\n".join(
        [
            "set -euo pipefail",
            f'source "{COMPOSE_FILES_SH}"',
            "declare -a args",
            f'resolve_compose_file_args args /target "{compose_project}"',
            'printf "%s\\n" "${args[@]}"',
        ]
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    argv = result.stdout.split()
    assert _f_paths(argv) == expected, argv


@pytest.mark.unit
def test_shared_resolver_fails_closed_on_unknown_lane() -> None:
    """An unrecognised compose project aborts rather than running on the base file."""
    script = "\n".join(
        [
            "set -euo pipefail",
            f'source "{COMPOSE_FILES_SH}"',
            "declare -a args",
            'resolve_compose_file_args args /target "omnibase-infra-typo"',
            'printf "%s\\n" "${args[@]}"',
        ]
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode != 0, result.stdout
    assert "Unknown lane" in result.stderr


@pytest.mark.unit
def test_no_script_spells_a_compose_file_path_by_hand() -> None:
    """No refresh script may name a compose yml on a ``-f`` line of its own.

    This is the class guard: the 2026-09-08 defect was not a wrong file, it was
    a SECOND copy of the file list. A new hand-spelled ``-f`` in any of these
    scripts fails here regardless of whether it happens to be correct today.
    """
    # `-f` as a compose FLAG followed by a compose file. Bash's file-existence
    # test (`[[ -f <path> ]]`) reads identically token-wise and is not an
    # invocation, so it is excluded by the leading-bracket guard.
    compose_flag = re.compile(
        r"(?<!\[)(?<!\[\[)\s-f\s+\S*docker-compose\.[a-z0-9.-]+\.yml"
    )
    bash_file_test = re.compile(r"\[\[?\s+-f\s")
    offenders: list[str] = []
    for script in (REFRESH_DEV, REFRESH_STABILITY, DEPLOY_RUNTIME):
        for lineno, line in enumerate(script.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or bash_file_test.search(stripped):
                continue
            if compose_flag.search(" " + stripped):
                offenders.append(f"{script.name}:{lineno}: {stripped}")
    assert not offenders, (
        "compose file paths must come from resolve_compose_file_args() in "
        "scripts/runtime_build/compose_files.sh, never a hand-spelled -f:\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 2. A provenance-only mismatch on a healthy lane does NOT recreate containers
# ---------------------------------------------------------------------------

# The exact shape of the 2026-09-08T18:48:41Z gate output: the lane was serving
# and the rebuild was a no-op, so the only failing dimension was the revision
# label the running containers carried.
GATE_18_48_41Z = {
    "lane": "dev",
    "require_digest_change": True,
    "digest_changed": False,
    "manifest_count": 312,
    "manifest_floor": 288,
    "manifest_ok": True,
    "health_ok": True,
    "cluster_healthy": True,
    "revision_readback_ok": False,
    "core_services_running": True,
    "core_services_not_running": [],
    "errors": [],
    "overall": "FAIL",
}


@pytest.mark.unit
def test_receipt_of_the_18_48_41z_shape_is_provenance_only(tmp_path: Path) -> None:
    """The 18:48:41Z receipt classifies as HEALTHY + provenance-failing.

    Asserts the script gates on exactly the four health dimensions, then
    reproduces its rule -- the same jq reads, over the fixture -- and shows the
    18:48:41Z receipt lands on HEALTHY with provenance-only failures.
    """
    gate_path = tmp_path / "gate1.json"
    gate_path.write_text(json.dumps(GATE_18_48_41Z))

    script_text = REFRESH_DEV.read_text()
    # The four dimensions the destructive rollback is gated on, each read from
    # the gate JSON by name. A dimension dropped from this list would silently
    # widen what counts as "healthy".
    for dim in (
        "health_ok",
        "manifest_ok",
        "cluster_healthy",
        "core_services_running",
    ):
        assert f'UNHEALTHY_DIMENSIONS+=("{dim}=false' in script_text, dim
    assert "FAILED_BUILD_PROVENANCE" in script_text

    # Reproduce the script's rule with the same jq reads it performs.
    def _jq(expr: str) -> str:
        out = subprocess.run(
            ["jq", "-r", expr, str(gate_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    unhealthy = [
        name
        for name, expr in (
            ("health_ok", ".health_ok // false"),
            ("manifest_ok", ".manifest_ok // false"),
            ("cluster_healthy", ".cluster_healthy // false"),
            ("core_services_running", ".core_services_running // false"),
        )
        if _jq(expr) != "true"
    ]
    assert _jq("(.errors // []) | length") == "0"
    assert unhealthy == [], unhealthy
    assert _jq(".revision_readback_ok // false") == "false"
    assert _jq(".digest_changed // false") == "false"


@pytest.mark.unit
def test_rollback_branch_is_guarded_by_lane_health() -> None:
    """The destructive branch is reachable only when a health dimension failed.

    Structural assertion on the real script: the provenance branch must be
    ordered BEFORE the rollback branch, so a healthy lane can never fall into
    the recreate. Ordering is the whole mechanism here -- a correct condition
    placed second would never be consulted.
    """
    text = REFRESH_DEV.read_text()
    provenance_at = text.index(
        'elif [[ "${BRANCH}" == "warm" && "${LANE_IS_HEALTHY}" == true ]]; then'
    )
    rollback_at = text.index('elif [[ "${BRANCH}" == "warm" ]]; then')
    assert provenance_at < rollback_at, (
        "the provenance branch must precede the unconditional warm rollback, "
        "or a healthy lane still gets recreated"
    )
    # And the destructive branch must not be entered on a provenance dimension.
    rollback_block = text[
        rollback_at : text.index("=== Re-verifying health after rollback ===")
    ]
    assert "revision_readback" not in rollback_block
    assert "docker tag" in rollback_block, "the retag is part of the destructive path"


# ---------------------------------------------------------------------------
# 3. The gate asserts every core service is RUNNING and names any that is not
# ---------------------------------------------------------------------------


class _FakeDocker:
    """Minimal ``subprocess.run`` stand-in over a {container: state} map."""

    def __init__(self, states: dict[str, str]) -> None:
        self.states = states

    def __call__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
        container = cmd[2]
        fmt = cmd[4]
        if ".State.Status" in fmt:
            out = self.states.get(container, "")
        elif "{{.Image}}" in fmt:
            out = f"sha256:image-of-{container}"
        else:
            out = "5773ffb"
        return subprocess.CompletedProcess(cmd, 0, out + "\n", "")


@pytest.mark.unit
def test_gate_names_a_service_stranded_in_created() -> None:
    """A container in State=created is not-running, and the gate NAMES it."""
    import verify_dev_refresh as vdr

    runner = _FakeDocker(
        {
            "c-runtime": "running",
            "c-effects": "created",
            "c-worker": "created",
            "c-api": "running",
        }
    )
    checks = [
        vdr.check_service_digest(svc, cid, None, "5773ffb", runner=runner)
        for svc, cid in (
            ("omninode-runtime", "c-runtime"),
            ("runtime-effects", "c-effects"),
            ("runtime-worker", "c-worker"),
            ("projection-api", "c-api"),
        )
    ]
    report = vdr.HealthGateReport(lane="dev", require_digest_change=False)
    report.services = checks
    report.manifest_ok = True
    report.health_ok = True
    report.cluster_healthy = True

    assert report.core_services_running is False
    assert sorted(report.core_services_not_running) == [
        "runtime-effects=created",
        "runtime-worker=created",
    ]
    assert report.overall == "FAIL"
    payload = report.to_dict()
    assert payload["core_services_running"] is False
    assert "runtime-effects=created" in payload["core_services_not_running"]


@pytest.mark.unit
def test_gate_passes_when_every_core_service_is_running() -> None:
    """Positive control: the same gate PASSes when all four are running."""
    import verify_dev_refresh as vdr

    runner = _FakeDocker({f"c-{i}": "running" for i in range(4)})
    report = vdr.HealthGateReport(lane="dev", require_digest_change=False)
    report.services = [
        vdr.check_service_digest(svc, f"c-{i}", None, "5773ffb", runner=runner)
        for i, svc in enumerate(vdr.CORE_SERVICE_NAMES)
    ]
    report.manifest_ok = True
    report.health_ok = True
    report.cluster_healthy = True

    assert report.core_services_running is True
    assert report.core_services_not_running == []
    assert report.overall == "PASS"


@pytest.mark.unit
def test_gate_reports_absent_container_as_not_running() -> None:
    """A service compose has no container for at all is named, not silently dropped."""
    import verify_dev_refresh as vdr

    check = vdr.check_service_digest("runtime-worker", None, None, "5773ffb")
    assert check.running is False
    assert check.container_state is None
    report = vdr.HealthGateReport(lane="dev", require_digest_change=False)
    report.services = [check]
    assert report.core_services_not_running == ["runtime-worker=absent"]


@pytest.mark.unit
def test_caller_resolves_container_ids_in_all_states() -> None:
    """``refresh_dev_lane.sh`` feeds the gate ``ps -aq``, not ``ps -q``.

    Without this the stranded container is never in the map and the gate cannot
    name it -- the running-only query is what made it invisible.
    """
    text = REFRESH_DEV.read_text()
    assert "compose_ps_q_any()" in text
    assert "ps -aq" in text
    gate_feed_start = text.index("declare -A NEW_CONTAINER_IDS")
    gate_feed = text[gate_feed_start : gate_feed_start + 400]
    assert "compose_ps_q_any" in gate_feed, gate_feed
