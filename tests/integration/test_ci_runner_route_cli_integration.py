# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end coverage of the routing decision as CI actually invokes it.

The unit suites exercise the handler with a typed request and the contract
suite asserts the node's shape. NEITHER covers the path that decides real
placement: a subprocess, reading the committed contract and the committed fleet
inventory off disk, writing the four lines a workflow consumes into
``GITHUB_OUTPUT`` and signalling a refusal through its exit status.

That seam has already broken twice in ways every unit test survived. The seam
value was interpolated into the shell command and reached the module as
non-JSON, so every decision came back `probe_error:seam_unparseable` and looked
exactly like an inert mechanism. And a refusal that exits non-zero is worthless
if the workflow's fail-closed floor overwrites its labels first. Both are
properties of the invocation, not of the decision, so they are tested here.

Ticket: OMN-18412
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "runner_route_decision.py"
LAB_SEAM = '["self-hosted","omnibase-ci"]'
HOSTED_SEAM = '["ubuntu-latest"]'

pytestmark = pytest.mark.integration


def _fleet_file(tmp_path: Path, online: int, busy: int) -> Path:
    path = tmp_path / "fleet.json"
    path.write_text(
        json.dumps({"ok": True, "online": online, "busy": busy, "total": online}),
        encoding="utf-8",
    )
    return path


def _lab_file(tmp_path: Path, ratio: float = 0.2, free_mem_mib: int = 40000) -> Path:
    path = tmp_path / "lab-load.json"
    path.write_text(
        json.dumps(
            {
                "sampled_at": datetime.now(UTC).isoformat(),
                "hosts": [
                    {"label": "h201", "ratio": ratio, "free_mem_mib": free_mem_mib}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(
    tmp_path: Path,
    *,
    repository: str,
    visibility: str,
    seam: str,
    fleet: Path,
    lab: Path | None = None,
    force: str = "auto",
) -> tuple[int, dict[str, str], dict[str, object]]:
    """Invoke the route CLI the way the workflow does, and read what CI reads."""
    output = tmp_path / "github_output"
    output.write_text("", encoding="utf-8")
    artifact = tmp_path / "runner-route-decision.json"
    args = [
        sys.executable,
        str(SCRIPT),
        "--event-name",
        "push",
        "--repository",
        repository,
        "--repo-visibility",
        visibility,
        "--workflow-path",
        ".github/workflows/ci.yml",
        "--seam-json",
        seam,
        "--public-json",
        HOSTED_SEAM,
        "--fleet-json",
        str(fleet),
        "--force",
        force,
    ]
    if lab is not None:
        args += ["--lab-record", str(lab)]
    completed = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        # A refusal exits 3 BY DESIGN and is one of the things under test, so a
        # non-zero status must reach the assertions rather than raise here.
        check=False,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GITHUB_OUTPUT": str(output),
            "RUNNER_ROUTE_ARTIFACT": str(artifact),
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )
    emitted: dict[str, str] = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            emitted[key] = value
    record = (
        json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else {}
    )
    return completed.returncode, emitted, record


def test_a_public_repo_with_an_idle_fleet_is_placed_on_the_fleet(
    tmp_path: Path,
) -> None:
    """The whole point of the mechanism, end to end.

    The thresholds come from the committed contract and the fleet size from the
    committed inventory -- nothing in this test supplies either, so a contract
    edit that made the fleet unreachable would fail here.
    """
    code, emitted, record = _run(
        tmp_path,
        repository="OmniNode-ai/omnibase_infra",
        visibility="public",
        seam=LAB_SEAM,
        fleet=_fleet_file(tmp_path, online=60, busy=2),
        lab=_lab_file(tmp_path),
    )
    assert code == 0
    assert json.loads(emitted["runs_on"]) == ["self-hosted", "omnibase-ci"]
    assert json.loads(emitted["labels"]) == ["self-hosted", "omnibase-ci"]
    assert emitted["decision"] == "self_hosted"
    assert emitted["reason"] == "capacity_available"
    assert record["evidence"]["idle"] == 58


def test_a_saturated_fleet_sends_a_public_repo_hosted(tmp_path: Path) -> None:
    code, emitted, _ = _run(
        tmp_path,
        repository="OmniNode-ai/omnibase_infra",
        visibility="public",
        seam=LAB_SEAM,
        fleet=_fleet_file(tmp_path, online=60, busy=58),
        lab=_lab_file(tmp_path),
    )
    assert code == 0
    assert json.loads(emitted["runs_on"]) == ["ubuntu-latest"]
    assert emitted["reason"] == "fleet_saturated"


def test_the_seam_survives_the_shell_and_reaches_the_decision(tmp_path: Path) -> None:
    """THE DEFECT THIS PINS was live for days and read as success.

    A JSON array interpolated into a quoted shell argument loses its inner
    quotes and arrives as `[self-hosted,omnibase-ci]`, which is not JSON. The
    decision then returned hosted with `probe_error:seam_unparseable` on every
    run -- indistinguishable from a mechanism that was correctly inert, because
    nothing read the reason.
    """
    code, emitted, _ = _run(
        tmp_path,
        repository="OmniNode-ai/omnibase_infra",
        visibility="public",
        seam=LAB_SEAM,
        fleet=_fleet_file(tmp_path, online=60, busy=2),
        lab=_lab_file(tmp_path),
    )
    assert code == 0
    assert emitted["reason"] != "probe_error:seam_unparseable"

    # The positive control: a genuinely unparseable seam DOES produce that
    # reason, so the assertion above is discriminating rather than vacuous.
    code, emitted, _ = _run(
        tmp_path,
        repository="OmniNode-ai/omnibase_infra",
        visibility="public",
        seam="[self-hosted,omnibase-ci]",
        fleet=_fleet_file(tmp_path, online=60, busy=2),
        lab=_lab_file(tmp_path),
    )
    assert emitted["reason"] == "probe_error:seam_unparseable"
    assert json.loads(emitted["runs_on"]) == ["ubuntu-latest"]


def test_a_private_repo_is_never_placed_hosted_by_saturation(tmp_path: Path) -> None:
    code, emitted, record = _run(
        tmp_path,
        repository="OmniNode-ai/omniweb",
        visibility="private",
        seam=LAB_SEAM,
        fleet=_fleet_file(tmp_path, online=60, busy=58),
        lab=_lab_file(tmp_path),
    )
    assert code == 0
    assert json.loads(emitted["runs_on"]) == ["self-hosted", "omnibase-ci"]
    assert emitted["reason"] == "private_repo_no_hosted_downgrade"
    assert record["evidence"]["downgrade_refused_from"] == "fleet_saturated"


def test_a_refusal_exits_nonzero_and_still_leaves_a_record(tmp_path: Path) -> None:
    """A refusal is not a crash, and it is not a placement either.

    The exit status is what stops the calling run. The outputs and the record
    are written FIRST, so the workflow's fail-closed floor -- which exists to
    hand a crashed router's run a usable runs-on -- finds a `labels=` line
    already present and cannot paper the refusal over with hosted labels.
    """
    code, emitted, record = _run(
        tmp_path,
        repository="OmniNode-ai/omniweb",
        visibility="private",
        seam=HOSTED_SEAM,
        fleet=_fleet_file(tmp_path, online=60, busy=2),
        lab=_lab_file(tmp_path),
    )
    assert code == 3
    assert json.loads(emitted["runs_on"]) == []
    assert "labels" in emitted, "the floor would overwrite a refusal without this line"
    assert emitted["decision"] == "blocked"
    assert emitted["reason"] == "private_repo_hosted_forbidden:seam_ceiling_hosted"
    assert record["evidence"]["visibility"] == "private"


def test_the_operator_override_reaches_the_decision_both_ways(tmp_path: Path) -> None:
    """The kill switch a static seam value used to provide, per run.

    Forcing hosted on an idle fleet proves the override is read at all;
    forcing fleet on a hosted-only seam proves it cannot widen past the
    ceiling, which is the half that would be dangerous if it were missing.
    """
    _, emitted, _ = _run(
        tmp_path,
        repository="OmniNode-ai/omnibase_infra",
        visibility="public",
        seam=LAB_SEAM,
        fleet=_fleet_file(tmp_path, online=60, busy=2),
        lab=_lab_file(tmp_path),
        force="hosted",
    )
    assert json.loads(emitted["runs_on"]) == ["ubuntu-latest"]
    assert emitted["reason"] == "forced_hosted"

    _, emitted, _ = _run(
        tmp_path,
        repository="OmniNode-ai/omnibase_infra",
        visibility="public",
        seam=HOSTED_SEAM,
        fleet=_fleet_file(tmp_path, online=60, busy=2),
        lab=_lab_file(tmp_path),
        force="fleet",
    )
    assert json.loads(emitted["runs_on"]) == ["ubuntu-latest"]
    assert emitted["reason"] == "seam_ceiling_hosted"
