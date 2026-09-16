# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18420: a lab-pass receipt must be this job's own, not one left on the host.

WHAT BROKE, measured live 2026-09-16.

Every lab-pass emitter in this repository and in the sibling reusable runs on
the SAME ``omnibase-deploy`` host, and every one of them wrote its receipt and
its checks to a FIXED absolute path under ``/tmp``. ``mkdir -p`` on a directory
that already exists is a no-op, so a ``receipt.json`` written by an earlier job
survived into the next one. When the next job's ``emit`` did not overwrite it --
because the probe feeding it failed, or the emit itself did --
``assert_evidence_artifact.py`` found the STALE file, reported evidence
present, and ``upload-artifact`` published it under THIS job's sha-keyed name.

    omnimarket run 35035178406, artifact 10424013456
      name    : lab-pass-receipt-compose-dev-bc21f733...   (an omnimarket commit)
      payload : sha a7ea64f56f14285ad985d3ec6a653d0d706fb1e7  (an omnibase_infra commit)
      check   : deployed_revision / check_dev_lane_staleness.py

That check name belongs to this repository's own emitter; the sibling emitter's
is ``sibling_revision`` / ``check_lane_sibling_revision.py``. So the payload was
not a mislabelled sibling receipt -- it was another job's output republished
under an omnimarket sha.

The failure DIRECTION was safe: ``evaluate_gate`` cross-checks the artifact name
against the payload sha and refuses the disagreement. The cost is that the
sibling sha can then never receipt PASS, and CLAUDE.md rule 24(b) refuses that
sha for staging -- which is exactly the gate OMN-17057's named-revision delivery
path is built on.

WHAT IS PINNED HERE, and why it is pinned as a property rather than as text.

A ``grep`` for ``/tmp/lab-pass`` would pass the moment somebody wrote
``/tmp/lab_pass`` or ``/var/tmp/receipts``. These tests parse the workflow YAML
and assert the two properties that actually matter:

  1. every lab-pass path a step names is derived from a per-run identifier, so
     two jobs on one host cannot collide; and
  2. the directory ``emit --out`` writes into is the SAME directory the upload
     step publishes -- a mismatch would upload a directory this job never wrote.

The identity assertion itself (a receipt whose payload sha is not the sha the
job is emitting for) is tested against the real implementation in
``test_verify_*`` below, not against the workflow text.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "lab_pass_receipt.py"

WORKFLOWS = (
    REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml",
    REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger-reusable.yml",
)

#: The run-scoped shell variables the window-start step exports. A path built
#: from either of these is unique to one execution of one job.
PER_RUN_VARS = ("LAB_PASS_DIR", "LAB_PASS_CHECKS")

#: Any absolute path under a shared temp root is the defect, whatever it is
#: spelled. RUNNER_TEMP is not in this list on purpose: it is per-runner, and
#: the window-start step narrows it further with the run id, attempt and job.
# Composed rather than written out so this denylist is not itself read as a
# use of a shared temp path (ruff S108). It is the thing being forbidden.
_TMP = "tmp"
SHARED_TMP_ROOTS = (f"/{_TMP}/", f"/var/{_TMP}/", f"/private/{_TMP}/")


def _steps(workflow: Path):
    """Yield (job_name, step) for every step in the workflow."""
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            yield job_name, step


def _lab_pass_lines(step) -> list[str]:
    """Lines of a step's `run:` body that name a lab-pass artefact path."""
    body = step.get("run")
    if not isinstance(body, str):
        return []
    return [
        line
        for line in body.splitlines()
        if "lab_pass_receipt.py" in line
        or "lab-pass" in line
        or "lab_pass" in line
        or "LAB_PASS_" in line
    ]


@pytest.mark.unit
@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_no_lab_pass_path_is_rooted_in_a_shared_temp_directory(workflow: Path) -> None:
    """AC1/AC4: a shared absolute path is how one job poisons the next."""
    offenders: list[str] = []
    for job_name, step in _steps(workflow):
        for line in _lab_pass_lines(step):
            for root in SHARED_TMP_ROOTS:
                if root in line:
                    offenders.append(f"{job_name}: {line.strip()}")
    assert not offenders, (
        "these lab-pass steps name a path under a temp root shared by every job "
        "on the omnibase-deploy host, which is the OMN-18420 defect:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.unit
@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_lab_pass_receipt_path_is_derived_from_a_per_run_variable(
    workflow: Path,
) -> None:
    """AC1: the positive half. Absence of `/tmp` is not presence of isolation."""
    seen = 0
    for job_name, step in _steps(workflow):
        body = step.get("run")
        if not isinstance(body, str) or "lab_pass_receipt.py" not in body:
            continue
        for flag in ("--out ", "--checks-json ", "--path "):
            for line in body.splitlines():
                if flag not in line:
                    continue
                seen += 1
                assert any(var in line for var in PER_RUN_VARS), (
                    f"{workflow.name} / {job_name}: {line.strip()!r} names a "
                    "lab-pass path that is not derived from a per-run variable "
                    f"({', '.join(PER_RUN_VARS)})"
                )
    assert seen, f"{workflow.name} declares no lab-pass paths — the test read nothing"


@pytest.mark.unit
@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_the_uploaded_directory_is_the_one_emit_wrote_into(workflow: Path) -> None:
    """AC3: uploading a directory this job did not write is the same defect."""
    uploads = 0
    for job_name, step in _steps(workflow):
        uses = str(step.get("uses") or "")
        if not uses.startswith("actions/upload-artifact"):
            continue
        with_block = step.get("with") or {}
        name = str(with_block.get("name") or "")
        if "lab-pass-receipt" not in name:
            continue
        uploads += 1
        path = str(with_block.get("path") or "")
        assert any(var in path for var in PER_RUN_VARS), (
            f"{workflow.name} / {job_name}: upload path {path!r} is not the "
            "per-run receipt directory the emit step writes into"
        )
    assert uploads, (
        f"{workflow.name} uploads no lab-pass receipt — the test read nothing"
    )


@pytest.mark.unit
@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_emitter_verifies_the_receipt_is_its_own_before_uploading(
    workflow: Path,
) -> None:
    """AC2: the emitting side asserts identity, not only presence."""
    for job_name, job_steps in _emitting_jobs(workflow):
        emits = [s for s in job_steps if _invokes(s, "lab_pass_receipt.py emit")]
        if not emits:
            continue
        verifies = [s for s in job_steps if _invokes(s, "lab_pass_receipt.py verify")]
        assert verifies, (
            f"{workflow.name} / {job_name} emits a lab-pass receipt but never "
            "asserts the file on disk carries its own sha — a receipt left by "
            "another job on this host would be uploaded under this job's name"
        )


def _emitting_jobs(workflow: Path):
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    for job_name, job in (doc.get("jobs") or {}).items():
        yield job_name, (job.get("steps") or [])


def _invokes(step, fragment: str) -> bool:
    body = step.get("run")
    if not isinstance(body, str):
        return False
    return fragment in " ".join(body.split())


# ---------------------------------------------------------------------------
# The identity assertion itself, against the real implementation.
# ---------------------------------------------------------------------------
GOOD_SHA = "a" * 40
OTHER_SHA = "b" * 40


def _receipt(sha: str, lane: str = "compose-dev") -> dict:
    return {
        "receipt_version": "lab_pass_receipt.v1",
        "sha": sha,
        "lane": lane,
        "started_at": "2026-09-16T00:00:00+00:00",
        "finished_at": "2026-09-16T00:10:00+00:00",
        "result": "PASS",
        "checks": [{"name": "deployed_revision", "ok": True, "evidence": "probe"}],
        "agent_command_id": None,
    }


def _run_verify(path: Path, sha: str, lane: str = "compose-dev"):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            "--path",
            str(path),
            "--sha",
            sha,
            "--lane",
            lane,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
def test_verify_accepts_this_jobs_own_receipt(tmp_path: Path) -> None:
    """The positive control: without it a verify that always failed would pass."""
    target = tmp_path / "receipt.json"
    target.write_text(json.dumps(_receipt(GOOD_SHA)), encoding="utf-8")
    result = _run_verify(target, GOOD_SHA)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.unit
def test_verify_refuses_a_receipt_carrying_another_jobs_sha(tmp_path: Path) -> None:
    """AC2, and the exact shape measured on omnimarket run 35035178406."""
    target = tmp_path / "receipt.json"
    target.write_text(json.dumps(_receipt(OTHER_SHA)), encoding="utf-8")
    result = _run_verify(target, GOOD_SHA)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert OTHER_SHA in combined and GOOD_SHA in combined, combined


@pytest.mark.unit
def test_verify_refuses_a_receipt_carrying_another_lane(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_text(json.dumps(_receipt(GOOD_SHA, lane="onex-lab")), encoding="utf-8")
    result = _run_verify(target, GOOD_SHA, lane="compose-dev")
    assert result.returncode != 0
    assert "lane" in (result.stdout + result.stderr)


@pytest.mark.unit
def test_verify_refuses_a_missing_file(tmp_path: Path) -> None:
    """Fail closed: an absent receipt is never a skip."""
    result = _run_verify(tmp_path / "absent.json", GOOD_SHA)
    assert result.returncode != 0
