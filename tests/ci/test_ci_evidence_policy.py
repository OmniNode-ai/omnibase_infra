# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CI evidence-artifact honesty gate (OMN-18247, epic OMN-18232 phase 3).

THE DEFECT CLASS
    An evidence-producing job reports green while producing nothing. The plan
    names three independent ways that happens, and a gate closing one leaves the
    other two open:

    1. ``if-no-files-found: warn`` makes absence indistinguishable from a
       measured empty result;
    2. ``continue-on-error`` -- at STEP level or on the ENCLOSING JOB -- makes
       every failure inside irrelevant to the run's conclusion;
    3. an ``if: always()`` uploader runs after a failed producer and uploads
       whatever is on disk, including nothing.

    This module is the gate for form 1 (OMN-18247). Forms 2 and 3 are OMN-18249
    and OMN-18251 and extend this same module, because they read the same parsed
    structure and share the same declaration surface.

WHY PARSED STRUCTURE, NEVER FILE TEXT
    A comment mentioning ``if-no-files-found: error`` must not satisfy the gate,
    and a folded scalar or an unusual quoting style must not evade it. Every
    check below reads ``yaml.safe_load`` output.

THE RED CASE IS THE REAL JOB
    ``tests/fixtures/ci_evidence_policy/dev-lane-liveness.pre-repair.yml.captured``
    is a
    byte copy of ``.github/workflows/dev-lane-liveness.yml`` at
    ``01d23ff644895da86b9eafd4e824aa95286309a0`` -- the head this gate was
    written against, where both uploaders carry ``if-no-files-found: warn`` and
    no assertion step exists. The gate must reject it. A synthetic fixture would
    have let the plan's first draft ship: that draft read step-level suppression
    only, which the motivating job does not use.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
POLICY_PATH = REPO_ROOT / "config" / "ci_evidence_policy.yaml"
FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent / "fixtures" / "ci_evidence_policy"
)

UPLOAD_ACTION = "actions/upload-artifact"
ASSERTION_SCRIPT = "assert_evidence_artifact.py"

# HALF THE MECHANISM, same shape as EXPECTED_LABELED_COMPOSE in
# tests/ci/test_compose_proof_teardown_policy.py and as the STRICT_GATE_JOBS
# registrations in scripts/ci/ci_summary_gate.py. Without this tuple, deleting an
# entry from config/ci_evidence_policy.yaml silently retires the gate for that
# artifact on a fully green run. With it, removing coverage costs a visible,
# reviewed edit to a test file.
EXPECTED_EVIDENCE_IDS: tuple[str, ...] = (
    "lab-load",
    "lab-pass-receipt-compose-dev",
    "lab-pass-receipt-onex-lab",
    "lab-pass-receipt-onex-lab-k3s",
    "saturation-record",
)


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------
def _load_workflow(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"{path} did not parse to a mapping")
    return data


def _jobs(doc: dict[str, Any]) -> dict[str, Any]:
    jobs = doc.get("jobs") or {}
    return {k: v for k, v in jobs.items() if isinstance(v, dict)}


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in (job.get("steps") or []) if isinstance(s, dict)]


def _is_uploader(step: dict[str, Any]) -> bool:
    return UPLOAD_ACTION in str(step.get("uses") or "")


def _artifact_name(step: dict[str, Any]) -> str:
    return str((step.get("with") or {}).get("name") or "")


def _if_no_files_found(step: dict[str, Any]) -> str:
    """The effective setting. An omitted key is the action's default, ``warn``."""
    with_block = step.get("with") or {}
    if "if-no-files-found" not in with_block:
        return "warn (action default, key omitted)"
    return str(with_block["if-no-files-found"])


def _is_assertion_for(step: dict[str, Any], artifact_id: str) -> bool:
    """True when ``step`` invokes the assertion script for ``artifact_id``.

    Matched on the script BASENAME because the delivery workflow checks this
    repository out into a subdirectory and invokes it as
    ``omnibase_infra/scripts/ci/assert_evidence_artifact.py``.
    """
    run = str(step.get("run") or "")
    if ASSERTION_SCRIPT not in run:
        return False
    return f"--artifact {artifact_id}" in run or f'--artifact "{artifact_id}"' in run


def load_policy(path: Path = POLICY_PATH) -> list[dict[str, Any]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = doc.get("evidence_artifacts") or []
    if not entries:
        raise AssertionError(
            f"{path} declares no evidence artifacts. An empty policy is not a "
            "clean bill of health; it is a gate that checks nothing."
        )
    return list(entries)


def all_uploaders(root: Path = WORKFLOWS_DIR) -> list[tuple[str, str, dict[str, Any]]]:
    """Every upload-artifact step in every workflow, as (file, job id, step)."""
    found: list[tuple[str, str, dict[str, Any]]] = []
    for wf in sorted(root.glob("*.y*ml")):
        doc = _load_workflow(wf)
        for job_id, job in _jobs(doc).items():
            for step in _steps(job):
                if _is_uploader(step):
                    found.append((wf.name, job_id, step))
    return found


# --------------------------------------------------------------------------
# the rule the policy declares (reusable by the fixture tests below)
# --------------------------------------------------------------------------
def check_entry(entry: dict[str, Any], workflows_dir: Path) -> list[str]:
    """Return every violation this entry produces. Empty list means compliant."""
    problems: list[str] = []
    artifact_id = str(entry.get("id") or "")
    wf_rel = str(entry.get("workflow") or "")
    job_id = str(entry.get("job") or "")
    declared_name = str(entry.get("artifact_name") or "")
    required = list(entry.get("required_paths") or [])

    if not required:
        problems.append(f"{artifact_id}: declares no required_paths")

    wf_path = workflows_dir / Path(wf_rel).name
    if not wf_path.is_file():
        problems.append(f"{artifact_id}: workflow {wf_rel} does not exist")
        return problems

    doc = _load_workflow(wf_path)
    job = _jobs(doc).get(job_id)
    if job is None:
        problems.append(f"{artifact_id}: job '{job_id}' not found in {wf_rel}")
        return problems

    steps = _steps(job)
    uploader_index: int | None = None
    for idx, step in enumerate(steps):
        if _is_uploader(step) and _artifact_name(step) == declared_name:
            uploader_index = idx
            break

    if uploader_index is None:
        problems.append(
            f"{artifact_id}: no upload-artifact step in {wf_rel}:{job_id} names "
            f"'{declared_name}'. A declaration that resolves to nothing is a gate "
            f"that checks nothing."
        )
        return problems

    uploader = steps[uploader_index]

    # AC1, first half: absence must be a failure, never a warning.
    setting = _if_no_files_found(uploader)
    if setting != "error":
        problems.append(
            f"{artifact_id}: uploader in {wf_rel}:{job_id} has "
            f"if-no-files-found: {setting}; declared evidence artifacts require "
            f"'error'."
        )

    # AC1, second half: the zero-byte case if-no-files-found cannot see.
    assertion_index: int | None = None
    for idx in range(uploader_index):
        if _is_assertion_for(steps[idx], artifact_id):
            assertion_index = idx
            break

    if assertion_index is None:
        problems.append(
            f"{artifact_id}: no step before the uploader in {wf_rel}:{job_id} runs "
            f"{ASSERTION_SCRIPT} --artifact {artifact_id}. if-no-files-found: error "
            f"fires only when NO path matches, so a zero-byte file satisfies it."
        )
        return problems

    assertion = steps[assertion_index]

    # An assertion that does not run when the producer failed is decorative:
    # the always() uploader would still upload the empty file.
    condition = str(assertion.get("if") or "")
    if "always()" not in condition:
        problems.append(
            f"{artifact_id}: the assertion step in {wf_rel}:{job_id} is not guarded "
            f"by if: always() (found: {condition!r}). A failed producer would skip it "
            f"and the uploader would still run."
        )

    if assertion.get("continue-on-error"):
        problems.append(
            f"{artifact_id}: the assertion step in {wf_rel}:{job_id} carries "
            f"continue-on-error, so it cannot fail the job it is asserting on."
        )

    return problems


# --------------------------------------------------------------------------
# tests over the live tree
# --------------------------------------------------------------------------
def test_policy_declares_the_expected_ids() -> None:
    """Non-shrink guard: dropping an entry must cost a visible test edit."""
    declared = tuple(sorted(str(e.get("id")) for e in load_policy()))
    assert declared == EXPECTED_EVIDENCE_IDS, (
        "config/ci_evidence_policy.yaml's declared set drifted from the set pinned "
        "here. Adding coverage means adding the id to EXPECTED_EVIDENCE_IDS; "
        "REMOVING coverage is a decision, and this assertion is where it gets made."
    )


def test_declared_ids_are_unique() -> None:
    ids = [str(e.get("id")) for e in load_policy()]
    assert len(ids) == len(set(ids)), f"duplicate evidence artifact ids: {ids}"


@pytest.mark.parametrize("entry", load_policy(), ids=lambda e: str(e.get("id")))
def test_declared_evidence_artifact_is_asserted_at_its_uploader(
    entry: dict[str, Any],
) -> None:
    """AC1: an artifact that is zero bytes or absent fails its job at the uploader."""
    problems = check_entry(entry, WORKFLOWS_DIR)
    assert not problems, "\n".join(problems)


def test_every_fatal_absence_uploader_is_declared() -> None:
    """Structural membership: ``if-no-files-found: error`` IS the evidence claim.

    Its author is stating that absence is fatal. Such an uploader cannot opt out
    of the policy by being left out of the declaration file; the only way out is
    to downgrade the setting, which is a visible change to the workflow.
    """
    declared_names = {str(e.get("artifact_name")) for e in load_policy()}
    undeclared: list[str] = []
    for wf_name, job_id, step in all_uploaders():
        if (step.get("with") or {}).get("if-no-files-found") != "error":
            continue
        if _artifact_name(step) not in declared_names:
            undeclared.append(f"{wf_name}:{job_id} -> {_artifact_name(step)}")
    assert not undeclared, (
        "these uploaders set if-no-files-found: error (declaring absence fatal) but "
        "are absent from config/ci_evidence_policy.yaml:\n  " + "\n  ".join(undeclared)
    )


# --------------------------------------------------------------------------
# RED control: the real workflow at the head this gate was written against
# --------------------------------------------------------------------------
PRE_REPAIR_FIXTURE = FIXTURES_DIR / "dev-lane-liveness.pre-repair.yml.captured"


def test_fixture_is_present() -> None:
    """A missing fixture must fail loudly rather than silently skip the RED case."""
    assert PRE_REPAIR_FIXTURE.is_file(), (
        f"{PRE_REPAIR_FIXTURE} is missing. Without it nothing proves this gate is "
        "red on the tree it was written to reject, and a gate only ever seen green "
        "is indistinguishable from a gate that checks nothing."
    )


@pytest.mark.parametrize("artifact_id", ["lab-load", "saturation-record"])
def test_gate_is_red_on_the_pre_repair_lab_lane_workflow(
    artifact_id: str, tmp_path: Path
) -> None:
    """The plan's fixture: the real job at 01d23ff6, which the gate must reject."""
    staged = tmp_path / "workflows"
    staged.mkdir()
    (staged / "dev-lane-liveness.yml").write_text(
        PRE_REPAIR_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    entry = next(e for e in load_policy() if str(e.get("id")) == artifact_id)
    problems = check_entry(entry, staged)
    assert problems, (
        f"the gate passed {artifact_id} on the pre-repair lab-lane workflow. That "
        "tree carries if-no-files-found: warn and no assertion step, which is "
        "exactly the shape this gate exists to reject."
    )
    joined = "\n".join(problems)
    assert "if-no-files-found" in joined
    assert ASSERTION_SCRIPT in joined


# --------------------------------------------------------------------------
# the assertion script itself: red on absent and on zero bytes, green on bytes
# --------------------------------------------------------------------------
def _run_assertion(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "ci" / ASSERTION_SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_assertion_fails_on_an_absent_file(tmp_path: Path) -> None:
    result = _run_assertion(
        ["--artifact", "probe", "--require", "missing.json"], tmp_path
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ABSENT" in result.stdout


def test_assertion_fails_on_a_zero_byte_file(tmp_path: Path) -> None:
    """The lab-load probe's actual failure: the redirect creates the file, the
    producer dies, and the file is zero bytes. ``if-no-files-found: error``
    cannot see this, because the path matches."""
    (tmp_path / "lab-load.json").write_text("", encoding="utf-8")
    result = _run_assertion(
        ["--artifact", "lab-load", "--require", "lab-load.json"], tmp_path
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "EMPTY" in result.stdout


def test_assertion_passes_on_a_measured_record(tmp_path: Path) -> None:
    """The green control. An assertion that is never seen to pass proves nothing."""
    (tmp_path / "lab-load.json").write_text(
        '{"busy": 3, "idle": 9}\n', encoding="utf-8"
    )
    result = _run_assertion(
        ["--artifact", "lab-load", "--require", "lab-load.json"], tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "present and non-empty" in result.stdout


def test_assertion_refuses_an_invocation_with_nothing_to_assert(tmp_path: Path) -> None:
    result = _run_assertion(["--artifact", "probe"], tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr


def test_assertion_fails_when_only_one_of_several_paths_is_empty(
    tmp_path: Path,
) -> None:
    """The directory-upload case: a sibling file makes the uploader happy while
    the file carrying the claim is empty."""
    (tmp_path / "record.json").write_text('{"ok": true}\n', encoding="utf-8")
    (tmp_path / "alerts.json").write_text("", encoding="utf-8")
    result = _run_assertion(
        [
            "--artifact",
            "saturation-record",
            "--require",
            "record.json",
            "--require",
            "alerts.json",
        ],
        tmp_path,
    )
    assert result.returncode == 1, result.stdout + result.stderr
