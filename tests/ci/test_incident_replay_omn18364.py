# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for OMN-18364 (OMN-15547 rule R1).

THE INCIDENT. ``4136c7d2870258d5bc13454cf7d53dba0f9058c8`` added the OMN-18316
step ``Assert the lab rollout deadlines are bounded under the applier wait`` to
``deliver-dev-candidate-to-staging.yml``'s ``candidate-boot-gate`` job. That
job runs on ``ubuntu-latest`` and installs ``uv`` — ninety lines BELOW the new
step. The step invokes ``uv run``, so it exited 127 with ``uv: command not
found`` on its first run and every run after it. Measured on run 34813917209:
the step started at 2026-09-14T07:14:51Z and completed at 2026-09-14T07:14:51Z,
zero seconds, because the shell never found the binary.

WHY THAT WAS WORSE THAN A RED STEP. The step's outcome is interpolated into the
onex-lab lab-pass receipt as ``rollout_deadline_bounded``. The receipt from that
run reads ``result: FAIL`` with ``rollout_deadline_bounded: fail`` and the
evidence string "every Deployment's progressDeadlineSeconds under
apply_lab_lane.sh's own 900s/300s rollout waits" — which is a substantive-looking
verdict on an invariant nothing had examined. The lab-pass gate is fail-closed
by design, so ``Announce the bundle to omninode_infra`` was skipped by
``needs:`` and NO candidate reached staging from 2026-09-13T15:03:41Z (the last
successful delivery, head 8f7417501e1a) onward. The C1 staging deploys then
failed on version skew — onex-api 0.4.81 against a runtime plane frozen at
0.4.75 — because the plane can only advance on an announced candidate.

THE ARTIFACT is the workflow file verbatim at ``4136c7d28``, fetched as a git
object, not retyped. Its sha256 is pinned in
``tests/incident_replays/registry.yaml``.

THE ACCEPT CONTROL is the repaired file in the working tree. A guard that
rejects every workflow would replay this incident perfectly and fail the whole
repository; the control is what tells the two apart.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.ci.check_workflow_uv_available import scan_document, scan_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests/fixtures/omn18364"
    / "deliver-dev-candidate-to-staging.4136c7d28.yml.captured"
)
FIXTURE_SHA256 = "98fcf87a5e2995d2314eaefe9ef63114ee54ac1f121d33254ae0a97c65b2a597"
LIVE = REPO_ROOT / ".github/workflows/deliver-dev-candidate-to-staging.yml"

GATE_JOB = "candidate-boot-gate"
DEADLINE_STEP = "Assert the lab rollout deadlines are bounded under the applier wait"


@pytest.fixture(scope="module")
def captured_bytes() -> bytes:
    raw = FIXTURE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == FIXTURE_SHA256, (
        f"{FIXTURE.name} no longer hashes to the sha256 pinned in "
        "tests/incident_replays/registry.yaml. A reformatted artifact is no "
        "longer the artifact that failed; re-fetch it with "
        "`git cat-file -p 4136c7d2870258d5bc13454cf7d53dba0f9058c8:"
        ".github/workflows/deliver-dev-candidate-to-staging.yml` rather than "
        "updating the hash."
    )
    return raw


def test_the_real_shipped_workflow_is_rejected_by_the_real_checker(
    captured_bytes: bytes,
) -> None:
    """The bytes that shipped, through the guard's own entry point."""
    document = yaml.safe_load(captured_bytes.decode("utf-8"))
    violations = scan_document(FIXTURE.name, document, REPO_ROOT)
    assert [(v.job, v.step) for v in violations] == [(GATE_JOB, DEADLINE_STEP)], (
        "the checker did not reject the step that stopped every candidate "
        f"reaching staging; it reported {violations!r}"
    )


def test_the_captured_job_really_did_install_uv_only_afterwards(
    captured_bytes: bytes,
) -> None:
    """The defect is ORDER, and the artifact has to exhibit that, not absence.

    A guard that merely asked "does this job install uv?" would have reported
    the shipped file clean. This asserts the artifact is the hard case: the
    installer is present in the same job, below the step that needs it.
    """
    document = yaml.safe_load(captured_bytes.decode("utf-8"))
    steps = document["jobs"][GATE_JOB]["steps"]
    deadline_index = next(
        index for index, step in enumerate(steps) if step.get("name") == DEADLINE_STEP
    )
    installer_indexes = [
        index
        for index, step in enumerate(steps)
        if "astral-sh/setup-uv" in str(step.get("uses") or "")
    ]
    assert installer_indexes, "the captured job installs uv nowhere; wrong artifact"
    assert min(installer_indexes) > deadline_index, (
        "in the captured artifact the installer already precedes the step, so "
        "this fixture does not exhibit the incident"
    )
    assert document["jobs"][GATE_JOB]["runs-on"] == "ubuntu-latest", (
        "the incident is specific to hosted images, where uv is genuinely "
        "absent; a fixture on a fleet label proves nothing"
    )


def test_the_repaired_workflow_is_accepted(captured_bytes: bytes) -> None:
    """The accept control.

    Without it, a checker that reported every workflow as a violation would
    pass the replay above and fail the entire repository.
    """
    assert scan_paths([LIVE], REPO_ROOT) == []
    assert LIVE.read_bytes() != captured_bytes, (
        "the live workflow is byte-identical to the captured artifact, so the "
        "accept control and the replay are the same file and one of them is "
        "lying"
    )
