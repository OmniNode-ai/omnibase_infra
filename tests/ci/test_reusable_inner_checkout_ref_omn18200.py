# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200: a reusable workflow's SELF-checkout must resolve to the commit the
workflow file itself was loaded from.

WHAT BROKE, measured live 2026-09-15.

``omnimarket`` calls this repo's ``runtime-rebuild-trigger-reusable.yml`` at a
pinned commit (``...@e95eb9ba1bbf3a76924d32cf0925d830a135edcb``). That pin
governs exactly one thing: which workflow FILE loads. It cannot reach a SECOND,
independently-pinned ``actions/checkout`` inside that file, and both of this
repo's self-checkouts in that workflow were written ``ref: dev``. So the YAML
came from the pinned commit while the SCRIPTS it invokes came from whatever
``dev`` held at run time, and the two drifted apart the moment a script's CLI
changed.

That is what happened. OMN-18387 made ``--projection-url`` a REQUIRED argument
of ``scripts/ci/lab_pass_receipt.py probe-lane`` and updated the caller in the
same commit -- correctly, atomically, in this repo. ``omnimarket``'s pinned copy
of the YAML still invoked ``probe-lane`` without it, against the new ``dev``
script:

    omnimarket run 35019423922 (2026-09-15T20:24Z)
      lab_pass_receipt.py probe-lane: exit 2 (required --projection-url absent)
      -> "refusing to emit an invalid receipt"
      -> a FAIL compose-dev lab-pass receipt

A FAIL receipt is not cosmetic: under CLAUDE.md rule 24(b) the delivery of a sha
to staging fails closed without a PASS receipt for it, so a CLI skew inside a
probe step is a delivery outage.

``github.sha`` is not the fix and is strictly worse: inside a reusable workflow
the ``github`` context is the CALLER's, so ``github.sha`` is a commit in
omnimarket that does not exist in this repository at all.
``github.job_workflow_sha`` is the only expression meaning "the commit this
workflow file was loaded from". It puts the workflow and the code it invokes on
one immutable commit behind the SINGLE caller-side pin, so they cannot diverge.

This is the OMN-16723 invariant, whose reference implementation is omniclaude's
``.github/workflows/kb-doc-gate-reusable.yml`` (an input defaulting to
``github.job_workflow_sha``). That gate's test lives in omniclaude and scans
only omniclaude's own workflows, which is why this repository's instances were
never linted and survived. This test is that lint, scoped to this repository.

SCOPE, stated as what is checked rather than as an aspiration:

- Every ``.github/workflows/*.yml`` whose ``on:`` block declares
  ``workflow_call``. A workflow that is not reusable has no
  ``job_workflow_sha`` to pin to and is out of scope by construction.
- Within those, only ``actions/checkout`` steps whose ``repository:`` names THIS
  repository. A bare checkout of the caller's tree is the normal case and is not
  touched.
- THIRD-REPO checkouts are deliberately NOT asserted here. They cannot use
  ``job_workflow_sha`` (that commit does not exist over there), so pinning them
  is a different change with a different blast radius -- it is OMN-16726's
  scope, and claiming it here would make this test assert something it does not
  check.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

SELF_REPOSITORY = "OmniNode-ai/omnibase_infra"

# The three accepted forms, matching the OMN-16723 reference implementation:
# the context itself, an input expression that falls back to it, or a literal
# 40-character commit SHA.
_JOB_WORKFLOW_SHA = re.compile(r"github\.job_workflow_sha")
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

# The escape hatch is an annotation in the workflow file, never a silent
# allowlist here -- a greppable debt list that must cite a ticket.
_ANNOTATION = re.compile(
    r"#\s*reusable-checkout-ref-ok:\s*(?P<job>\S+)\s+(?P<ticket>OMN-\d+)\s+\S"
)


def _reusable_workflows() -> list[Path]:
    paths = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(
        WORKFLOWS_DIR.glob("*.yaml")
    ):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            continue
        # PyYAML resolves the bare key `on:` to the boolean True (YAML 1.1).
        triggers = doc.get("on", doc.get(True))
        if isinstance(triggers, dict) and "workflow_call" in triggers:
            paths.append(path)
    return paths


def _self_checkout_steps(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return (job_id, step) for every actions/checkout of THIS repository."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    found: list[tuple[str, dict[str, Any]]] = []
    for job_id, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            if not isinstance(step, dict):
                continue
            uses = str(step.get("uses") or "")
            if not uses.startswith("actions/checkout@"):
                continue
            with_block = step.get("with") or {}
            if str(with_block.get("repository") or "").strip() == SELF_REPOSITORY:
                found.append((job_id, step))
    return found


def _annotated_jobs(path: Path) -> set[str]:
    return {
        match.group("job")
        for match in _ANNOTATION.finditer(path.read_text(encoding="utf-8"))
    }


def test_the_lint_has_something_to_check() -> None:
    """Positive control: a zero-row sweep must not read as a clean bill of health.

    CLAUDE.md rule 16. If a refactor moves these workflows or renames the
    directory, every assertion below passes vacuously and the invariant is
    silently unenforced. This test is what makes that show up as red.
    """
    reusable = _reusable_workflows()
    assert reusable, f"no reusable workflows found under {WORKFLOWS_DIR}"

    self_checkouts = [
        (path.name, job_id)
        for path in reusable
        for job_id, _ in _self_checkout_steps(path)
    ]
    assert self_checkouts, (
        "no self-checkout steps found in any reusable workflow -- the lint below "
        "would pass without checking anything"
    )


@pytest.mark.parametrize("workflow_path", _reusable_workflows(), ids=lambda p: p.name)
def test_self_checkout_is_pinned_to_the_workflows_own_commit(
    workflow_path: Path,
) -> None:
    """A self-checkout inside a reusable workflow must not float on a branch."""
    annotated = _annotated_jobs(workflow_path)

    for job_id, step in _self_checkout_steps(workflow_path):
        if job_id in annotated:
            continue
        ref = str((step.get("with") or {}).get("ref") or "").strip()
        step_name = step.get("name") or "<unnamed step>"

        assert ref, (
            f"{workflow_path.name} job '{job_id}' step '{step_name}': checks out "
            f"{SELF_REPOSITORY} with NO ref, which resolves to the default branch "
            "at run time. A caller's `uses: ...@<sha>` pin cannot reach it, so the "
            "workflow YAML and the scripts it invokes drift apart silently. Use "
            "`ref: ${{ inputs.<x> || github.job_workflow_sha }}`."
        )

        accepted = bool(_JOB_WORKFLOW_SHA.search(ref)) or bool(_FULL_SHA.match(ref))
        assert accepted, (
            f"{workflow_path.name} job '{job_id}' step '{step_name}': `ref: {ref}` "
            "floats. Inside a reusable workflow only `github.job_workflow_sha` "
            "means 'the commit this workflow file was loaded from'; `github.sha` is "
            "the CALLER's commit and does not exist in this repository. "
            "Pin to `github.job_workflow_sha` (directly or as an input default), "
            "or to a 40-character SHA."
        )


def test_probe_lane_caller_and_script_are_pinned_together() -> None:
    """The specific pairing whose skew produced the OMN-18387 receipt failure.

    ``verify-sibling-converged`` checks out this repository and then runs
    ``scripts/ci/lab_pass_receipt.py probe-lane`` from that checkout. The
    invocation and the script are one contract, so the checkout that supplies
    the script must be pinned to the same commit as the YAML that spells the
    arguments.
    """
    path = WORKFLOWS_DIR / "runtime-rebuild-trigger-reusable.yml"
    assert path.exists(), f"{path} is missing"

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    job = (doc.get("jobs") or {}).get("verify-sibling-converged")
    assert job is not None, (
        "runtime-rebuild-trigger-reusable.yml no longer defines "
        "'verify-sibling-converged'; if the probe moved, move this assertion with it"
    )

    steps = job.get("steps") or []
    runs_probe_lane = any(
        "lab_pass_receipt.py probe-lane" in str(step.get("run") or "") for step in steps
    )
    assert runs_probe_lane, (
        "'verify-sibling-converged' no longer invokes lab_pass_receipt.py "
        "probe-lane -- this test is asserting a pairing that no longer exists"
    )

    checkouts = [
        step
        for step in steps
        if str(step.get("uses") or "").startswith("actions/checkout@")
        and str((step.get("with") or {}).get("repository") or "").strip()
        == SELF_REPOSITORY
    ]
    assert checkouts, (
        "'verify-sibling-converged' runs probe-lane but checks out no copy of "
        f"{SELF_REPOSITORY} to run it from"
    )

    for step in checkouts:
        ref = str((step.get("with") or {}).get("ref") or "").strip()
        assert _JOB_WORKFLOW_SHA.search(ref), (
            "the checkout that supplies lab_pass_receipt.py must resolve to this "
            f"workflow file's own commit, not `{ref or '<default branch>'}`. A "
            "floating script under a pinned YAML is how omnimarket run 35019423922 "
            "invoked probe-lane without the --projection-url that the dev script "
            "had just made required, producing a FAIL compose-dev receipt."
        )
