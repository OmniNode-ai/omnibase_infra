# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A NON_RUNTIME merge still answers for queued merges; only the wait is skipped.

OMN-18976, the residual gap. ``verify-lane-converged`` used to run only when the
trigger PUBLISHED a rebuild. A NON_RUNTIME merge publishes nothing, so its run
skipped the whole job, including the only step that answers for merges queued
behind an earlier one. On 2026-09-26 two NON_RUNTIME merges in a row (#4164,
#4175) skipped it while another repository's push rebuilt the lane past the
queued ``84a1e7c3ec`` (#4160), which was orphaned for good.

These pin the workflow graph, not prose: the job runs for every dev merge; with
nothing published the guard runs ``--observe`` with no wait; an observation is
uploaded under a name no gate reads and is never announced as a lab-pass
verdict; and the re-emission job re-keys from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"


def _jobs() -> dict[str, Any]:
    jobs: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    return jobs


def _steps(job: str) -> dict[str, dict[str, Any]]:
    return {
        s["name"]: s for s in _jobs()[job]["steps"] if isinstance(s.get("name"), str)
    }


def _flat(value: object) -> str:
    return " ".join(str(value).split())


def test_the_verify_job_runs_for_every_dev_merge_not_only_published_ones() -> None:
    condition = _flat(_jobs()["verify-lane-converged"]["if"])
    assert "runtime_lane == 'dev'" in condition
    assert "published" not in condition, (
        "gating the job on a published rebuild skips the re-emission answer on "
        "every NON_RUNTIME merge, which is how 84a1e7c3ec was orphaned"
    )


def test_an_unpublished_merge_observes_with_no_wait() -> None:
    step = _steps("verify-lane-converged")[
        "Wait for the dev lane to report a revision containing the merge SHA"
    ]
    run = step["run"]
    observe_branch = run.split('if [[ "$PUBLISHED" != "true" ]]; then', 1)
    assert len(observe_branch) == 2, "the unpublished branch must exist"
    branch = observe_branch[1].split("fi\n", 1)[0]
    assert "--observe" in branch
    assert "--expect-revision" not in branch, "an observation must not wait"
    # Fail-closed defaults are written BEFORE the guard runs.
    assert branch.index("emit_receipt=false") < branch.index("--observe")
    assert branch.index("probe_lane=false") < branch.index("--observe")
    assert step["env"]["PUBLISHED"] == "${{ needs.trigger-rebuild.outputs.published }}"


def test_both_paths_read_the_durable_floor() -> None:
    run = _steps("verify-lane-converged")[
        "Wait for the dev lane to report a revision containing the merge SHA"
    ]["run"]
    assert run.count("--runtime-path-validator") == 2
    assert run.count("--clone .") == 2


def test_the_verify_job_carries_history_and_the_canonical_classifier() -> None:
    steps = list(_jobs()["verify-lane-converged"]["steps"])
    checkout = steps[0]
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.ref }}"
    assert int(checkout["with"]["fetch-depth"]) >= 50
    classifier = _steps("verify-lane-converged")[
        "Fetch canonical deploy-path classifier"
    ]
    assert classifier["with"]["repository"] == "OmniNode-ai/omniclaude"
    assert (
        ".github/actions/deploy-gate/validate_pr_deploy_required.py"
        in classifier["with"]["sparse-checkout"]
    )


@pytest.mark.parametrize(
    "name",
    [
        "Derive this lane's declared writer consumer groups",
        "Take the earlier consumer-lag sample",
        "Probe the dev lane",
    ],
)
def test_an_observation_with_nothing_owed_probes_nothing(name: str) -> None:
    condition = _flat(_steps("verify-lane-converged")[name]["if"])
    assert condition == "always() && steps.converge.outputs.probe_lane != 'false'"


def test_an_observation_is_uploaded_under_a_name_no_gate_reads() -> None:
    steps = _steps("verify-lane-converged")
    receipt = steps["Upload the compose-dev lab-pass receipt"]
    observation = steps["Upload the compose-dev lab-pass observation"]
    assert "published == 'true'" in receipt["if"]
    assert "published != 'true'" in observation["if"]
    assert receipt["with"]["name"].startswith("lab-pass-receipt-")
    assert observation["with"]["name"].startswith("lab-pass-observation-"), (
        "an observation keyed by a non-runtime merge must never land under the "
        "name the delivery gate reads for that merge"
    )
    for step in (receipt, observation):
        assert "steps.converge.outputs.emit_receipt != 'false'" in step["if"]
        assert step["with"]["if-no-files-found"] == "error"


def test_the_reemission_job_rekeys_from_the_observation() -> None:
    name = _steps("reemit-queued-receipts")["Download the converged receipt"]["with"][
        "name"
    ]
    assert "lab-pass-observation-compose-dev-{0}" in name
    assert "lab-pass-receipt-compose-dev-{0}" in name


def test_an_observation_is_never_announced_as_a_lab_pass_verdict() -> None:
    step = _steps("verify-lane-converged")[
        "Publish the compose-dev lab-pass verdict to the bus"
    ]
    assert step["if"] == "always()"
    run = step["run"]
    assert run.index('if [[ "$PUBLISHED" != "true" ]]') < run.index(
        "publish_lab_fact_event.py"
    )


def test_the_delivery_reread_runs_when_an_observation_answered() -> None:
    condition = _flat(_jobs()["rerun-refused-deliveries"]["if"])
    assert "needs.trigger-rebuild.outputs.published == 'true' ||" in condition
    assert "reemit_candidates != '[]'" in condition


def test_the_lab_overlay_job_still_needs_a_published_rebuild() -> None:
    # An observation rebuilt nothing, so there is no overlay apply to verify.
    condition = _flat(_jobs()["verify-lab-overlay-converged"]["if"])
    assert "published == 'true'" in condition
