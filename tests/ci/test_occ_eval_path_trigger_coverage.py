# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression guard: occ-preflight EVAL-path caller workflows must be able to
re-validate after a stamp-only `Evidence-Source:` body edit (OMN-14241).

AMENDED BY OMN-16171 -- the invariant is now the OUTCOME, not one mechanism.
This module originally required the literal `edited` trigger on every eval-path
caller. That is still one valid way to satisfy it (and `hostile-reviewer.yml`,
at four jobs, still uses it), but on `ci.yml` it meant a brand-new run of the
whole ~48-job matrix per PR-body edit, and -- because that workflow's
concurrency group cancels in progress -- each edit also killed whatever matrix
was mid-flight. Measured on #2784, one unchanged head SHA 7993b115: waves at
13:28:50Z (42 workflows), 13:50:09Z (10) and 13:51:33Z (8), against an org-wide
backlog of 247 queued jobs. The edits are frequent because OCC's own bots
author the stamps with a minted App token, which does emit
`pull_request: edited` (GitHub's recursion guard suppresses only edits made by
the ambient GITHUB_TOKEN). `ci.yml` therefore returned to the default trigger
types and is now healed by `occ-preflight-heal.yml`, which re-runs the failed
jobs of the EXISTING run in place. Both mechanisms are accepted below; a
workflow with NEITHER is still the failure this module exists to catch.

Note on the alternative that was rejected: skipping the heavy jobs on an
edited-triggered run does not work, because a job skipped by an `if:` still
publishes a `skipped` check run, which branch protection counts as passing --
vector 2 of the required-check skip-vector guard, i.e. a body-edit-shaped way
to green a red required context.

Sibling gate to ``test_occ_born_path_trigger_coverage.py`` (OMN-14987), which
guards the born-path publishers (``call-occ-autobind.yml`` /
``call-occ-companion-effect.yml``) against missing ``reopened`` /
``ready_for_review``. This module guards the EVAL-path consumers -- the
workflows whose ``occ-preflight`` job actually gates merge by reading
``Evidence-Source:`` out of the PR body -- against missing ``edited``.

Live-root-caused 2026-08-09 (mergesweep-0809-infraunblock) on infra#2696 and
#2694: both PRs added their `Evidence-Source: OCC#<n>` line via a body-only
edit *after* opening. ``ci.yml`` (no ``types:`` key -> GitHub's implicit
``[opened, synchronize, reopened]``) and ``hostile-reviewer.yml``
(``types: [opened, synchronize, reopened]``) each declare a top-level job
literally named ``occ-preflight`` that calls the canonical reusable
``occ-preflight.yml``. Neither reruns on a body-only edit, so the
`occ-preflight / eligibility` check stays FAILURE from the pre-stamp run
forever -- even though the reusable workflow itself already live-fetches the
current PR body via `gh pr view` (a frozen-event-payload read is only a
`gh pr view` API-failure fallback, never the primary path). This is the same
"detection shelf structurally blind" failure mode as OMN-14987: a missing
trigger type produces silence (no new run), not a red check that would
prompt investigation -- `CI Summary` (the required umbrella) just fails
closed on the stale run and never recovers without a manual empty-commit
retrigger.

Contrast case proving the mechanism: `call-reject-skip.yml` already lists
`edited` and its nested `occ-preflight / eligibility` copy correctly
self-heals on the same body edit (live-observed on #2696: CI's copy stayed
FAILURE from 22:05:03Z; `call-reject-skip-token`'s copy re-ran and passed at
22:07:14Z after the 22:06:20Z body edit).

Design notes (mirrors OMN-14987's guard):

* Parses ``on.pull_request.types`` via YAML, not a line regex.
* Handles PyYAML 1.1's ``on:`` -> ``True`` resolution.
* ``test_extraction_catches_a_deliberately_underspecified_fixture`` proves
  RED against exists-but-wrong (the pre-fix shape), not merely green-by-
  absence (feedback_prove_red_against_exists_but_wrong).
* ``test_eval_path_workflow_set_is_current`` guards against a future third
  occ-preflight-caller workflow being added without updating the guarded set
  here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# `edited` is the one event this gate cares about: it is what fires when a
# PR's body is changed without a new commit (the Evidence-Source stamp
# case). The born-path gate (OMN-14987) already covers reopened/
# ready_for_review for the *minting* workflows; this gate is scoped to the
# narrower eval-path defect.
REQUIRED_EVAL_PATH_PR_EVENT_TYPES = frozenset({"edited"})

# Every workflow in THIS repo that declares a top-level job literally named
# `occ-preflight` calling the canonical reusable
# `OmniNode-ai/omnibase_core/.github/workflows/occ-preflight.yml`, i.e. every
# workflow whose own `occ-preflight / eligibility` check-run gates merge via
# `CI Summary` or an equivalent required umbrella. `call-reject-skip.yml` is
# deliberately excluded: it does not declare its own `occ-preflight` job (it
# reaches occ-preflight only transitively through the omniclaude-hosted
# reusable, producing a differently-named 3-segment check context), and it
# already lists `edited` today -- it is the working contrast case cited
# above, not a workflow this gate needs to constrain.
EVAL_PATH_WORKFLOWS: tuple[str, ...] = (
    "ci.yml",
    "hostile-reviewer.yml",
)

# OMN-16171. The second way to satisfy the self-heal requirement: instead of
# firing a whole new run on `edited`, re-run the failed jobs of the run that
# already exists. `ci.yml` uses this because its `edited` trigger cost a full
# ~48-job matrix per PR-body edit, and (via `cancel-in-progress`) killed the
# matrix that was mid-flight. Membership here is a claim that the healer really
# does cover the workflow -- the tests below check the healer's shape and its
# name/job pins against the live files rather than taking this tuple's word.
HEAL_WORKFLOW = "occ-preflight-heal.yml"
HEALER_COVERED_WORKFLOWS: tuple[str, ...] = ("ci.yml",)


def _pull_request_trigger_types(workflow_path: Path) -> set[str]:
    """Extract ``on.pull_request.types`` from a workflow file as a set."""
    loaded = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{workflow_path} did not parse to a mapping"
    # PyYAML 1.1's default resolver turns an unquoted `on:` key into the
    # boolean True -- check both so a formatting change can't blind this.
    on_block = loaded.get("on")
    if on_block is None:
        on_block = loaded.get(True)
    assert isinstance(on_block, dict), f"{workflow_path} has no on: mapping"
    pull_request_block = on_block.get("pull_request")
    assert isinstance(pull_request_block, dict), (
        f"{workflow_path} on.pull_request is not a mapping -- an unqualified "
        f"`pull_request` trigger (or a bare list of types) is a different "
        f"authoring style this gate does not (yet) understand; update the "
        f"gate deliberately, do not let it silently pass."
    )
    types = pull_request_block.get("types")
    if types is None:
        # Omitting `types:` entirely defaults to GitHub's implicit
        # [opened, synchronize, reopened] -- still missing `edited`. Treat
        # absence as the empty set rather than raising, since (unlike the
        # born-path gate) an implicit-default workflow is a real, common,
        # currently-broken shape in this repo (ci.yml today) and the
        # assertion below must catch it, not skip it.
        return set()
    assert isinstance(types, list) and types, (
        f"{workflow_path} on.pull_request.types is present but empty/not a "
        f"list -- fix the YAML shape."
    )
    return set(types)


@pytest.mark.unit
def test_extraction_catches_a_deliberately_underspecified_fixture(
    tmp_path: Path,
) -> None:
    """Prove the check is RED against exists-but-wrong (the pre-fix trigger
    shape used by hostile-reviewer.yml today), not just absent
    (feedback_prove_red_against_exists_but_wrong)."""
    fixture = tmp_path / "fixture.yml"
    fixture.write_text(
        "on:\n  pull_request:\n    types: [opened, synchronize, reopened]\n",
        encoding="utf-8",
    )
    types = _pull_request_trigger_types(fixture)
    missing = REQUIRED_EVAL_PATH_PR_EVENT_TYPES - types
    assert missing == {"edited"}


@pytest.mark.unit
def test_extraction_handles_implicit_default_types_fixture(tmp_path: Path) -> None:
    """A workflow with no `types:` key at all relies on GitHub's implicit
    default ([opened, synchronize, reopened]) -- still missing `edited`.
    This is ci.yml's actual pre-fix shape; the gate must fail loud on it, not
    treat "no types: key" as "nothing to check"."""
    fixture = tmp_path / "fixture.yml"
    fixture.write_text(
        "on:\n  pull_request:\n    branches: [main, dev]\n", encoding="utf-8"
    )
    types = _pull_request_trigger_types(fixture)
    assert types == set()
    missing = REQUIRED_EVAL_PATH_PR_EVENT_TYPES - types
    assert missing == {"edited"}


@pytest.mark.unit
@pytest.mark.parametrize("workflow_name", EVAL_PATH_WORKFLOWS)
def test_eval_path_workflow_has_a_self_heal_path(workflow_name: str) -> None:
    """OMN-14241, as amended by OMN-16171.

    The invariant is the OUTCOME, not the mechanism: a PR body edit that adds
    `Evidence-Source: OCC#<n>` after opening must be able to clear the stale
    pre-stamp `occ-preflight / eligibility` FAILURE. Otherwise that check-run
    stays the latest status forever and `CI Summary` fails closed on it (live
    cases: infra#2696, #2694).

    Two mechanisms satisfy it, and this gate accepts either:

    * the workflow lists `edited` itself, so the edit produces a fresh run; or
    * the workflow is healed in place by `occ-preflight-heal.yml`, which fires
      on `edited` and re-runs the FAILED JOBS of the existing run, keeping the
      run id -- which is what `CI Summary` needs, since it polls its own run's
      job list rather than the SHA's check-runs.

    `ci.yml` moved to the second mechanism because the first cost a full ~48-job
    matrix per body edit (and, via `cancel-in-progress`, killed whatever matrix
    was mid-flight). `hostile-reviewer.yml` stays on the first: it is four jobs,
    so the trigger is cheaper than the indirection.
    """
    workflow_path = WORKFLOWS_DIR / workflow_name
    assert workflow_path.is_file(), f"expected workflow file at {workflow_path}"
    types = _pull_request_trigger_types(workflow_path)
    has_own_trigger = not (REQUIRED_EVAL_PATH_PR_EVENT_TYPES - types)
    # Membership in HEALER_COVERED_WORKFLOWS is a claim, not evidence. Deleting
    # occ-preflight-heal.yml while leaving this tuple alone must not leave a
    # workflow with no self-heal path certified as having one -- that is exactly
    # the vacuous-gate shape (a guard that passes by reading its own
    # configuration rather than the world) this module must not become.
    healer_covered = (
        workflow_name in HEALER_COVERED_WORKFLOWS
        and (WORKFLOWS_DIR / HEAL_WORKFLOW).is_file()
    )
    assert has_own_trigger or healer_covered, (
        f"{workflow_name} has NO self-heal path: it does not list `edited` "
        f"(types={sorted(types)}) and is not in HEALER_COVERED_WORKFLOWS. A PR "
        f"body edit adding the Evidence-Source stamp will never clear this "
        f"workflow's stale pre-stamp occ-preflight FAILURE, blocking merge "
        f"indefinitely without a manual empty-commit retrigger. Restore the "
        f"`edited` trigger, or cover it in {HEAL_WORKFLOW}."
    )


@pytest.mark.unit
def test_healer_fires_on_edited_and_reruns_only_failed_jobs() -> None:
    """The healer is the mechanism `ci.yml` now depends on; pin its shape.

    Three properties matter. It must fire on `edited` (nothing else clears a
    body-only stamp). It must re-run the failed jobs of the EXISTING run rather
    than dispatch a new one -- a new run would recreate the full-matrix cost the
    healer exists to avoid, and would not update the run `CI Summary` polls. And
    it must gate on occ-preflight specifically, so an unrelated body edit does
    not re-queue genuine test failures.
    """
    heal_path = WORKFLOWS_DIR / HEAL_WORKFLOW
    assert heal_path.is_file(), f"expected the healer at {heal_path}"
    assert _pull_request_trigger_types(heal_path) == {"edited"}, (
        f"{HEAL_WORKFLOW} must fire on exactly `edited` -- broadening it "
        f"reintroduces per-push cost, narrowing it disables the heal"
    )
    body = heal_path.read_text(encoding="utf-8")
    assert "--failed" in body and "gh run rerun" in body, (
        f"{HEAL_WORKFLOW} must heal via `gh run rerun ... --failed` (same run "
        f"id, failed jobs only). A fresh dispatch reintroduces the full-matrix "
        f"cost and leaves the run CI Summary polls untouched."
    )
    assert "PREFLIGHT_JOB_PREFIX: occ-preflight" in body, (
        f"{HEAL_WORKFLOW} must gate the rerun on a failed occ-preflight job"
    )


@pytest.mark.unit
def test_healer_targets_the_workflow_name_ci_actually_publishes() -> None:
    """The healer finds its run by workflow NAME, so a rename silently disables it.

    This is the same class of failure the module guards elsewhere: the break is
    silence (no rerun) rather than a red check. Pin the two strings against the
    live files instead of trusting the comment next to them.
    """
    heal_body = (WORKFLOWS_DIR / HEAL_WORKFLOW).read_text(encoding="utf-8")
    ci_data = yaml.safe_load((WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8"))

    assert f"CI_WORKFLOW_NAME: {ci_data['name']}" in heal_body, (
        f"{HEAL_WORKFLOW} looks for a run named CI_WORKFLOW_NAME, which must "
        f"equal ci.yml's `name:` ({ci_data['name']!r}); it does not"
    )
    jobs = ci_data.get("jobs") or {}
    assert "occ-preflight" in jobs, (
        "ci.yml no longer declares a job id `occ-preflight`, so the healer's "
        "PREFLIGHT_JOB_PREFIX can never match a failed job"
    )


@pytest.mark.unit
def test_eval_path_workflow_set_is_current() -> None:
    """Guard against a future third occ-preflight-caller workflow (a top-
    level job literally named `occ-preflight`) being added without updating
    EVAL_PATH_WORKFLOWS above -- otherwise the new file is silently
    unchecked by this module, repeating the OMN-14241 failure class one
    layer up."""
    discovered: set[str] = set()
    for path in WORKFLOWS_DIR.glob("*.yml"):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        jobs = loaded.get("jobs")
        if not isinstance(jobs, dict):
            continue
        job = jobs.get("occ-preflight")
        if not isinstance(job, dict):
            continue
        uses = job.get("uses", "")
        if isinstance(uses, str) and "occ-preflight.yml" in uses:
            discovered.add(path.name)
    assert discovered == set(EVAL_PATH_WORKFLOWS), (
        f"discovered occ-preflight-caller workflows {sorted(discovered)} do "
        f"not match the guarded set {sorted(EVAL_PATH_WORKFLOWS)} -- update "
        f"EVAL_PATH_WORKFLOWS (and re-verify edited-trigger coverage) for "
        f"the new/removed file before this gate can pass"
    )
