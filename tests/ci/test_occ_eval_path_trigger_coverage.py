# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression guard: occ-preflight EVAL-path caller workflows must listen for
the `edited` PR event, or a stamp-only `Evidence-Source:` body edit never
re-validates (OMN-14241).

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
def test_eval_path_workflow_listens_for_edited(workflow_name: str) -> None:
    """OMN-14241: a PR body edit that adds `Evidence-Source: OCC#<n>` after
    opening must re-trigger the occ-preflight eligibility check. Missing
    `edited` here means the check-run from the last commit-triggered run --
    which legitimately failed before the stamp existed -- sits as the
    latest, permanently-stale status for `occ-preflight / eligibility`, and
    `CI Summary` fails closed on it forever (live case: infra#2696, #2694).
    """
    workflow_path = WORKFLOWS_DIR / workflow_name
    assert workflow_path.is_file(), f"expected workflow file at {workflow_path}"
    types = _pull_request_trigger_types(workflow_path)
    missing = REQUIRED_EVAL_PATH_PR_EVENT_TYPES - types
    assert not missing, (
        f"{workflow_name} on.pull_request.types is missing {sorted(missing)} -- "
        f"a PR body edit (e.g. adding the Evidence-Source stamp) will NEVER "
        f"retrigger this workflow's occ-preflight job (OMN-14241 failure "
        f"class: the stale pre-stamp FAILURE is never superseded, blocking "
        f"merge indefinitely without a manual empty-commit retrigger)."
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
