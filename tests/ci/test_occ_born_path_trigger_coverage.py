# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression guard: OCC born-path trigger workflows must listen for the full
set of ``pull_request`` events that can make a PR mint-eligible (OMN-14987).

OMN-14987 (live-observed, PROVE canary session 2026-07-23): both
``call-occ-autobind.yml`` and ``call-occ-companion-effect.yml`` declared
``on.pull_request.types: [opened, synchronize]`` only. A PR opened as a draft
is correctly suppressed by the downstream F-17 draft guard on ``opened``; when
it is later marked ready-for-review WITHOUT an intervening push, GitHub emits
a distinct ``ready_for_review`` event that neither workflow listened for. No
run fires at all -- not a failing run, no run whatsoever -- so the PR is
stranded unbound forever with no red signal anywhere (live case: infra#2407,
plus 6 further PRs in the same session). ``reopened`` (closed PR reopened
without a new commit) is the same failure class via a different event.

This is the "detection shelf structurally blind" failure mode: a missing
trigger type produces silence, not a red check, so nothing short of a
structural assertion over the trigger declaration itself can catch it. This
module is that assertion.

Design notes:

* Parses the ``on.pull_request.types`` list via YAML (not a line regex), so a
  reformatted or reordered trigger block does not evade the check.
* PyYAML 1.1 resolves an unquoted ``on:`` key to the boolean ``True`` -- the
  parser here checks both ``"on"`` and ``True`` so a change to a workflow
  file's YAML formatting cannot silently blind the extractor.
* ``test_extraction_catches_a_deliberately_underspecified_fixture`` proves the
  assertion is RED against an exists-but-wrong fixture (the old
  ``[opened, synchronize]`` trigger), not merely green-by-absence
  (feedback_prove_red_against_exists_but_wrong).
* ``test_born_path_workflow_set_is_current`` guards against a future third
  born-path publisher workflow being added (fan-out beyond omnibase_infra,
  OMN-14811/OMN-14941 lineage) without updating the guarded set here --
  silent coverage drift is exactly the OMN-14987 failure class recurring one
  layer up.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Every PR event that can transition a PR into mint-eligible state (open,
# non-draft, ticketed). Missing any of these from a born-path publisher's
# trigger is the OMN-14987 failure class: the publisher simply never runs,
# and because it never runs there is no failing check to notice -- silence,
# not red.
REQUIRED_BORN_PATH_PR_EVENT_TYPES = frozenset(
    {
        "opened",  # PR created (already non-draft)
        "synchronize",  # new commit pushed (forces a re-check regardless)
        "reopened",  # closed PR reopened without a new commit
        "ready_for_review",  # draft -> ready transition without a new commit
    }
)

# The full set of OCC born-path caller workflows in THIS repo (canary before
# fan-out to core/omniclaude per the header comments in both files). If a
# third caller is added, `test_born_path_workflow_set_is_current` below fails
# until this tuple is updated -- new coverage must be a deliberate edit, not
# an accident of glob discovery.
BORN_PATH_WORKFLOWS: tuple[str, ...] = (
    "call-occ-autobind.yml",
    "call-occ-companion-effect.yml",
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
    assert isinstance(types, list) and types, (
        f"{workflow_path} on.pull_request.types is missing or empty -- "
        f"omitting `types:` entirely defaults to GitHub's [opened, "
        f"synchronize, reopened] (still missing ready_for_review), which is "
        f"an implicit, easy-to-miss-in-review way to reintroduce this gap."
    )
    return set(types)


@pytest.mark.unit
def test_extraction_catches_a_deliberately_underspecified_fixture(
    tmp_path: Path,
) -> None:
    """Prove the check is RED against exists-but-wrong (the pre-fix trigger
    shape), not just absent (feedback_prove_red_against_exists_but_wrong)."""
    fixture = tmp_path / "fixture.yml"
    fixture.write_text(
        "on:\n  pull_request:\n    types: [opened, synchronize]\n",
        encoding="utf-8",
    )
    types = _pull_request_trigger_types(fixture)
    missing = REQUIRED_BORN_PATH_PR_EVENT_TYPES - types
    assert missing == {"reopened", "ready_for_review"}


@pytest.mark.unit
def test_extraction_handles_implicit_default_types_fixture(tmp_path: Path) -> None:
    """A workflow with no `types:` key at all relies on GitHub's implicit
    default ([opened, synchronize, reopened]) -- still missing
    ready_for_review. The gate must fail loud on this shape too, not treat
    "no types: key" as "nothing to check"."""
    fixture = tmp_path / "fixture.yml"
    fixture.write_text("on:\n  pull_request: {}\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="missing or empty"):
        _pull_request_trigger_types(fixture)


@pytest.mark.unit
@pytest.mark.parametrize("workflow_name", BORN_PATH_WORKFLOWS)
def test_born_path_workflow_listens_for_full_mint_eligible_event_set(
    workflow_name: str,
) -> None:
    """OMN-14987: opened/synchronize alone silently strand any PR that
    transitions draft->ready_for_review, or closed->reopened, without an
    intervening push -- no run fires, nothing goes red, occ-preflight sits
    red forever with no `Evidence-Source:`. Asserts both born-path publisher
    workflows in this repo listen for the full mint-eligible event set."""
    workflow_path = WORKFLOWS_DIR / workflow_name
    assert workflow_path.is_file(), f"expected workflow file at {workflow_path}"
    types = _pull_request_trigger_types(workflow_path)
    missing = REQUIRED_BORN_PATH_PR_EVENT_TYPES - types
    assert not missing, (
        f"{workflow_name} on.pull_request.types is missing {sorted(missing)} -- "
        f"a PR that transitions via a missing event type will NEVER trigger "
        f"the born-path publisher (OMN-14987 failure class: nothing runs, "
        f"nothing goes red, the PR is silently stranded unbound)."
    )


@pytest.mark.unit
def test_born_path_workflow_set_is_current() -> None:
    """Guard against the optional-input-silent-skip trap one layer up: if a
    third born-path publisher workflow is added later (fan-out beyond
    omnibase_infra, OMN-14811/OMN-14941 lineage) without updating
    BORN_PATH_WORKFLOWS above, the new file would be silently unchecked by
    every test in this module. Forces that update to be deliberate."""
    discovered = {
        p.name for p in WORKFLOWS_DIR.glob("call-occ-*.yml") if "reusable" not in p.name
    }
    assert discovered == set(BORN_PATH_WORKFLOWS), (
        f"discovered born-path caller workflows {sorted(discovered)} do not "
        f"match the guarded set {sorted(BORN_PATH_WORKFLOWS)} -- update "
        f"BORN_PATH_WORKFLOWS (and re-verify REQUIRED_BORN_PATH_PR_EVENT_TYPES "
        f"coverage) for the new/removed file before this gate can pass"
    )
