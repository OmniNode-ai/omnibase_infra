# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106 Item 0 — one applying Evidence Autoclose Sweep at a time.

The sweep runs every 30 minutes AND is dispatchable by hand, and until this
change it declared no GitHub Actions ``concurrency`` key at all. Two runs of a
writer whose whole job is to move tickets to Done could therefore overlap: a
cron tick and an operator dispatch, or two cron ticks when one runs long
against a ``dod_verify`` budget of ~15s per candidate. Overlapping applying
runs each read the flip budget, the ticket state, the comment history and the
prior-revert history INDEPENDENTLY, so the per-run bound (``max_flips_per_run``
= 5) stops bounding the fleet, and the OMN-16808 read-before-write comment
dedup races itself.

What this file pins:

1. **The key is at WORKFLOW scope**, not job scope, and its group is a literal
   constant. A group rendered from an expression (``${{ github.event_name }}``,
   ``${{ github.ref }}``) puts a cron tick and a dispatch in DIFFERENT groups,
   which is the same as having no key for the overlap that actually matters.

2. **``cancel-in-progress`` is the literal boolean false.** Cancelling the
   in-flight member of this group is not a safe default for a writer: the run
   being cancelled may have already written a Done and not yet read it back,
   which is the exact shape that produced ERROR_READBACK_UNCONFIRMED and the
   OMN-16106 rollback path. Waiting is correct; pre-empting is not.

3. **No job in the file re-declares its own concurrency**, which would split
   the workflow-scope group back apart.

What this file deliberately does NOT claim: that queued work is never lost.
GitHub keeps ONE running member plus at most ONE pending member per group, and
a later queued run REPLACES the earlier pending one — the replaced run is
cancelled, not deferred. A cancelled pending run is not a completed run, so a
pilot has to reconcile every selected ticket to a terminal outcome and re-offer
whatever a replacement discarded. That is a procedural obligation recorded in
the plan; no YAML key can discharge it, and asserting otherwise here would be
the kind of string that reads as a fact and is not one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"

EXPECTED_GROUP = "evidence-autoclose-apply"


def _load_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(SWEEP_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{SWEEP_WORKFLOW} did not parse to a mapping"
    return loaded


def _triggers() -> dict[str, Any]:
    # `on:` is parsed by YAML 1.1 as the boolean True, not the string "on".
    workflow: dict[Any, Any] = _load_workflow()
    for key in (True, "on"):
        triggers = workflow.get(key)
        if isinstance(triggers, dict):
            return triggers
    raise AssertionError(f"{SWEEP_WORKFLOW} declares no trigger block")


def test_the_workflow_declares_a_workflow_scope_concurrency_group() -> None:
    """Workflow scope, so every trigger of this file shares one group."""
    workflow = _load_workflow()
    concurrency = workflow.get("concurrency")
    assert isinstance(concurrency, dict), (
        "evidence-autoclose-sweep.yml declares no workflow-scope `concurrency` "
        "mapping. Without one, a cron tick and a dispatch — or two cron ticks "
        "when one runs long — apply concurrently, and `max_flips_per_run` "
        "stops bounding the fleet because each run counts its own budget."
    )
    assert concurrency.get("group") == EXPECTED_GROUP, (
        f"the concurrency group must be the literal {EXPECTED_GROUP!r}; got "
        f"{concurrency.get('group')!r}"
    )


def test_the_group_is_a_literal_and_not_rendered_from_the_event() -> None:
    """A templated group is the same as no group for the overlap that matters.

    ``group: autoclose-${{ github.event_name }}`` reads like serialization and
    puts the scheduled writer and the dispatched writer in two different
    groups, which is precisely the pair this key exists to serialize.
    """
    group = _load_workflow()["concurrency"]["group"]
    assert isinstance(group, str)
    assert "${{" not in group, (
        f"the concurrency group {group!r} is rendered from an expression, so "
        "cron and workflow_dispatch land in different groups and never wait "
        "for each other"
    )


def test_cancel_in_progress_is_the_literal_boolean_false() -> None:
    """Never pre-empt a writer mid-flight.

    A cancelled applying run can have written a Done whose readback never ran,
    leaving the board changed with no receipt saying so — the shape that
    produced ERROR_READBACK_UNCONFIRMED and the OMN-16106 rollback path. The
    literal `false` also has to be a BOOLEAN, not the string "false": YAML
    parses an unquoted `false` as a boolean and a quoted `'false'` as a truthy
    string, and GitHub evaluates the string form as false only by coincidence
    of its own coercion — pinning the type keeps the diff honest.
    """
    concurrency = _load_workflow()["concurrency"]
    assert "cancel-in-progress" in concurrency, (
        "`cancel-in-progress` must be stated explicitly. Its GitHub default is "
        "already false, but an omitted key cannot be reviewed and cannot fail "
        "this test when somebody flips it."
    )
    value = concurrency["cancel-in-progress"]
    assert value is False, (
        f"cancel-in-progress must be the literal boolean false; got {value!r}"
    )


def test_every_trigger_of_this_workflow_is_covered_by_the_one_group() -> None:
    """Both the cron and the manual reference sit under the same key.

    Workflow-scope concurrency covers every trigger by construction, so the
    assertion that carries weight is that the triggers this workflow declares
    are the ones the group was reasoned about — a third trigger added later
    (``repository_dispatch``, ``workflow_call``) inherits the group silently,
    and this test is where somebody notices that it should.
    """
    triggers = _triggers()
    assert set(triggers) == {"schedule", "workflow_dispatch"}, (
        "the sweep's trigger set changed. The workflow-scope concurrency group "
        "covers every trigger automatically, but a new trigger is a new class "
        f"of writer: re-read the group's reasoning. Found {sorted(triggers)}"
    )
    schedule = triggers["schedule"]
    assert isinstance(schedule, list) and schedule, "the cron trigger vanished"


def test_no_job_redeclares_its_own_concurrency() -> None:
    """A job-scope key would split the workflow-scope group back apart."""
    jobs = _load_workflow()["jobs"]
    assert isinstance(jobs, dict) and jobs
    offenders = [
        job_id
        for job_id, job in jobs.items()
        if isinstance(job, dict) and "concurrency" in job
    ]
    assert offenders == [], (
        f"job(s) {offenders} declare their own `concurrency`, which overrides "
        "the workflow-scope group and lets two applying runs proceed in "
        "parallel again"
    )
