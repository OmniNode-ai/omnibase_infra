# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Refuse a deploy ref that would move the lane's clone backwards (OMN-18122).

WHAT THIS COST, MEASURED
------------------------

Between 2026-09-10T00:48:53Z and 03:31:48Z five rebuild commands carrying
``git_ref: "origin/main"`` reached the .201 dev agent -- jobs ``d62ec83e``,
``68295ceb``, ``1316f524``, ``77343308``, ``ced0ac22`` -- and each ran
``git reset --hard origin/main`` on the SHARED deploy-source clone. At that
moment ``origin/main`` was ``276d69383`` and ``origin/dev`` was ``bcb3e6460``:
77 commits apart, with ``main`` carrying none of its own. The clone landed
behind the commit that introduced ``scripts/preflight_required_compose_env.py``,
so every CI-triggered dev rebuild died at compose generation until the clone was
reconciled by hand.

``deploy_agent.tracking_ref`` already carries the operator ruling that produced
this module's sibling fix: *the deploy agent tracks the lane's deploy branch,
never* ``main``. OMN-16442 applied that to the agent's own self-update and to the
``ModelRebuildRequested.git_ref`` DEFAULT. It did not apply it to a request that
names another branch EXPLICITLY, which is the case that actually fired -- the
upstream producer (OMN-18121) put the literal there itself.

WHY THE RULE IS SHAPED THIS WAY
-------------------------------

Three candidate rules were weighed against the measured facts.

*Refuse the literal* ``main``. Rejected. It reintroduces exactly the hardcoded
branch literal OMN-16442 removed from this package, and it silently passes a repo
whose release-synced branch is named something else. Denying a literal is a
smaller sin than defaulting to one, but it is the same sin.

*Require the ref to be reachable from the tracking branch.* Rejected on
evidence, not on taste: ``git merge-base --is-ancestor origin/main origin/dev``
exited 0 at the time of the incident. ``main`` on these repos is release-synced,
so its tip is normally an ancestor of ``dev``. This rule does not catch the case
it was proposed for.

*Refuse a branch reference that is STRICTLY BEHIND the tracking branch* -- no
commits of its own, at least one commit missing. Implemented. It names no
branch, it encodes the actual harm (the deploy clone moves backwards off its
lane's own lineage), and it leaves a feature-branch deploy working because a
feature branch carries commits the tracking branch does not.

WHAT IS DELIBERATELY OUT OF SCOPE
---------------------------------

A bare commit SHA. A SHA is an unambiguous pin the caller chose and typed, and
the GitHub Actions publisher pins one on every dev deploy -- ``gha-redeploy``
records on the start topic carry a 40-hex ref, never a branch. The defect class
here is a branch ALIAS that silently resolves somewhere stale; a SHA cannot
drift between the moment it is published and the moment it is reset to.

There is no override flag and no environment escape. An override would make this
fence advisory, and an advisory fence is what the five jobs above already had.
"""

from __future__ import annotations

from dataclasses import dataclass

from deploy_agent.tracking_ref import ENV_TRACKING_REF


class StaleBranchRefError(RuntimeError):
    """Raised when a deploy ref names a branch strictly behind the lane's own."""


@dataclass(frozen=True)
class ModelRefLineageFacts:
    """What git says about the requested ref relative to the lane's tracking ref.

    Kept separate from the git calls that gather it so the decision is a pure
    function: the rule above is the part worth pinning in tests, and it should
    not need a repository on disk to exercise.
    """

    requested_ref: str
    tracking_ref: str
    is_branch_reference: bool
    commits_ahead_of_tracking: int
    commits_behind_tracking: int


def assert_ref_not_stale_branch(facts: ModelRefLineageFacts) -> None:
    """Raise ``StaleBranchRefError`` for a branch strictly behind the tracking ref.

    Strictly behind means: it is a branch reference, it is not the tracking
    branch itself, it carries no commit the tracking branch lacks, and the
    tracking branch carries at least one commit it lacks. Every other shape --
    a SHA, the tracking branch, a branch with commits of its own, a branch level
    with the tracking branch -- passes untouched.
    """
    if not facts.is_branch_reference:
        return
    if facts.requested_ref == facts.tracking_ref:
        return
    if facts.commits_ahead_of_tracking > 0:
        return
    if facts.commits_behind_tracking == 0:
        return

    raise StaleBranchRefError(
        f"deploy ref {facts.requested_ref!r} names a branch that is strictly "
        f"behind this lane's tracking branch {facts.tracking_ref!r}: it carries "
        f"no commit of its own and is {facts.commits_behind_tracking} commits "
        "behind. Resetting the deploy clone to it would move the lane backwards "
        "off the code it exists to run. This lane declares its branch in "
        f"{ENV_TRACKING_REF}; publish an explicit commit SHA, or the tracking "
        "branch itself, if this deploy is intended."
    )
