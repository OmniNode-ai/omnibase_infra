# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the newest queued rebuild for a lane, and record the ones it replaced (OMN-18143).

WHAT THIS COSTS TODAY, MEASURED
-------------------------------
The agent pins compose content per rebuild command to the sha that triggered
it and drains its control topic in commit order, at a mean service time of
1799.8 s per job (the rolling mean ``queue_depth.mean_service_time`` served on
``/queue`` the night this was written). At 2026-09-19T02:58Z job ``52e430b2``
was staging ``f8475f9b``, sixteen commits behind ``origin/dev``, and the job
before it had staged an older sha still. With roughly forty runtime-affecting
merges in one night, the lane spends hours rebuilding shas that a later commit
has already replaced, recreating the runtime family once per superseded sha,
while the actual dev head waits at the back of the queue.

Nothing surfaced that a declared change was pending. The queue endpoint
OMN-18144 landed reports the DEPTH; it deliberately observes and never acts
(``queue_depth``'s own docstring: "Nothing here cancels, reorders, coalesces or
times out a command"), and it names this ticket as the owner of the acting
half. This module is that half.

WHAT IS COALESCED, AND WHAT IS REFUSED
---------------------------------------
A PREFIX of the queue, and only a prefix. The scan starts at the command the
agent is about to accept and walks forward while each next command is
foldable into the same group; the FIRST command that is not ends the group and
stays queued, with everything behind it. Ordering is preserved exactly: the
agent never runs a later command ahead of an earlier one it refused to fold.

Every one of the conditions below is a REFUSAL that ends the group, and each
has its own reason so the journal says which one fired rather than reporting a
bare "not coalesced":

``DIFFERENT_LANE``
    Commands for different lanes are never coalesced. They mutate different
    compose projects; folding them would drop a deploy of one lane entirely.

``NOT_FULL_SCOPE``
    Only ``scope=full`` folds into ``scope=full``. The scope decides WHICH
    services are recreated, so folding a ``core`` command into a ``full`` one
    -- or the reverse -- silently drops the services only one of them names.
    Deliberately narrower than it has to be: two ``core`` commands could in
    principle fold, but a ``core`` rebuild is not the 45-minute job this
    module exists to stop spending, and a rule that only ever folds like into
    like is one a reader can check.

``SERVICES_DIFFER`` / ``BUILD_SOURCE_DIFFERS``
    Same argument. A command naming services, or built from a different
    source, is not the same work as the one ahead of it.

``PINNED_IMAGE``
    A command carrying ``image_ref`` or ``image_digest`` is a promotion of one
    exact artifact, not a rebuild of a branch tip, and it is never folded. This
    is also what makes a ``prod`` command structurally un-coalescable without a
    lane check of its own: ``ModelRebuildRequested`` refuses a prod request
    that carries no digest, so every prod command trips this refusal.

``REF_NOT_A_SHA``
    A branch alias resolves at BUILD time to whatever the branch is then, so
    two alias commands are not comparable by ancestry and "newer" is not a
    property either of them carries. Only 40-hex pins fold.

``ANCESTRY_UNPROVEN`` / ``NOT_A_DESCENDANT``
    The load-bearing one. Coalescing is only safe when the sha that RUNS
    contains the sha it replaces; otherwise the replaced change is simply
    dropped and nothing says so. The check is
    ``git merge-base --is-ancestor <earlier> <later>`` on consecutive members,
    which gives containment across the whole group by transitivity. An
    unresolvable comparison -- an object the clone does not have, a git
    failure, a timeout -- is ``ANCESTRY_UNPROVEN`` and ends the group, so every
    failure of this module falls back to the behaviour it replaces: run every
    command, in order, exactly as before.

WHY THERE IS NO KILL SWITCH
----------------------------
Every failure mode above degrades to the pre-change behaviour by construction,
so an ``off`` setting would buy nothing an incident cannot already get by
letting the refusals fire -- and an opt-out left set is how a lane silently
loses an optimisation nobody notices is gone (rule 5).

WHAT A SUPERSEDED COMMAND GETS
-------------------------------
Not silence, and not a failure. Each superseded command gets a durable job
record with status ``superseded``, naming the sha and the correlation id that
ran in its place, and a terminal event on the rejection topic carrying the
same two fields. OMN-18143 AC6 asks for exactly that: a terminal event that
says "superseded", distinguishably from a timeout and from a rollback.
"""

from __future__ import annotations

import logging
import re
import subprocess  # fixed argv, no shell, trusted git binary
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deploy_agent.events import ModelRebuildRequested, Scope

logger = logging.getLogger(__name__)

#: A full 40-character lowercase commit sha. The same shape
#: ``scripts/ci/lab_pass_receipt.py`` gates on, and for the same reason: an
#: abbreviated ref cannot be compared safely.
SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")

#: Seconds any single git invocation in the ancestry resolver may take. A
#: resolver that hangs would hold the poll loop, and the poll loop is what
#: serves the queue endpoint's freshness.
GIT_TIMEOUT_SECONDS: Final = 30

#: Seconds the one-shot fetch may take when a sha is not in the clone yet.
#: Larger than the comparison above because it is a network round trip, and
#: still bounded, because an unfetchable sha is ``ANCESTRY_UNPROVEN`` rather
#: than something to wait out.
GIT_FETCH_TIMEOUT_SECONDS: Final = 120

#: Seconds before the resolver will bring the clone current again.
#:
#: The resolver instance lives as long as the agent process, so a plain
#: once-ever flag would mean the FIRST scan that met a sha the clone did not
#: have was the only one ever allowed to fetch -- every later scan would find
#: the ancestry unproven and coalescing would quietly stop working for the life
#: of the process. A cooldown keeps the property that actually matters, which
#: is that a batch of twelve commands costs at most one round trip, while
#: leaving a later poll free to try again. Well under the agent's measured mean
#: service time of 1799.8 s, so a fetch is available on every scan that follows
#: a real job.
GIT_FETCH_COOLDOWN_SECONDS: Final = 60


class EnumCoalesceRefusal(StrEnum):
    """Why a queued command was NOT folded into the group ahead of it.

    Every value ends the group. They are distinct rather than collapsed into
    one "not coalescable" because they send a reader to different places: a
    lane mismatch is the fence working, a pinned image is a promotion, and an
    unproven ancestry is a clone that needs a look.
    """

    DIFFERENT_LANE = "different_lane"
    NOT_FULL_SCOPE = "not_full_scope"
    SERVICES_DIFFER = "services_differ"
    BUILD_SOURCE_DIFFERS = "build_source_differs"
    PINNED_IMAGE = "pinned_image"
    REF_NOT_A_SHA = "ref_not_a_sha"
    ANCESTRY_UNPROVEN = "ancestry_unproven"
    NOT_A_DESCENDANT = "not_a_descendant"


class ModelQueuedCommand(BaseModel):
    """One validated command sitting in the batch this poll already fetched."""

    model_config = ConfigDict(frozen=True)

    command: ModelRebuildRequested
    #: Where the record sits on the control topic. Carried so the journal can
    #: name the queue position of the command that ran, which is the fact a
    #: reader of a coalesced job most wants and cannot otherwise recover.
    partition: int = Field(ge=0)
    offset: int = Field(ge=0)


class ModelSupersession(BaseModel):
    """One command that will not run, and the one that runs in its place."""

    model_config = ConfigDict(frozen=True)

    superseded: ModelQueuedCommand
    superseded_by_sha: str
    superseded_by_correlation_id: UUID

    @model_validator(mode="after")
    def _by_sha_is_exact(self) -> ModelSupersession:
        if not SHA_RE.match(self.superseded_by_sha):
            msg = (
                f"superseded_by_sha={self.superseded_by_sha!r} is not a "
                "40-character lowercase commit sha. A supersession names the "
                "exact commit that ran instead, or it names nothing useful."
            )
            raise ValueError(msg)
        if self.superseded.command.correlation_id == self.superseded_by_correlation_id:
            msg = (
                "a command cannot supersede itself "
                f"({self.superseded_by_correlation_id})"
            )
            raise ValueError(msg)
        return self


class ModelCoalescePlan(BaseModel):
    """Which queued command runs, which ones it replaces, and where it stopped."""

    model_config = ConfigDict(frozen=True)

    runner: ModelQueuedCommand
    superseded: tuple[ModelSupersession, ...] = ()
    #: Why the group ended. ``None`` only when the scan reached the end of the
    #: batch with everything folded -- in every other case a reason is
    #: recorded, because "the group stopped" with no reason is the shape that
    #: makes a coalescing decision unauditable.
    stop_reason: EnumCoalesceRefusal | None = None
    #: How many commands the scan looked at beyond the head, folded or not.
    #: Distinguishes "nothing was queued behind it" from "three were queued and
    #: none could be folded", which read identically from the plan alone.
    examined: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _runner_is_not_superseded(self) -> ModelCoalescePlan:
        runner_id = self.runner.command.correlation_id
        if any(
            s.superseded.command.correlation_id == runner_id for s in self.superseded
        ):
            msg = f"the runner {runner_id} appears in its own superseded list"
            raise ValueError(msg)
        return self

    @property
    def superseded_count(self) -> int:
        return len(self.superseded)

    def journal_line(self) -> str:
        """One line naming the decision, for the agent's journal.

        Names the queue position of the command that RAN, how many it
        replaced, and why the group ended -- the three facts a reader needs to
        tell a coalesced deploy from a plain one without reading the job store.
        """
        parts = [
            f"coalesce: running {self.runner.command.correlation_id} "
            f"(ref={self.runner.command.git_ref}) at queue position "
            f"{self.runner.partition}:{self.runner.offset}",
            f"superseding {self.superseded_count} command(s)",
            f"examined {self.examined} behind the head",
        ]
        if self.stop_reason is not None:
            parts.append(f"group ended on {self.stop_reason.value}")
        else:
            parts.append("group ran to the end of the fetched batch")
        return "; ".join(parts)


#: Resolves whether ``earlier`` is contained in ``later``.
#:
#: ``True`` proven contained, ``False`` proven NOT contained, ``None`` could not
#: be established. The third value is the point of the signature: an
#: unresolvable comparison must not read as either answer, and it is the value
#: every failure path of :class:`GitAncestryResolver` returns.
AncestryResolver = Callable[[str, str], bool | None]


def _refusal_for(
    head: ModelRebuildRequested,
    candidate: ModelRebuildRequested,
) -> EnumCoalesceRefusal | None:
    """The static half of the decision: everything that is not ancestry.

    Compared against the HEAD rather than against the previous member. All of
    these are equality tests, so the two are the same comparison, and naming
    the head is what makes the group's identity a single command's attributes
    rather than a chain a reader has to replay.
    """
    if candidate.runtime_lane != head.runtime_lane:
        return EnumCoalesceRefusal.DIFFERENT_LANE
    if head.scope is not Scope.FULL or candidate.scope is not Scope.FULL:
        return EnumCoalesceRefusal.NOT_FULL_SCOPE
    if list(candidate.services) != list(head.services):
        return EnumCoalesceRefusal.SERVICES_DIFFER
    if candidate.build_source != head.build_source:
        return EnumCoalesceRefusal.BUILD_SOURCE_DIFFERS
    if (
        head.image_ref
        or head.image_digest
        or candidate.image_ref
        or candidate.image_digest
    ):
        return EnumCoalesceRefusal.PINNED_IMAGE
    if not SHA_RE.match(head.git_ref) or not SHA_RE.match(candidate.git_ref):
        return EnumCoalesceRefusal.REF_NOT_A_SHA
    return None


def plan_coalesce(
    queued: Sequence[ModelQueuedCommand],
    *,
    contains: AncestryResolver,
) -> ModelCoalescePlan:
    """Decide which of a fetched batch runs, and which it supersedes.

    ``queued`` is in control-topic order, head first, and must be non-empty.
    The head is the command the agent has already validated and is about to
    accept; everything after it is look-ahead.

    Pure apart from ``contains``. Every host fact this decision needs is behind
    that one callable, which is what makes the rule table above testable
    without a clone.
    """
    if not queued:
        msg = "plan_coalesce needs at least the command being accepted"
        raise ValueError(msg)

    head = queued[0]
    group = [head]
    stop_reason: EnumCoalesceRefusal | None = None
    examined = 0

    for candidate in queued[1:]:
        examined += 1
        refusal = _refusal_for(head.command, candidate.command)
        if refusal is not None:
            stop_reason = refusal
            break
        # Consecutive comparison, not head-to-candidate: containment is
        # transitive, so a chain of proven links gives every member of the
        # group containment in the runner while costing one comparison per
        # command instead of one per pair.
        previous = group[-1]
        verdict = contains(previous.command.git_ref, candidate.command.git_ref)
        if verdict is None:
            stop_reason = EnumCoalesceRefusal.ANCESTRY_UNPROVEN
            break
        if not verdict:
            stop_reason = EnumCoalesceRefusal.NOT_A_DESCENDANT
            break
        group.append(candidate)

    runner = group[-1]
    supersessions = tuple(
        ModelSupersession(
            superseded=member,
            superseded_by_sha=runner.command.git_ref,
            superseded_by_correlation_id=runner.command.correlation_id,
        )
        for member in group[:-1]
    )
    return ModelCoalescePlan(
        runner=runner,
        superseded=supersessions,
        stop_reason=stop_reason,
        examined=examined,
    )


class GitAncestryResolver:
    """``contains`` answered by the deploy-source clone.

    THE FETCH IS DELIBERATELY NOT UNDER THE LANE LOCK, and the distinction is
    the whole reason this is safe. OMN-18124 put ``git_pull`` under the lane's
    per-compose-project lock because ``git reset --hard`` rewrites the shared
    WORKING TREE, and two writers rewrote it concurrently. Nothing here touches
    the tree, the index or ``HEAD``: ``cat-file``, ``merge-base`` and ``fetch``
    read the tree not at all and write only to ``.git``, where git takes its own
    ref locks. Taking the lane lock here would instead make this read wait on a
    deploy -- the exact deploy whose queue it is trying to shorten.

    Every failure is ``None``. A clone that cannot answer is a clone that has
    not answered, and the caller's contract is that an unproven comparison ends
    the group rather than guessing at it.
    """

    def __init__(
        self,
        repo_dir: str,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.repo_dir = repo_dir
        self._run = run or self._default_run
        self._clock = clock or time.monotonic
        #: When the clone was last brought current, or ``None`` for never. See
        #: ``GIT_FETCH_COOLDOWN_SECONDS`` for why this is a timestamp rather
        #: than the once-ever flag it started as.
        self._fetched_at: float | None = None

    @staticmethod
    def _default_run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )

    def __call__(self, earlier: str, later: str) -> bool | None:
        try:
            if not self._have_both(earlier, later):
                if not self._may_fetch():
                    return None
                self._fetch()
                if not self._have_both(earlier, later):
                    return None
            result = self._run(
                [
                    "git",
                    "-C",
                    self.repo_dir,
                    "merge-base",
                    "--is-ancestor",
                    earlier,
                    later,
                ],
                timeout=GIT_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - an unanswerable clone is unproven
            logger.info(
                "coalesce: ancestry %s -> %s is unproven (%s: %s)",
                earlier,
                later,
                type(exc).__name__,
                exc,
            )
            return None
        # git documents exactly two statuses for --is-ancestor: 0 yes, 1 no.
        # Anything else is a failure to ANSWER, not an answer, so it is
        # unproven rather than a refusal of the pair.
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        logger.info(
            "coalesce: git merge-base --is-ancestor exited %d for %s -> %s: %s",
            result.returncode,
            earlier,
            later,
            (result.stderr or "")[:200],
        )
        return None

    def _have_both(self, earlier: str, later: str) -> bool:
        for sha in (earlier, later):
            result = self._run(
                ["git", "-C", self.repo_dir, "cat-file", "-e", f"{sha}^{{commit}}"],
                timeout=GIT_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                return False
        return True

    def _may_fetch(self) -> bool:
        if self._fetched_at is None:
            return True
        return (self._clock() - self._fetched_at) >= GIT_FETCH_COOLDOWN_SECONDS

    def _fetch(self) -> None:
        self._fetched_at = self._clock()
        result = self._run(
            ["git", "-C", self.repo_dir, "fetch", "--quiet", "--no-tags", "origin"],
            timeout=GIT_FETCH_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            logger.info(
                "coalesce: fetch into %s exited %d, so a sha the clone does not "
                "have stays unproven: %s",
                self.repo_dir,
                result.returncode,
                (result.stderr or "")[:200],
            )
