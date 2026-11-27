# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Run the newest queued rebuild for a lane, and record the ones it replaced (OMN-18143).
#
# WHAT THIS COSTS TODAY, MEASURED
# -------------------------------
# The agent drains its control topic in commit order at a mean service time of
# 1799.8 s per job (``queue_depth.mean_service_time`` the night this was written).
# At 2026-09-19T02:58Z job ``52e430b2`` was staging ``f8475f9b``, sixteen commits
# behind ``origin/dev``. With roughly forty runtime-affecting merges in one night
# the lane spends hours rebuilding shas a later commit has already replaced,
# while the dev head waits at the back. The queue endpoint (OMN-18144) only
# observes; this module is the acting half.
#
# WHAT IS COALESCED, AND WHAT IS REFUSED
# ---------------------------------------
# A PREFIX of the queue, and only a prefix. The scan starts at the command the
# agent is about to accept and walks forward while each next command is
# foldable into the same group; the FIRST command that is not ends the group and
# stays queued, with everything behind it, so ordering is preserved exactly.
# Each refusal below ends the group and has its own reason in the journal:
#
# ``DIFFERENT_LANE``
#     Different lanes mutate different compose projects; folding drops a deploy.
#
# ``ROLLBACK_DECLARED``
#     A command carrying a signed rollback declaration (OMN-19270) is never
#     folded, in either direction: a newer command would undo the deliberate one.
#
# ``NOT_FULL_SCOPE``
#     Only ``scope=full`` folds into ``scope=full``: the scope decides WHICH
#     services are recreated, so mixing scopes silently drops services. Narrower
#     than it has to be, because a ``core`` rebuild is not the job this stops.
#
# ``SERVICES_DIFFER`` / ``BUILD_SOURCE_DIFFERS``
#     Same argument. A command naming services, or built from a different
#     source, is not the same work as the one ahead of it.
#
# ``PINNED_IMAGE``
#     A command carrying ``image_ref`` or ``image_digest`` is a promotion of one
#     exact artifact, never folded. Every ``prod`` command trips this, since
#     ``ModelRebuildRequested`` refuses a prod request that carries no digest.
#
# ``REF_NOT_A_SHA``
#     A branch alias resolves at BUILD time, so two aliases are not comparable
#     by ancestry. Only 40-hex pins fold.
#
# ``ANCESTRY_UNPROVEN`` / ``NOT_A_DESCENDANT``
#     The load-bearing one. Coalescing is only safe when the sha that RUNS
#     contains the sha it replaces, or the replaced change is dropped silently.
#     The check is ``git merge-base --is-ancestor <earlier> <later>`` on
#     consecutive members, transitive across the group. An unresolvable
#     comparison (missing object, git failure, timeout) is ``ANCESTRY_UNPROVEN``
#     and ends the group, so every failure falls back to running every command
#     in order, exactly as before.
#
# ``NOT_A_DESCENDANT`` AND THE JUMP (OMN-26316)
#     Refs that are each contained in a LATER command need not contain one
#     another. See ``_newest_containing``: the group jumps to the newest command
#     that contains everything it passes over, and otherwise ends here.
#
# ``QUEUE SUPERSEDE`` (OMN-26637)
#     For a lane named in ``supersede_lanes`` the ancestry questions above are not
#     asked at all. Operator ruling 2026-11-24: when a new build comes out, every
#     older queued build is cancelled. The static refusals still end the group,
#     because they name work the newest command does not do (another lane, a
#     declared rollback, a narrower scope, a pinned image); only the question
#     "does the newer sha contain the older one" is dropped. Squash merges make
#     sibling PR commits non-ancestors of each other's heads, so that question
#     ended the group and the stale commands behind it each ran for ~28 minutes.
#
# ``DUPLICATE_COMMAND``
#     A look-ahead record whose correlation id is already in the group is a
#     redelivered copy, not newer work (OMN-19521): folding it recorded the first
#     copy as superseded by its own id, and that raise crash-looped the agent on
#     2026-09-25. The copy ends the group and stays queued.
#
# WHY THERE IS NO KILL SWITCH
# ----------------------------
# Every failure mode above degrades to the pre-change behaviour by construction,
# and an opt-out left set is how a lane silently loses the optimisation (rule 5).
#
# WHAT A SUPERSEDED COMMAND GETS
# -------------------------------
# Not silence, and not a failure: a durable job record with status
# ``superseded`` naming the sha and correlation id that ran in its place, and a
# terminal event on the rejection topic carrying the same two fields (OMN-18143 AC6).

from __future__ import annotations

import logging
import re
import runpy
import subprocess  # fixed argv, no shell, trusted git binary
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested

logger = logging.getLogger(__name__)

#: node_queued_request_priority_compute's way in (OMN-26718), loaded from the agent's own checkout.
entry = runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "src/omnibase_infra/handlers/queued_request_priority/agent_entry.py"
    )
)

#: A full 40-character lowercase commit sha. The same shape
#: ``scripts/ci/lab_pass_receipt.py`` gates on, and for the same reason: an
#: abbreviated ref cannot be compared safely.
SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")

#: The requester of the overdue dev merge build check's commands (OMN-26316); the node's publish
#: handler issues them under this name.
OVERDUE_BUILD_REQUESTER: Final = "agent/overdue-dev-merge-build"

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

    Every value ends the group, and each sends a reader somewhere different.
    """

    DIFFERENT_LANE = "different_lane"
    ROLLBACK_DECLARED = "rollback_declared"
    NOT_FULL_SCOPE = "not_full_scope"
    SERVICES_DIFFER = "services_differ"
    BUILD_SOURCE_DIFFERS = "build_source_differs"
    PINNED_IMAGE = "pinned_image"
    REF_NOT_A_SHA = "ref_not_a_sha"
    ANCESTRY_UNPROVEN = "ancestry_unproven"
    NOT_A_DESCENDANT = "not_a_descendant"
    DUPLICATE_COMMAND = "duplicate_command"
    OPERATOR_REQUEST = "operator_request"


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
    #: ``True`` when ancestry proved the runner contains the replaced ref;
    #: ``False`` when the lane's queue-supersede rule replaced it without asking
    #: (OMN-26637). The job record states which, so a reader never has to infer.
    by_containment: bool = True
    #: What the job record of the superseded command says, and it names ``superseded_by``.
    reason: str

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

        Names the runner's queue position, how many it replaced, and why the
        group ended.
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


def _newest_containing(
    queued: Sequence[ModelQueuedCommand],
    index: int,
    limit: int,
    group: Sequence[ModelQueuedCommand],
    known: AncestryResolver,
) -> int | None:
    """Position of the newest command that safely replaces ``queued[index:]``'s run."""
    # OMN-26316. ``queued[index]`` does not contain the group's last member,
    # but the NEWEST command of the foldable run behind it may contain that
    # member and every command between. Two PR heads and the merge carrying
    # both are not ancestor-related pairwise, and the consecutive scan ended
    # at the first, rebuilding each stale ref in turn at a mean service time
    # over 1700 s (queue_depth, 2026-11-24). Every command passed over must
    # pass the static rules, be no redelivered copy and be PROVEN contained in
    # the runner: anything unproven or negative returns None and the group ends
    # where the consecutive scan ended. Only the newest command is a candidate,
    # so a poll asks at most one question per queued command. The head was
    # vetted by the lineage fence, so the runner descends from the running ref.
    run = queued[index:limit]
    if len(run) < 2:
        return None
    runner_ref = run[-1].command.git_ref
    members = [group[-1], *run[:-1]]
    if all(known(m.command.git_ref, runner_ref) is True for m in members):
        return index + len(run) - 1
    return None


def plan_coalesce(
    queued: Sequence[ModelQueuedCommand],
    *,
    contains: AncestryResolver,
    supersede_lanes: frozenset[EnumRuntimeLane] = frozenset(),
) -> ModelCoalescePlan:
    """Decide which of a fetched batch runs, and which it supersedes.

    ``queued`` is in control-topic order, head first, and must be non-empty.
    The head is the validated command about to be accepted; the rest is
    look-ahead. Pure apart from ``contains``, the one callable behind every
    host fact.

    ``supersede_lanes`` names the lanes whose queue is superseded outright
    (OMN-26637): for a head on one of them every look-ahead command that passes
    the static refusals folds into the newest, with no ancestry question.
    """
    if not queued:
        msg = "plan_coalesce needs at least the command being accepted"
        raise ValueError(msg)

    head = queued[0]
    group = [head]
    stop_reason: EnumCoalesceRefusal | None = None
    examined = 0
    supersede_queue = head.command.runtime_lane in supersede_lanes
    answers: dict[tuple[str, str], bool | None] = {}

    def known(earlier: str, later: str) -> bool | None:
        # One answer per pair per plan; each is up to three git calls.
        if (earlier, later) not in answers:
            answers[earlier, later] = contains(earlier, later)
        return answers[earlier, later]

    # OMN-26718. node_queued_request_priority_compute owns the static half (which queued
    # requests may fold, and an operator-path request running ahead of automatic ones).
    handler = entry["handler_class"]()
    fold = handler().handle(entry["request"](queued, supersede_queue=supersede_queue))
    limit, index = fold.foldable, 1
    if fold.operator_path:
        group, index = list(queued[:limit]), limit
    while index < limit:
        candidate = queued[index]
        examined += 1
        if supersede_queue:
            # OMN-26637. Newest wins, whatever git says about the pair.
            group.append(candidate)
            index += 1
            continue
        # Consecutive comparison, not head-to-candidate: containment is
        # transitive, so a chain of proven links gives every member of the
        # group containment in the runner while costing one comparison per
        # command instead of one per pair.
        verdict = known(group[-1].command.git_ref, candidate.command.git_ref)
        if verdict is None:
            stop_reason = EnumCoalesceRefusal.ANCESTRY_UNPROVEN
            break
        newest: int | None = index
        if not verdict:
            newest = _newest_containing(queued, index, limit, group, known)
        if newest is None:
            stop_reason = EnumCoalesceRefusal.NOT_A_DESCENDANT
            break
        examined += newest - index
        group.extend(queued[index : newest + 1])
        index = newest + 1
    else:
        stop_reason = EnumCoalesceRefusal(fold.refusal) if fold.refusal else None
        examined += bool(fold.refusal)

    runner = group[-1]
    # OMN-26637. Exactly one reason, and it is true: containment is claimed only when ancestry
    # proved it. The priority node's reason (OMN-26718) names the operator-path request.
    why = (
        "a newer request for this lane that was accepted while this one was queued; the "
        "lane cancels every older queued request whatever its git ancestry"
        if supersede_queue
        else "which contains this command's ref and was already queued when the agent "
        "reached this one"
    )
    supersessions = tuple(
        ModelSupersession(
            superseded=member,
            superseded_by_sha=runner.command.git_ref,
            superseded_by_correlation_id=runner.command.correlation_id,
            by_containment=not supersede_queue,
            reason=reason
            or f"superseded by {runner.command.correlation_id} at "
            f"{runner.command.git_ref}, {why}",
        )
        for member, reason in zip(
            group[:-1], fold.reasons or ("",) * len(group), strict=False
        )
    )
    return ModelCoalescePlan(
        runner=runner,
        superseded=supersessions,
        stop_reason=stop_reason,
        examined=examined,
    )


class GitAncestryResolver:
    """``contains`` answered by the deploy-source clone.

    THE FETCH IS DELIBERATELY NOT UNDER THE LANE LOCK. OMN-18124 locked
    ``git_pull`` because ``git reset --hard`` rewrites the shared WORKING TREE;
    ``cat-file``, ``merge-base`` and ``fetch`` touch only ``.git``, where git
    takes its own ref locks, and the lane lock would make this read wait on the
    very deploy whose queue it is shortening. Every failure is ``None``: an
    unproven comparison ends the group rather than guessing.
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
            argv, timeout=timeout, capture_output=True, text=True, check=False
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
