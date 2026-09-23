# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Refuse a rebuild whose ref is behind the build the lane already runs (OMN-19270).

WHAT THIS COST, MEASURED
------------------------
On 2026-09-23 the .201 dev lane's runtime image tags record each build's
infra ref. Job ``c009462c`` built the lane at ``0edf5c914`` (image
``20260923T141329Z-0edf5c91``). At 14:32:21Z the agent accepted job
``ab27aedd`` (``scope=full``, ``git_ref=c159b7118``, an infra commit from
10:32:14Z whose command had waited in the queue for hours) and rebuilt the lane
onto it (``20260923T144642Z-c159b711``). ``c159b7118`` is a strict ancestor of
``0edf5c914``, so that deploy moved the lane BACKWARDS by every commit in
between. The next command, ``e074126b`` at ``533b19c23``, was also behind
``0edf5c914``. The lane did not return to ``0edf5c914`` until ``602b1d61``
completed at 17:18:52Z, nearly three hours after the rollback.

Nothing in the accept protocol could have refused it. ``coalesce`` folds a
prefix of commands that arrived TOGETHER, so a stale command that arrives alone
is invisible to it. ``ref_fence`` refuses a stale branch ALIAS and by design
lets every 40-hex SHA through, and the deploy-publish path pins SHAs. The
image already records what it was built from (``infra_vcs_ref`` in
``/app/build-provenance.json``), but the agent never read it.

THE RULE: ANCESTRY, NEVER ORDERING
-----------------------------------
"Older" is decided by git ancestry against the running build's
``infra_vcs_ref``, never by a timestamp or an offset. Only a 40-hex SHA
command is compared:

``STALE_ANCESTOR`` (refused, ``superseded_by_running_build``)
    The ref is a strict ancestor of the running build. The lane already runs
    that work, and building it would roll the lane back.

``DESCENDANT`` (accepted)
    The running build is an ancestor of the ref. This is the normal forward
    deploy.

``EQUAL`` (not refused here)
    The ref IS the running build's ref. A command at the same infra ref is NOT
    a duplicate build. A sibling-repository merge publishes the infra dev HEAD
    as its ``git_ref`` and gets the sibling's new code only because the build
    stages each sibling at build time. Refusing an equal ref would strand every
    sibling merge that lands while infra is quiet. An equal-ref command
    therefore goes through the existing duplicate path, which refuses the same
    correlation id twice and nothing else.

``RETURNS_TO_TRACKING`` (accepted)
    The ref and the running build have diverged, and the ref is on the lane's
    own tracking branch. This is a lane coming back to its lineage after it ran
    a build from off that branch.

``DIVERGENT`` (refused, ``divergent_ref``)
    The ref and the running build have diverged, and the ref is not shown to
    be on the tracking branch. Building it would move the lane sideways onto
    code nothing merged. The refusal stands when the tracking-branch check
    cannot be answered, because an unproven exemption does not lift a proven
    divergence.

``ROLLBACK_DECLARED`` (accepted)
    The command carries a signed ``ModelRollbackDeclaration`` naming an actor
    and a reason. Deliberate rollbacks MUST get through, or this fence would
    remove the way to recover from a bad build.

WHAT FALLS BACK TO THE PRE-CHANGE BEHAVIOUR
-------------------------------------------
``NOT_APPLICABLE`` covers a command that pins an image, which is a promotion of
an exact artifact and not a rebuild from a ref, and a branch alias, which
``ref_fence`` owns. ``UNPROVEN`` covers a running build whose ref cannot be read
(a lane that is down has no container to read, and a lane that is down is
exactly the lane that must accept a rebuild) and an ancestry question the clone
cannot answer. Both are accepted, and both write a journal line. This is the
same rule ``coalesce`` follows: a comparison the host cannot make degrades to
running the command as before, rather than to a refusal nothing can correct.
"""

from __future__ import annotations

import json
import logging
import subprocess  # fixed argv, no shell, trusted docker binary
from collections.abc import Callable
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

from deploy_agent.coalesce import SHA_RE, AncestryResolver
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested

logger = logging.getLogger(__name__)

#: Where every runtime image records what it was built from. Written by
#: ``docker/Dockerfile.runtime`` for both workspace and release builds.
BUILD_PROVENANCE_PATH: Final = "/app/build-provenance.json"

#: Seconds the ``docker exec`` read of the provenance file may take. The read
#: runs on the poll path, and the poll path serves the queue endpoint's
#: freshness.
PROVENANCE_READ_TIMEOUT_SECONDS: Final = 15

#: Returns the infra commit the lane's running build was made from, or ``None``
#: when it cannot be read. ``None`` is a measured absence, never a guess.
RunningBuildRefReader = Callable[[EnumRuntimeLane], str | None]


class EnumLineageVerdict(StrEnum):
    """Where a command's ref sits relative to the build the lane runs."""

    DESCENDANT = "descendant"
    EQUAL = "equal"
    RETURNS_TO_TRACKING = "returns_to_tracking"
    ROLLBACK_DECLARED = "rollback_declared"
    NOT_APPLICABLE = "not_applicable"
    UNPROVEN = "unproven"
    STALE_ANCESTOR = "stale_ancestor"
    DIVERGENT = "divergent"

    @property
    def refuses(self) -> bool:
        return self in (EnumLineageVerdict.STALE_ANCESTOR, EnumLineageVerdict.DIVERGENT)


class ModelLineageDecision(BaseModel):
    """One command's lineage verdict and the facts it was reached from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumLineageVerdict
    requested_ref: str
    running_ref: str | None
    detail: str

    def journal_line(self) -> str:
        return (
            f"lineage: {self.verdict.value} (requested={self.requested_ref} "
            f"running={self.running_ref or 'unread'}): {self.detail}"
        )


def decide_lineage(
    cmd: ModelRebuildRequested,
    *,
    read_running_ref: Callable[[], str | None],
    contains: AncestryResolver,
    tracking_ref: str | None,
) -> ModelLineageDecision:
    """Classify ``cmd`` against the running build. Pure apart from its callables.

    ``read_running_ref`` is called only for a command the fence applies to, so
    a promotion or an alias never costs a ``docker exec``.
    """
    requested = cmd.git_ref

    def decision(
        verdict: EnumLineageVerdict, detail: str, running: str | None = None
    ) -> ModelLineageDecision:
        return ModelLineageDecision(
            verdict=verdict,
            requested_ref=requested,
            running_ref=running,
            detail=detail,
        )

    if cmd.rollback is not None:
        return decision(
            EnumLineageVerdict.ROLLBACK_DECLARED,
            f"rollback declared by {cmd.rollback.actor!r}: {cmd.rollback.reason}",
        )
    if cmd.image_ref or cmd.image_digest:
        return decision(
            EnumLineageVerdict.NOT_APPLICABLE,
            "the command pins an image, which is a promotion and not a rebuild "
            "from a ref",
        )
    if not SHA_RE.match(requested):
        return decision(
            EnumLineageVerdict.NOT_APPLICABLE,
            "the ref is not a 40-hex commit sha; a branch alias is ref_fence's",
        )

    running = read_running_ref()
    if running is None or not SHA_RE.match(running):
        return decision(
            EnumLineageVerdict.UNPROVEN,
            "the running build's infra_vcs_ref could not be read, so there is "
            "nothing to compare against and the command runs as before",
            running,
        )
    if requested == running:
        return decision(
            EnumLineageVerdict.EQUAL,
            "same infra ref as the running build; a sibling-triggered rebuild "
            "carries new sibling code at the same ref, so only the correlation-id "
            "duplicate check applies",
            running,
        )

    behind = contains(requested, running)
    if behind is None:
        return decision(
            EnumLineageVerdict.UNPROVEN,
            "whether the ref is an ancestor of the running build could not be "
            "established",
            running,
        )
    if behind:
        return decision(
            EnumLineageVerdict.STALE_ANCESTOR,
            "the ref is a strict ancestor of the running build; building it "
            "would roll the lane back",
            running,
        )

    ahead = contains(running, requested)
    if ahead is None:
        return decision(
            EnumLineageVerdict.UNPROVEN,
            "whether the ref descends from the running build could not be established",
            running,
        )
    if ahead:
        return decision(
            EnumLineageVerdict.DESCENDANT,
            "the ref descends from the running build",
            running,
        )

    if tracking_ref is not None and contains(requested, tracking_ref) is True:
        return decision(
            EnumLineageVerdict.RETURNS_TO_TRACKING,
            f"the ref has diverged from the running build and is on "
            f"{tracking_ref}; the lane returns to its own lineage",
            running,
        )
    return decision(
        EnumLineageVerdict.DIVERGENT,
        "the ref has diverged from the running build and is not shown to be on "
        f"the tracking branch {tracking_ref or '(undeclared)'}; a deliberate "
        "deploy of it carries a rollback declaration",
        running,
    )


class DockerProvenanceReader:
    """Reads ``infra_vcs_ref`` out of a lane's running runtime container.

    Every failure is ``None``: a container that is absent, stopped, or carries
    a provenance file this cannot parse has no ref to compare against, and the
    caller's contract is that an unread ref lets the command run.
    """

    def __init__(
        self,
        container_for_lane: Callable[[EnumRuntimeLane], str],
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._container_for_lane = container_for_lane
        self._run = run or self._default_run

    @staticmethod
    def _default_run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )

    def __call__(self, lane: EnumRuntimeLane) -> str | None:
        try:
            container = self._container_for_lane(lane)
            result = self._run(
                ["docker", "exec", container, "cat", BUILD_PROVENANCE_PATH],
                timeout=PROVENANCE_READ_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - an unreadable build is unread
            logger.info(
                "lineage: the running build of lane %s could not be read (%s: %s)",
                lane.value,
                type(exc).__name__,
                exc,
            )
            return None
        if result.returncode != 0:
            logger.info(
                "lineage: reading %s from %s exited %d: %s",
                BUILD_PROVENANCE_PATH,
                container,
                result.returncode,
                (result.stderr or "")[:200],
            )
            return None
        try:
            ref = json.loads(result.stdout).get("infra_vcs_ref")
        except (ValueError, AttributeError) as exc:
            logger.info(
                "lineage: %s in %s is not a JSON object (%s)",
                BUILD_PROVENANCE_PATH,
                container,
                exc,
            )
            return None
        if not isinstance(ref, str) or not SHA_RE.match(ref):
            logger.info(
                "lineage: %s in %s carries infra_vcs_ref=%r, not a 40-hex sha",
                BUILD_PROVENANCE_PATH,
                container,
                ref,
            )
            return None
        return ref
