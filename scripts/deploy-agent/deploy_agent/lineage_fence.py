# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Never rebuild what the lane already runs, and never rebuild it backwards (OMN-19270).

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

Behind those two came four more full rebuilds -- ``602b1d61``, ``5ce1f33e``,
``f645854b``, ``98bb76df`` -- each triggered by an omnimarket merge from
13:37Z-13:51Z, each at infra ``0edf5c914``, and each accepted hours later.
The trigger resolves the infra dev head when it publishes, so every ref was
right when it was sent. By the time each ran, the lane already vendored the
omnimarket merge it was sent to deliver. ``98bb76df``, the last, failed
post-deploy verification and left the runtime container in ``Created``.

Nothing in the accept protocol could have prevented any of it. ``coalesce``
folds a prefix of commands that arrived TOGETHER, so a stale command that
arrives alone is invisible to it. ``ref_fence`` refuses a stale branch ALIAS
and by design lets every 40-hex SHA through. The image already records what
it was built from (``infra_vcs_ref`` and each sibling's ``vcs_ref`` in
``/app/build-provenance.json``), but the agent never read it.

THE RULE: ANCESTRY AGAINST PROVENANCE, NEVER ORDERING
------------------------------------------------------
Every decision is git ancestry against the running build's provenance, never a
timestamp or an offset.

1. **Supersede what is already running.** A CI-triggered command
   (``requested_by`` ``gha/<repo>/...``) whose infra ref AND every named
   sibling ref are ancestors of, or equal to, the running build's refs is
   ``CONTAINED``: recorded ``superseded``, acknowledged with
   ``superseded_by_running_build``, never built.

   A command is only superseded on proof. A command triggered by a SIBLING
   that names no ref for that sibling (``sibling_refs``) cannot be shown to be
   delivered already, and it builds. Its new code is the sibling's, staged at
   build time, and superseding it on the infra ref alone would strand every
   sibling merge that lands while infra is quiet. A command from anyone other
   than CI -- an operator, lab-health triage -- is deliberate, and it is never
   superseded, because a same-ref rebuild is how a wedged lane is recovered.

2. **Never build backwards.** A command that is not superseded builds at the
   newer of its ref and the running ref. An infra ref that is a strict
   ancestor of the running build is ``RAISED`` to the running ref. Siblings need
   no raise: a workspace build stages each from its dev branch, which is never
   behind what the lane vendors.

3. **Diverged is refused unless it is the way home.** An infra ref that is
   neither ancestor nor descendant of the running build is ``DIVERGENT`` and
   refused with ``divergent_ref``, unless it is on the lane's tracking branch.
   That exception is ``RETURNS_TO_TRACKING``: a lane left on an off-branch
   build, such as a pre-PR proof at a PR head, must not refuse every dev command
   until someone declares a rollback.

4. **A declared rollback is the only way backwards.** A command carrying a
   signed ``ModelRollbackDeclaration`` is built exactly as asked, and
   coalescing never folds it.

5. **A symbolic ref is resolved at accept time.** ``origin/dev`` is resolved in
   the deploy clone after a fetch, and the job builds that exact sha and
   records it. Job ``c009462c`` asked for ``origin/dev`` and its record kept only
   the alias, so which commit it built had to be recovered from an image tag.

6. **Missing provenance fails open.** A running build that cannot be read, an
   alias that cannot be resolved, or an ancestry the clone cannot answer is
   ``UNPROVEN``, and the command builds as requested. A lane that is down has
   no container to read, and a lane that is down is exactly the lane that must
   accept a rebuild.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess  # fixed argv, no shell, trusted docker and git binaries
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from deploy_agent.coalesce import (
    GIT_FETCH_TIMEOUT_SECONDS,
    GIT_TIMEOUT_SECONDS,
    SHA_RE,
    AncestryResolver,
    GitAncestryResolver,
)
from deploy_agent.events import (
    INFRA_REPOSITORY,
    EnumLineageVerdict,
    EnumRuntimeLane,
    ModelLineageDecision,
    ModelRebuildRequested,
)

logger = logging.getLogger(__name__)

#: Where every runtime image records what it was built from. Written by
#: ``docker/Dockerfile.runtime`` for both workspace and release builds.
BUILD_PROVENANCE_PATH: Final = "/app/build-provenance.json"

#: Seconds the ``docker exec`` read of the provenance file may take. The read
#: runs on the poll path, and the poll path serves the queue endpoint's
#: freshness.
PROVENANCE_READ_TIMEOUT_SECONDS: Final = 15

#: ``requested_by`` of a command the CI rebuild trigger published:
#: ``gha/<source repository>/pr-<n>``.
_CI_REQUESTER_RE: Final = re.compile(r"^gha/([a-z][a-z0-9_]*)/")


class ModelRunningBuild(BaseModel):
    """What the lane's runtime image says it was built from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    infra_ref: str
    #: Sibling repository -> the commit the build vendored. A sibling the image
    #: recorded as dirty, or did not record, is absent: nothing can be proven
    #: contained in a tree nobody can name.
    sibling_refs: dict[str, str] = {}


#: Returns the lane's running build, or ``None`` when it cannot be read.
RunningBuildReader = Callable[[EnumRuntimeLane], ModelRunningBuild | None]

#: Resolves a symbolic ref to a 40-hex sha in the deploy clone, or ``None``.
RefResolver = Callable[[str], str | None]

#: ``(repository, earlier, later)`` -> whether ``earlier`` is an ancestor of,
#: or equal to, ``later`` in that sibling's clone; ``None`` when unanswerable.
SiblingAncestryResolver = Callable[[str, str, str], bool | None]


def ci_source_repository(requested_by: str) -> str | None:
    """The repository whose merge the CI trigger published this command for."""
    match = _CI_REQUESTER_RE.match(requested_by)
    return match.group(1) if match else None


def _containment(
    cmd: ModelRebuildRequested,
    running: ModelRunningBuild,
    contains_sibling: SiblingAncestryResolver,
) -> tuple[bool, str]:
    """Whether every sibling ref ``cmd`` names is in ``running``, and why not.

    Called only once the infra ref is known to be contained. Returns the
    reason in the negative case, because the journal line for a command that
    builds must say why it was not superseded.
    """
    source = ci_source_repository(cmd.requested_by)
    if source is None:
        return False, (
            f"requested by {cmd.requested_by!r}, not by a CI trigger; a "
            "deliberate request is never superseded"
        )
    if source != INFRA_REPOSITORY and source not in cmd.sibling_refs:
        return False, (
            f"triggered by a {source} merge but names no {source} ref, so it "
            "cannot be proven already delivered"
        )
    for repo, sha in sorted(cmd.sibling_refs.items()):
        running_sha = running.sibling_refs.get(repo)
        if running_sha is None:
            return False, f"the running build records no clean {repo} revision"
        if sha == running_sha:
            continue
        if contains_sibling(repo, sha, running_sha) is not True:
            return False, (
                f"{repo} {sha} is not shown to be in the running build's "
                f"{repo} {running_sha}"
            )
    return True, "every ref the command names is already in the running build"


def decide_lineage(
    cmd: ModelRebuildRequested,
    *,
    read_running_build: Callable[[], ModelRunningBuild | None],
    contains: AncestryResolver,
    contains_sibling: SiblingAncestryResolver,
    resolve_ref: RefResolver | None,
    tracking_ref: str | None,
) -> ModelLineageDecision:
    """Classify ``cmd`` against the running build. Pure apart from its callables.

    ``read_running_build`` is called only for a command the comparison applies
    to, so a promotion or a declared rollback never costs a ``docker exec``.
    """
    requested = cmd.git_ref

    def decision(
        verdict: EnumLineageVerdict,
        detail: str,
        *,
        build: str,
        resolved: str | None = None,
        running: str | None = None,
    ) -> ModelLineageDecision:
        return ModelLineageDecision(
            verdict=verdict,
            requested_ref=requested,
            resolved_ref=resolved,
            running_ref=running,
            build_ref=build,
            detail=detail,
        )

    if cmd.image_ref or cmd.image_digest:
        return decision(
            EnumLineageVerdict.NOT_APPLICABLE,
            "the command pins an image, which is a promotion and not a rebuild "
            "from a ref",
            build=requested,
        )

    ref = requested
    resolved: str | None = None
    if not SHA_RE.match(requested):
        answer_sha = resolve_ref(requested) if resolve_ref is not None else None
        if answer_sha is None or not SHA_RE.match(answer_sha):
            return decision(
                EnumLineageVerdict.UNPROVEN,
                f"the symbolic ref {requested!r} could not be resolved at accept "
                "time, so the command builds as requested",
                build=requested,
            )
        ref = resolved = answer_sha

    if cmd.rollback is not None:
        return decision(
            EnumLineageVerdict.ROLLBACK_DECLARED,
            f"rollback declared by {cmd.rollback.actor!r}: {cmd.rollback.reason}",
            build=ref,
            resolved=resolved,
        )

    running = read_running_build()
    if running is None:
        return decision(
            EnumLineageVerdict.UNPROVEN,
            "the running build's provenance could not be read, so the command "
            "builds as requested",
            build=ref,
            resolved=resolved,
        )
    running_ref = running.infra_ref

    def unproven(question: str) -> ModelLineageDecision:
        return decision(
            EnumLineageVerdict.UNPROVEN,
            f"{question} could not be established, so the command builds as requested",
            build=ref,
            resolved=resolved,
            running=running_ref,
        )

    if ref == running_ref:
        behind = False
    else:
        answer = contains(ref, running_ref)
        if answer is None:
            return unproven("whether the ref is an ancestor of the running build")
        behind = answer
        if not behind:
            ahead = contains(running_ref, ref)
            if ahead is None:
                return unproven("whether the ref descends from the running build")
            if ahead:
                return decision(
                    EnumLineageVerdict.DESCENDANT,
                    "the ref descends from the running build",
                    build=ref,
                    resolved=resolved,
                    running=running_ref,
                )
            if tracking_ref is not None and contains(ref, tracking_ref) is True:
                return decision(
                    EnumLineageVerdict.RETURNS_TO_TRACKING,
                    f"the ref has diverged from the running build and is on "
                    f"{tracking_ref}; the lane returns to its own lineage",
                    build=ref,
                    resolved=resolved,
                    running=running_ref,
                )
            return decision(
                EnumLineageVerdict.DIVERGENT,
                "the ref has diverged from the running build and is not shown "
                f"to be on the tracking branch {tracking_ref or '(undeclared)'}; "
                "a deliberate deploy of it carries a rollback declaration",
                build=ref,
                resolved=resolved,
                running=running_ref,
            )

    # The infra ref is at or behind the running build.
    contained, why = _containment(cmd, running, contains_sibling)
    if contained:
        return decision(
            EnumLineageVerdict.CONTAINED,
            why,
            build=running_ref,
            resolved=resolved,
            running=running_ref,
        )
    if behind:
        return decision(
            EnumLineageVerdict.RAISED,
            "the ref is a strict ancestor of the running build, so it builds at "
            f"the running ref instead of rolling the lane back; not superseded "
            f"because {why}",
            build=running_ref,
            resolved=resolved,
            running=running_ref,
        )
    return decision(
        EnumLineageVerdict.EQUAL,
        f"same infra ref as the running build; not superseded because {why}",
        build=ref,
        resolved=resolved,
        running=running_ref,
    )


def _clean_sibling_refs(provenance: dict[str, Any]) -> dict[str, str]:
    siblings = (provenance.get("per_repo_vcs_provenance") or {}).get("siblings")
    if not isinstance(siblings, dict):
        return {}
    refs: dict[str, str] = {}
    for repo, record in siblings.items():
        if not isinstance(record, dict) or record.get("vcs_dirty") is not False:
            continue
        sha = record.get("vcs_ref")
        if isinstance(sha, str) and SHA_RE.match(sha):
            refs[str(repo)] = sha
    return refs


class DockerProvenanceReader:
    """Reads the running build out of a lane's runtime container.

    Every failure is ``None``: a container that is absent, stopped, or carries
    a provenance file this cannot parse has no ref to compare against, and the
    caller's contract is that an unread build lets the command run.
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

    def __call__(self, lane: EnumRuntimeLane) -> ModelRunningBuild | None:
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
            provenance = json.loads(result.stdout)
            ref = provenance.get("infra_vcs_ref")
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
        return ModelRunningBuild(
            infra_ref=ref, sibling_refs=_clean_sibling_refs(provenance)
        )


class GitRefResolver:
    """Resolves a symbolic ref to the sha it names NOW, in the deploy clone.

    Fetches first and unconditionally: the point is to record the commit the
    job will build, and an alias resolved against a stale remote-tracking ref
    names a commit the git phase's own fetch would move past. A failed fetch
    is ``None`` rather than a resolution against what the clone happens to
    hold, for the same reason; the command then builds as requested and the
    git phase reports its own fetch failure.
    """

    def __init__(
        self,
        repo_dir: str,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.repo_dir = repo_dir
        self._run = run or GitAncestryResolver._default_run

    def __call__(self, ref: str) -> str | None:
        try:
            fetched = self._run(
                ["git", "-C", self.repo_dir, "fetch", "--quiet", "--no-tags", "origin"],
                timeout=GIT_FETCH_TIMEOUT_SECONDS,
            )
            if fetched.returncode != 0:
                logger.info(
                    "lineage: fetch into %s exited %d, so %r stays unresolved: %s",
                    self.repo_dir,
                    fetched.returncode,
                    ref,
                    (fetched.stderr or "")[:200],
                )
                return None
            result = self._run(
                [
                    "git",
                    "-C",
                    self.repo_dir,
                    "rev-parse",
                    "--verify",
                    "--quiet",
                    f"{ref}^{{commit}}",
                ],
                timeout=GIT_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - an unresolvable ref is unresolved
            logger.info(
                "lineage: %r could not be resolved (%s: %s)",
                ref,
                type(exc).__name__,
                exc,
            )
            return None
        sha = (result.stdout or "").strip()
        if result.returncode != 0 or not SHA_RE.match(sha):
            logger.info("lineage: %r does not name a commit in %s", ref, self.repo_dir)
            return None
        return sha


class SiblingCloneAncestry:
    """Answers sibling ancestry in the clone the workspace build stages from.

    The build stages ``<OMNI_HOME>/<repository>``, so that clone is the one
    whose history the vendored ``vcs_ref`` came from. One ``GitAncestryResolver``
    per repository keeps each clone's fetch cooldown separate. With no
    ``OMNI_HOME``, or a clone that is not there, every answer is ``None``, and
    a command is then never superseded on a sibling it could not check.
    """

    def __init__(
        self,
        omni_home: str | None,
        *,
        resolver_for: Callable[[str], AncestryResolver] | None = None,
    ) -> None:
        self._omni_home = omni_home
        self._resolver_for = resolver_for or GitAncestryResolver
        self._resolvers: dict[str, AncestryResolver] = {}

    def __call__(self, repo: str, earlier: str, later: str) -> bool | None:
        if not self._omni_home:
            return None
        clone = Path(self._omni_home) / repo
        if not (clone / ".git").exists():
            logger.info("lineage: no %s clone at %s to compare in", repo, clone)
            return None
        resolver = self._resolvers.get(repo)
        if resolver is None:
            resolver = self._resolvers[repo] = self._resolver_for(str(clone))
        return resolver(earlier, later)
