# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev lane built from an unmerged branch goes back to dev by itself (OMN-22996).

THE FAILURE
-----------
On 2026-11-07 the dev lane kept the build of omnibase_infra#5755's branch
(commit 29dd9aa2c765) for hours after #5755 merged at about 10:30Z. A lane that
proves a fix puts the lane on the pull request's branch with a declared open pull
request (``open_pr``, ``<repo>#<number>@<head sha>``, on the rebuild command); nothing
kept that as a record the agent acts on, so nothing returned the lane, and a
collaborator's release could not pass its lab check for dev until a person noticed.

WHAT THIS IS
------------
On every idle tick the deploy agent runs ``settle`` over its own state:

* The agent's job records are read. When the newest successful dev runtime job
  declared an open pull request, a :class:`ModelDevLaneBranchBuild` is recorded for it
  (repository, pull request, branch read from the clone, head, requester, job, expiry
  at most seven days out). A record that a LATER successful dev job replaced is dropped:
  the lane was built again, from dev or from another branch.
* A record whose lane already runs a commit on the dev branch is dropped as well.
* A remaining record is due for its return when its pull request is on the dev branch
  (a squash merge leaves a commit whose subject ends ``(#<pr>)``; a merge commit makes
  the head an ancestor), when its branch is gone from origin and no merge is on dev (a
  closed pull request), or when it has expired.

The agent then runs ONE rebuild at the tracking ref through its ordinary deploy path (a
job record, a terminal event, the lane lock). That job succeeding is the release: it is
the later successful dev job that drops the record at the next ``settle``.

Anything unread (no clone, a git error, an unreachable origin) keeps the record: a
return is only started on evidence. An expired record is the one reason that needs no
read, so a build whose pull request never merges still comes back.

DEV LANE ONLY. The agent calls this for its dev lane and never for another. NO
DEFAULTS (rule 8): the state directory, the jobs directory, the dev ref, the clones and
the git timeout are always named by the caller.

Command line, run by the agent from the deploy-source clone's ``src``::

    python -m omnibase_infra.handlers.handler_dev_lane_return_to_dev '<request json>'
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_dev_lane_return_reason import EnumDevLaneReturnReason
from omnibase_infra.models.model_dev_lane_branch_build import (
    MAX_LIFETIME,
    ModelDevLaneBranchBuild,
)
from omnibase_infra.models.model_dev_lane_branch_builds import (
    ModelDevLaneBranchBuilds,
)
from omnibase_infra.models.model_dev_lane_return_due import ModelDevLaneReturnDue
from omnibase_infra.models.model_dev_lane_return_to_dev_request import (
    ModelDevLaneReturnToDevRequest,
)
from omnibase_infra.models.model_dev_lane_return_to_dev_result import (
    ModelDevLaneReturnToDevResult,
)

logger = logging.getLogger(__name__)

RECORD_FILE_NAME = "records.json"
RETURN_RETRY_AFTER = timedelta(minutes=60)
REQUESTER = "agent/return-to-dev"
MERGE_SEARCH_DEPTH = 500
_NO_MATCHING_REF = 2
_OPEN_PR = re.compile(r"^([a-z][a-z0-9_]*)#([1-9][0-9]*)@([0-9a-f]{40})$")
_RUNTIME_SCOPES = frozenset({"runtime", "full"})


def record_path(state_dir: Path) -> Path:
    """Where the record of ``state_dir`` lives."""
    return state_dir / RECORD_FILE_NAME


def read_records(state_dir: Path) -> ModelDevLaneBranchBuilds:
    """The record file; empty when it is absent, unreadable or invalid."""
    target = record_path(state_dir)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ModelDevLaneBranchBuilds()
    except OSError as exc:
        logger.warning(
            "branch builds %s unreadable, ignored: %s "
            "friction_type=branch_builds_unreadable",
            target,
            exc,
        )
        return ModelDevLaneBranchBuilds()
    try:
        return ModelDevLaneBranchBuilds.model_validate_json(text)
    except ValidationError as exc:
        logger.warning(
            "branch builds %s invalid, ignored: %s "
            "friction_type=branch_builds_unreadable",
            target,
            exc.errors(include_url=False, include_input=False),
        )
        return ModelDevLaneBranchBuilds()


def read_builds(state_dir: Path) -> tuple[ModelDevLaneBranchBuild, ...]:
    """Every recorded build."""
    return read_records(state_dir).builds


def write_records(state_dir: Path, body: ModelDevLaneBranchBuilds) -> Path:
    """Write the record atomically."""
    target = record_path(state_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".records.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body.model_dump_json(indent=2) + "\n")
        Path(tmp).replace(target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def _git(
    clone: Path, timeout: int, *args: str
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(clone), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _is_ancestor(clone: Path, timeout: int, commit: str, ref: str) -> bool:
    done = _git(clone, timeout, "merge-base", "--is-ancestor", commit, ref)
    return done is not None and done.returncode == 0


def _merge_commit(
    clone: Path, timeout: int, dev_ref: str, build: ModelDevLaneBranchBuild
) -> str | None:
    """The commit on ``dev_ref`` that carries the pull request, or None.

    A squash merge leaves a commit whose subject ends ``(#<pr>)`` and never makes the
    head an ancestor; a merge commit does. Both shapes are read.
    """
    log = _git(
        clone, timeout, "log", f"-n{MERGE_SEARCH_DEPTH}", "--format=%H%x1f%s", dev_ref
    )
    if log is not None and log.returncode == 0:
        for line in log.stdout.splitlines():
            sha, _, subject = line.partition("\x1f")
            if len(sha) == 40 and subject.rstrip().endswith(f"(#{build.pr})"):
                return sha
    return build.head if _is_ancestor(clone, timeout, build.head, dev_ref) else None


def _branch_gone(clone: Path, timeout: int, branch: str) -> bool:
    """True only when origin answers and holds no such branch."""
    done = _git(clone, timeout, "ls-remote", "--exit-code", "--heads", "origin", branch)
    return done is not None and done.returncode == _NO_MATCHING_REF


def _branch_at(clone: Path | None, timeout: int, head: str) -> str:
    """The origin branch whose tip is ``head``, or empty when none is."""
    if clone is None:
        return ""
    done = _git(
        clone,
        timeout,
        "for-each-ref",
        "--points-at",
        head,
        "--format=%(refname:lstrip=3)",
        "refs/remotes/origin",
    )
    names = [n for n in (done.stdout.split() if done else []) if n != "HEAD"]
    return names[0] if names else ""


def _declared_builds(
    request: ModelDevLaneReturnToDevRequest,
) -> tuple[datetime | None, ModelDevLaneBranchBuild | None]:
    """The newest successful dev runtime job, and the build it declared, if any."""
    if request.jobs_dir is None or request.now is None:
        return None, None
    newest: tuple[datetime, dict[str, object], str] | None = None
    for path in request.jobs_dir.glob("*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            command = job["command"]
            done = datetime.fromisoformat(job["completed_at"])
            ok = (
                job["status"] == "success"
                and command["runtime_lane"] == "dev"
                and command["scope"] in _RUNTIME_SCOPES
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if ok and done.tzinfo is not None and (newest is None or done > newest[0]):
            newest = (done, command, str(job["correlation_id"]))
    if newest is None:
        return None, None
    done, command, job_id = newest
    matched = _OPEN_PR.match(str(command.get("open_pr", "")))
    if matched is None:
        return done, None
    repo, pr, head = matched.group(1), int(matched.group(2)), matched.group(3)
    try:
        return done, ModelDevLaneBranchBuild(
            repo=repo,
            pr=pr,
            branch=_branch_at(
                request.clones.get(repo), request.git_timeout_seconds, head
            ),
            head=head,
            requested_by=str(command["requested_by"]),
            job_id=job_id,
            recorded_at=done,
            expires_at=done + MAX_LIFETIME,
        )
    except (ValidationError, KeyError):
        return done, None


class HandlerDevLaneReturnToDev:
    """Record dev-lane branch builds and say which are due to go back to dev."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelDevLaneReturnToDevRequest
    ) -> ModelDevLaneReturnToDevResult:
        """Run the requested operation; the node's single entry point."""
        return run(request)


def run(request: ModelDevLaneReturnToDevRequest) -> ModelDevLaneReturnToDevResult:
    """Record one build, or settle every record against its pull request."""
    records = read_records(request.state_dir)
    if request.operation == "record":
        if request.build is None:  # unreachable past the request validator
            raise ValueError("record needs a build")
        kept = (
            *(b for b in records.builds if b.key != request.build.key),
            request.build,
        )
        path = write_records(
            request.state_dir, records.model_copy(update={"builds": kept})
        )
        return ModelDevLaneReturnToDevResult(
            operation="record", path=path, recorded=kept
        )
    return _settle(request, records)


def _settle(
    request: ModelDevLaneReturnToDevRequest, records: ModelDevLaneBranchBuilds
) -> ModelDevLaneReturnToDevResult:
    if request.now is None:  # unreachable past the request validator
        raise ValueError("settle needs the instant to judge")
    newest_job, declared = _declared_builds(request)
    builds = records.builds
    if declared is not None and declared.key not in {b.key for b in builds}:
        builds = (*builds, declared)
    due: list[ModelDevLaneReturnDue] = []
    dropped: list[ModelDevLaneReturnDue] = []
    kept: list[ModelDevLaneBranchBuild] = []
    for build in builds:
        gone = _gone(build, newest_job, request)
        if gone is not None:
            dropped.append(gone)
            continue
        kept.append(build)
        verdict = _judge(build, request)
        if verdict is not None:
            due.append(verdict)
    last = records.last_return_attempt
    free = not request.hold and (
        last is None or request.now - last >= RETURN_RETRY_AFTER
    )
    start = bool(due) and free
    spent = last is not None and not kept
    changed = bool(dropped) or declared is not None or start or spent
    path = record_path(request.state_dir)
    if changed:
        path = write_records(
            request.state_dir,
            ModelDevLaneBranchBuilds(
                builds=tuple(kept),
                last_return_attempt=request.now if start else (None if spent else last),
            ),
        )
    return ModelDevLaneReturnToDevResult(
        operation="settle",
        path=path,
        recorded=tuple(kept),
        due=tuple(due),
        dropped=tuple(dropped),
        command=_return_command() if start else None,
    )


def _return_command() -> dict[str, str]:
    """The one rebuild a return runs: the dev lane's runtime at the tracking ref."""
    return {
        "correlation_id": str(uuid4()),
        "requested_by": REQUESTER,
        "scope": "runtime",
        "runtime_lane": "dev",
        "build_source": "workspace",
    }


def _gone(
    build: ModelDevLaneBranchBuild,
    newest_job: datetime | None,
    request: ModelDevLaneReturnToDevRequest,
) -> ModelDevLaneReturnDue | None:
    """Why the lane no longer runs this build, or None when it still may."""
    if newest_job is not None and build.recorded_at < newest_job:
        return ModelDevLaneReturnDue(
            build=build,
            reason=EnumDevLaneReturnReason.SUPERSEDED,
            detail=f"a dev job completed {newest_job:%Y-%m-%dT%H:%M:%SZ}, after "
            f"the build of {build.key}",
        )
    clone = request.clones.get(build.repo)
    running = request.running_refs.get(build.repo, "")
    timeout = request.git_timeout_seconds
    if (
        clone is not None
        and running
        and _is_ancestor(clone, timeout, running, request.dev_ref)
    ):
        return ModelDevLaneReturnDue(
            build=build,
            reason=EnumDevLaneReturnReason.LANE_ON_DEV,
            detail=f"the lane runs {running[:12]}, which {request.dev_ref} carries",
        )
    return None


def _judge(
    build: ModelDevLaneBranchBuild, request: ModelDevLaneReturnToDevRequest
) -> ModelDevLaneReturnDue | None:
    clone = request.clones.get(build.repo)
    timeout = request.git_timeout_seconds
    if clone is not None:
        merge = _merge_commit(clone, timeout, request.dev_ref, build)
        if merge is not None:
            return ModelDevLaneReturnDue(
                build=build,
                reason=EnumDevLaneReturnReason.MERGED,
                detail=f"{build.repo}#{build.pr} merged as {merge[:12]} on "
                f"{request.dev_ref}; the lane runs the branch build {build.head[:12]}",
            )
        if build.branch and _branch_gone(clone, timeout, build.branch):
            return ModelDevLaneReturnDue(
                build=build,
                reason=EnumDevLaneReturnReason.CLOSED,
                detail=f"branch {build.branch} of {build.repo}#{build.pr} is gone "
                f"from origin and no merge is on {request.dev_ref}",
            )
    if request.now is not None and build.expires_at <= request.now:
        return ModelDevLaneReturnDue(
            build=build,
            reason=EnumDevLaneReturnReason.EXPIRED,
            detail=f"recorded {build.recorded_at:%Y-%m-%dT%H:%M:%SZ}, expired "
            f"{build.expires_at:%Y-%m-%dT%H:%M:%SZ} before {build.repo}#{build.pr} "
            "merged or closed",
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the one request named on the command line; the result is one stdout line."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("return-to-dev takes one request as JSON\n")
        return 2
    try:
        request = ModelDevLaneReturnToDevRequest.model_validate_json(args[0])
    except ValidationError as exc:
        sys.stderr.write(f"return-to-dev refused: {exc}\n")
        return 2
    sys.stdout.write(run(request).model_dump_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
