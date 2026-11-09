# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev lane built from an unmerged branch is due to go back to dev (OMN-22996).

Every case runs the handler over real git: a bare ``origin``, the clone the deploy
agent keeps of it and the pull request's branch, replaying 2026-11-07, when the lane
kept the build of omnibase_infra#5755's branch after #5755 merged. The agent's job
records are real files in its jobs directory, written in the shape ``JobStore`` writes
them. Nothing here touches Docker, a broker or the deploy agent.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.enums.enum_dev_lane_return_reason import EnumDevLaneReturnReason
from omnibase_infra.handlers import handler_dev_lane_return_to_dev as handler_module
from omnibase_infra.handlers.handler_dev_lane_return_to_dev import (
    REQUESTER,
    HandlerDevLaneReturnToDev,
    main,
    read_builds,
    record_path,
)
from omnibase_infra.models.model_dev_lane_branch_build import ModelDevLaneBranchBuild
from omnibase_infra.models.model_dev_lane_return_to_dev_request import (
    ModelDevLaneReturnToDevRequest,
)
from omnibase_infra.models.model_dev_lane_return_to_dev_result import (
    ModelDevLaneReturnToDevResult,
)

pytestmark = pytest.mark.unit

REPO = "omnibase_infra"
PR = 5755
BRANCH = "omn-22808-estimator-follows-web"
NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
JOB_DONE = NOW - timedelta(hours=1)
DEV_REF = "origin/dev"
GIT_TIMEOUT = 60


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    return done.stdout.strip()


def _commit(cwd: Path, name: str, message: str) -> str:
    (cwd / name).write_text(message, encoding="utf-8")
    _git(cwd, "add", name)
    _git(cwd, "commit", "-m", message)
    return _git(cwd, "rev-parse", "HEAD")


class _World:
    """A bare origin, the agent's clone of it and a lab host's worktree for pushes."""

    def __init__(self, root: Path) -> None:
        root.mkdir()
        self.origin = root / "origin.git"
        self.work = root / "work"
        self.clone = root / "clone"
        self.jobs = root / "state"
        self.state = self.jobs / "branch-builds"
        _git(root, "init", "--bare", "-b", "dev", str(self.origin))
        _git(root, "clone", str(self.origin), str(self.work))
        _git(self.work, "checkout", "-b", "dev")
        self.base = _commit(self.work, "a.txt", "base")
        _git(self.work, "push", "origin", "dev")
        _git(self.work, "checkout", "-b", BRANCH)
        self.head = _commit(self.work, "b.txt", "estimator follows the web repository")
        _git(self.work, "push", "origin", BRANCH)
        _git(root, "clone", str(self.origin), str(self.clone))
        self.jobs.mkdir()

    def refresh(self) -> None:
        _git(self.clone, "fetch", "--prune", "origin")

    def squash_merge(self) -> str:
        _git(self.work, "checkout", "dev")
        merge = _commit(
            self.work,
            "b.txt",
            f"fix(OMN-22808): the estimator follows the web repository (#{PR})",
        )
        _git(self.work, "push", "origin", "dev")
        self.refresh()
        return merge

    def delete_branch(self) -> None:
        _git(self.work, "push", "origin", "--delete", BRANCH)
        self.refresh()

    def job(
        self,
        *,
        open_pr: str = "",
        done: datetime = JOB_DONE,
        status: str = "success",
        lane: str = "dev",
        requested_by: str = "estimator-deploy-2340",
    ) -> str:
        """Write one job record in the shape ``JobStore`` writes."""
        correlation_id = str(uuid4())
        command = {
            "correlation_id": correlation_id,
            "requested_by": requested_by,
            "scope": "runtime",
            "runtime_lane": lane,
            "open_pr": open_pr,
        }
        record = {
            "correlation_id": correlation_id,
            "command": command,
            "status": status,
            "completed_at": done.isoformat(),
        }
        (self.jobs / f"{correlation_id}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        return correlation_id

    @property
    def declared(self) -> str:
        return f"{REPO}#{PR}@{self.head}"


@pytest.fixture
def world(tmp_path: Path) -> _World:
    return _World(tmp_path / "world")


def _build(world: _World, **override: object) -> ModelDevLaneBranchBuild:
    values: dict[str, object] = {
        "repo": REPO,
        "pr": PR,
        "branch": BRANCH,
        "head": world.head,
        "requested_by": "estimator-deploy-2340",
        "job_id": "8dd5eb7f",
        "recorded_at": NOW - timedelta(hours=1),
        "expires_at": NOW + timedelta(days=2),
    }
    values.update(override)
    return ModelDevLaneBranchBuild.model_validate(values)


def _record(world: _World, build: ModelDevLaneBranchBuild) -> None:
    HandlerDevLaneReturnToDev().handle(
        ModelDevLaneReturnToDevRequest(
            operation="record",
            state_dir=world.state,
            git_timeout_seconds=GIT_TIMEOUT,
            build=build,
        )
    )


def _settle(
    world: _World,
    *,
    running: str | None,
    now: datetime = NOW,
    clone: bool = True,
    hold: str = "",
) -> ModelDevLaneReturnToDevResult:
    return HandlerDevLaneReturnToDev().handle(
        ModelDevLaneReturnToDevRequest(
            operation="settle",
            state_dir=world.state,
            git_timeout_seconds=GIT_TIMEOUT,
            jobs_dir=world.jobs,
            now=now,
            clones={REPO: world.clone} if clone else {},
            running_refs={REPO: running} if running else {},
            dev_ref=DEV_REF,
            hold=hold,
        )
    )


class TestRecord:
    def test_a_recorded_build_names_the_branch_and_its_pull_request(
        self, world: _World
    ) -> None:
        _record(world, _build(world))

        (stored,) = read_builds(world.state)
        assert (stored.repo, stored.pr, stored.branch, stored.head) == (
            REPO,
            PR,
            BRANCH,
            world.head,
        )
        assert record_path(world.state).is_file()

    def test_recording_the_same_build_again_keeps_one_record(
        self, world: _World
    ) -> None:
        _record(world, _build(world))
        _record(world, _build(world, job_id="second"))

        (stored,) = read_builds(world.state)
        assert stored.job_id == "second"

    @pytest.mark.parametrize(
        "override",
        [
            {"head": "abc123"},
            {"pr": 0},
            {"repo": "Omnibase-Infra"},
            {"expires_at": NOW + timedelta(days=9)},
            {"expires_at": NOW - timedelta(days=1)},
            {"recorded_at": datetime(2026, 11, 7, 11, 0)},
        ],
    )
    def test_a_malformed_build_is_refused(
        self, world: _World, override: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError):
            _build(world, **override)

    def test_an_unreadable_record_is_ignored_not_raised(self, world: _World) -> None:
        world.state.mkdir(parents=True)
        record_path(world.state).write_text("{not json", encoding="utf-8")

        assert read_builds(world.state) == ()


class TestJobRecords:
    """The build is read from the agent's own job records: nothing else declares it."""

    def test_the_newest_dev_job_that_declared_a_pull_request_is_recorded(
        self, world: _World
    ) -> None:
        world.job(open_pr=world.declared)

        result = _settle(world, running=world.head)

        (stored,) = read_builds(world.state)
        assert (stored.repo, stored.pr, stored.head) == (REPO, PR, world.head)
        assert stored.branch == BRANCH, "the branch is read from the clone"
        assert stored.requested_by == "estimator-deploy-2340"
        assert result.due == () and result.command is None

    def test_a_job_that_declared_nothing_records_nothing(self, world: _World) -> None:
        world.job()

        assert _settle(world, running=world.head).recorded == ()

    @pytest.mark.parametrize(
        "kwargs",
        [{"status": "failed"}, {"status": "in_progress"}, {"lane": "prod"}],
    )
    def test_a_failed_or_other_lane_job_records_nothing(
        self, world: _World, kwargs: dict[str, str]
    ) -> None:
        world.job(open_pr=world.declared, **kwargs)

        assert _settle(world, running=world.head).recorded == ()

    def test_a_later_dev_job_replaces_the_build(self, world: _World) -> None:
        world.job(open_pr=world.declared)
        _settle(world, running=world.head)
        world.job(done=JOB_DONE + timedelta(minutes=5), requested_by="agent/idle")

        result = _settle(world, running=world.head)

        (dropped,) = result.dropped
        assert dropped.reason is EnumDevLaneReturnReason.SUPERSEDED
        assert read_builds(world.state) == ()

    def test_unreadable_job_files_are_skipped(self, world: _World) -> None:
        (world.jobs / "garbage.json").write_text("{nope", encoding="utf-8")
        world.job(open_pr=world.declared)

        assert len(_settle(world, running=world.head).recorded) == 1


class TestSettle:
    def test_an_open_pull_request_keeps_the_lane_where_it_is(
        self, world: _World
    ) -> None:
        _record(world, _build(world))

        result = _settle(world, running=world.head)

        assert result.due == () and result.command is None
        assert len(result.recorded) == 1

    def test_a_squash_merged_pull_request_is_due_and_names_the_merge(
        self, world: _World
    ) -> None:
        """Replay of 2026-11-07: #5755 merged and the lane still ran 29dd9aa2c765."""
        world.job(open_pr=world.declared)
        merge = world.squash_merge()

        result = _settle(world, running=world.head)

        (due,) = result.due
        assert due.reason is EnumDevLaneReturnReason.MERGED
        assert merge[:12] in due.detail
        assert world.head[:12] in due.detail
        assert result.command is not None
        assert result.command["requested_by"] == REQUESTER
        assert result.command["runtime_lane"] == "dev"
        assert result.command["scope"] == "runtime"

    def test_without_a_declared_build_a_merged_branch_lane_is_never_returned(
        self, world: _World
    ) -> None:
        """The failing control: before the record existed nothing could name the lane."""
        world.job()
        world.squash_merge()

        result = _settle(world, running=world.head)

        assert result.due == ()
        assert result.recorded == ()
        assert result.command is None

    def test_a_merge_commit_puts_the_head_on_dev_so_nothing_is_left_to_return(
        self, world: _World
    ) -> None:
        _record(world, _build(world))
        _git(world.work, "checkout", "dev")
        _git(world.work, "merge", "--no-ff", BRANCH, "-m", "Merge the branch")
        _git(world.work, "push", "origin", "dev")
        world.refresh()

        result = _settle(world, running=world.head)

        assert result.due == ()
        (dropped,) = result.dropped
        assert dropped.reason is EnumDevLaneReturnReason.LANE_ON_DEV

    def test_a_closed_pull_request_with_its_branch_gone_is_due(
        self, world: _World
    ) -> None:
        _record(world, _build(world))
        world.delete_branch()

        (due,) = _settle(world, running=world.head).due

        assert due.reason is EnumDevLaneReturnReason.CLOSED
        assert BRANCH in due.detail

    def test_an_unreachable_origin_is_not_read_as_a_closed_pull_request(
        self, world: _World, tmp_path: Path
    ) -> None:
        _record(world, _build(world))
        _git(world.clone, "remote", "set-url", "origin", str(tmp_path / "nowhere.git"))

        assert _settle(world, running=world.head).due == ()

    def test_an_expired_record_is_due_without_any_clone(self, world: _World) -> None:
        _record(world, _build(world))

        result = _settle(
            world, running=world.head, now=NOW + timedelta(days=3), clone=False
        )

        (due,) = result.due
        assert due.reason is EnumDevLaneReturnReason.EXPIRED
        assert result.command is not None

    def test_with_no_clone_and_no_expiry_nothing_is_read_so_nothing_is_due(
        self, world: _World
    ) -> None:
        _record(world, _build(world))
        world.squash_merge()

        assert _settle(world, running=world.head, clone=False).due == ()

    def test_a_lane_already_on_dev_is_dropped_not_returned(self, world: _World) -> None:
        """Dev's own rebuild after the merge already put the lane back."""
        _record(world, _build(world))
        merge = world.squash_merge()

        result = _settle(world, running=merge)

        assert result.due == ()
        (dropped,) = result.dropped
        assert dropped.reason is EnumDevLaneReturnReason.LANE_ON_DEV
        assert read_builds(world.state) == ()

    def test_an_unknown_running_commit_still_returns_a_merged_lane(
        self, world: _World
    ) -> None:
        _record(world, _build(world))
        world.squash_merge()

        (due,) = _settle(world, running=None).due

        assert due.reason is EnumDevLaneReturnReason.MERGED

    def test_a_git_that_cannot_run_keeps_the_record(
        self, world: _World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _record(world, _build(world))
        world.squash_merge()

        def hang(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(argv, GIT_TIMEOUT)

        monkeypatch.setattr(handler_module.subprocess, "run", hang)

        result = _settle(world, running=world.head)

        assert result.due == ()
        assert len(read_builds(world.state)) == 1


class TestStartingTheReturn:
    def _merged(self, world: _World) -> None:
        world.job(open_pr=world.declared)
        world.squash_merge()

    def test_a_hold_keeps_the_command_back_and_the_build_due(
        self, world: _World
    ) -> None:
        self._merged(world)

        held = _settle(world, running=world.head, hold="a job is in flight")

        assert len(held.due) == 1 and held.command is None
        assert _settle(world, running=world.head).command is not None

    def test_a_return_is_not_started_twice_inside_the_hour(self, world: _World) -> None:
        self._merged(world)

        first = _settle(world, running=world.head)
        again = _settle(world, running=world.head, now=NOW + timedelta(minutes=30))
        later = _settle(world, running=world.head, now=NOW + timedelta(minutes=61))

        assert first.command is not None
        assert len(again.due) == 1 and again.command is None
        assert later.command is not None

    def test_the_return_job_drops_the_record_and_the_hour_with_it(
        self, world: _World
    ) -> None:
        self._merged(world)
        _settle(world, running=world.head)
        world.job(done=NOW + timedelta(minutes=2), requested_by=REQUESTER)

        result = _settle(world, running=world.head, now=NOW + timedelta(minutes=3))

        assert result.due == () and result.command is None
        assert read_builds(world.state) == ()
        records = json.loads(record_path(world.state).read_text(encoding="utf-8"))
        assert records["last_return_attempt"] is None


class TestCommandLine:
    def test_the_request_on_the_command_line_runs_and_prints_the_result(
        self, world: _World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _record(world, _build(world))
        request = ModelDevLaneReturnToDevRequest(
            operation="settle",
            state_dir=world.state,
            git_timeout_seconds=GIT_TIMEOUT,
            jobs_dir=world.jobs,
            now=NOW + timedelta(days=3),
            dev_ref=DEV_REF,
        )

        assert main([request.model_dump_json()]) == 0

        printed = json.loads(capsys.readouterr().out)
        assert printed["due"][0]["reason"] == "expired"
        assert printed["command"]["requested_by"] == REQUESTER

    @pytest.mark.parametrize("argv", [[], ["{}"], ["not json"], ["a", "b"]])
    def test_a_request_the_node_cannot_run_is_refused(
        self, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(argv) == 2
        assert "return-to-dev" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "fields",
        [{"operation": "record"}, {"operation": "settle"}],
    )
    def test_a_request_missing_what_its_operation_needs_is_refused(
        self, tmp_path: Path, fields: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError):
            ModelDevLaneReturnToDevRequest.model_validate(
                {**fields, "state_dir": tmp_path, "git_timeout_seconds": 5}
            )
