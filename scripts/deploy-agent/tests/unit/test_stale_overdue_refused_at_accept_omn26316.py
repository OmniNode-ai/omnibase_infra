# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An overdue dev build the lane already carries is refused at accept (OMN-26316).

THE RECORD
----------
2026-11-25 on h201: after the overdue check stopped publishing, about 18 older commands
requested by ``agent/overdue-dev-merge-build`` at git_ref ``f553b1d6`` stayed queued on the
control topic. Each declares a rollback of its own actor, so the lineage fence let it through
as ``rollback_declared`` and the agent accepted one about every 12 minutes (from 10:25:18Z),
rebuilding dev at an old ref with the old memory limit.

WHAT THESE TESTS PIN
--------------------
* The 18 queued commands give zero builds and 18 superseded records.
* An operator's rollback to the very same ref still builds.
* An overdue command for a merge the running build lacks still builds.
* A lineage the host cannot read (ancestry unanswered, running build unread) builds as before.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    EnumLineageVerdict,
    EnumRejectionReason,
    EnumRuntimeLane,
    ModelRebuildRequested,
)
from deploy_agent.job_state import JobState, JobStore
from deploy_agent.lineage_fence import ModelRunningBuild, decide_lineage

pytestmark = pytest.mark.unit

TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
TRACKING = "origin/dev"
OVERDUE = "agent/overdue-dev-merge-build"

#: One line of dev history, oldest first. ``STALE`` is f553b1d6, the ref the 18 commands name.
BEFORE = "1111111" + "0" * 33
STALE = "f553b1d6" + "8" * 32
BUILT = "a1b2c3d4" + "9" * 32
HEAD = "c0ffee01" + "7" * 32
_LINE = [BEFORE, STALE, BUILT, HEAD]
_REFS = {TRACKING: HEAD}


def contains(earlier: str, later: str) -> bool | None:
    earlier = _REFS.get(earlier, earlier)
    later = _REFS.get(later, later)
    if earlier not in _LINE or later not in _LINE:
        return None
    return _LINE.index(earlier) <= _LINE.index(later)


def _unanswered(earlier: str, later: str) -> bool | None:
    return None


def _command(
    git_ref: str = STALE,
    *,
    requested_by: str = OVERDUE,
    rollback_actor: str | None = OVERDUE,
) -> ModelRebuildRequested:
    payload: dict[str, Any] = {
        "correlation_id": uuid4(),
        "requested_by": requested_by,
        "scope": "runtime",
        "runtime_lane": "dev",
        "build_source": "workspace",
        "git_ref": git_ref,
    }
    if rollback_actor is not None:
        payload["rollback"] = {
            "actor": rollback_actor,
            "reason": "merge 3a527226 had no dev build by 2026-11-25T05:51:09+00:00",
        }
    return ModelRebuildRequested.model_validate(payload)


def _message(cmd: ModelRebuildRequested, offset: int) -> SimpleNamespace:
    return SimpleNamespace(
        value=cmd.model_dump(mode="json") | {"_signature": "a" * 64},
        topic=TOPIC,
        partition=0,
        offset=offset,
        key=None,
        timestamp=None,
    )


def _built_job(store: JobStore, ref: str) -> None:
    from datetime import UTC, datetime

    store._save(
        JobState(
            correlation_id=uuid4(),
            command={
                "scope": "runtime",
                "runtime_lane": "dev",
                "build_source": "workspace",
                "git_ref": ref,
            },
            accepted_at=datetime(2026, 11, 25, 10, 0, tzinfo=UTC),
            completed_at=datetime(2026, 11, 25, 10, 13, tzinfo=UTC),
            status="success",
        )
    )


def _consumer(
    store: JobStore,
    running: ModelRunningBuild | None,
    *,
    resolver: Any = contains,
) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = store
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    consumer.ancestry_resolver = resolver
    consumer.running_build = Mock(return_value=running)
    consumer.ref_resolver = _REFS.get
    consumer.tracking_ref = TRACKING
    consumer.notices = []
    consumer.on_rejected = consumer.notices.append
    return consumer


def _process(
    consumer: DeployConsumer, cmd: ModelRebuildRequested, offset: int, lookahead: Any
) -> tuple[ModelRebuildRequested | None, str | None]:
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        return consumer._process_message(_message(cmd, offset), lookahead=lookahead)


def _statuses(store: JobStore) -> list[str]:
    return sorted(job.status for job in store._jobs())


def test_the_18_queued_overdue_commands_give_zero_builds_and_18_superseded_records(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    _built_job(store, BUILT)
    consumer = _consumer(store, ModelRunningBuild(infra_ref=BUILT))
    queued = [_command() for _ in range(18)]
    messages = [_message(c, 100 + i) for i, c in enumerate(queued)]

    accepted: list[ModelRebuildRequested] = []
    reasons: list[str | None] = []
    for index, cmd in enumerate(queued):
        got, reason = _process(consumer, cmd, 100 + index, messages[index + 1 :])
        if got is not None:
            accepted.append(got)
        reasons.append(reason)

    assert accepted == []
    assert reasons == [EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD.value] * 18
    assert _statuses(store) == ["success"] + ["superseded"] * 18
    superseded = [j for j in store._jobs() if j.status == "superseded"]
    assert all(j.superseded_by_running_build for j in superseded)
    assert all(OVERDUE in " ".join(j.errors) or j.errors for j in superseded)
    assert all(
        j.lineage is not None
        and j.lineage.verdict is EnumLineageVerdict.CONTAINED
        and "overdue dev merge build" in j.lineage.detail
        for j in superseded
    )


def test_the_origin_dev_head_alone_is_enough_when_no_job_built_yet(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    consumer = _consumer(store, ModelRunningBuild(infra_ref=BUILT))

    accepted, reason = _process(consumer, _command(), 100, [])

    assert accepted is None
    assert reason == EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD.value


def test_an_operators_rollback_to_the_same_ref_still_builds(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    _built_job(store, BUILT)
    consumer = _consumer(store, ModelRunningBuild(infra_ref=BUILT))
    cmd = _command(requested_by="operator/lane", rollback_actor="operator/lane")

    accepted, reason = _process(consumer, cmd, 100, [])

    assert reason is None
    assert accepted is not None
    assert accepted.git_ref == STALE
    assert _statuses(store) == ["accepted", "success"]


def test_a_hourly_command_at_the_same_ref_is_not_the_overdue_rule(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    _built_job(store, BUILT)
    consumer = _consumer(store, ModelRunningBuild(infra_ref=BUILT))
    cmd = _command(requested_by="agent/hourly-dev-build", rollback_actor=None)

    accepted, _ = _process(consumer, cmd, 100, [])

    assert accepted is not None


def test_an_overdue_command_for_a_merge_the_running_build_lacks_still_builds(
    tmp_path: Path,
) -> None:
    """After a restore to an older image the merge is not in the running build."""
    store = JobStore(tmp_path)
    _built_job(store, BUILT)
    consumer = _consumer(store, ModelRunningBuild(infra_ref=BEFORE))

    accepted, reason = _process(consumer, _command(), 100, [])

    assert reason is None
    assert accepted is not None
    assert accepted.git_ref == STALE


def test_unanswered_ancestry_builds_as_before(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    _built_job(store, BUILT)
    consumer = _consumer(
        store, ModelRunningBuild(infra_ref=BUILT), resolver=_unanswered
    )

    accepted, reason = _process(consumer, _command(), 100, [])

    assert reason is None
    assert accepted is not None


def test_an_unread_running_build_builds_as_before(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    _built_job(store, BUILT)
    consumer = _consumer(store, None)

    accepted, reason = _process(consumer, _command(), 100, [])

    assert reason is None
    assert accepted is not None


def test_a_ref_ahead_of_every_reference_builds(tmp_path: Path) -> None:
    """A newer overdue command than anything built or on origin/dev's read is a new merge."""
    store = JobStore(tmp_path)
    _built_job(store, BUILT)
    consumer = _consumer(store, ModelRunningBuild(infra_ref=HEAD))
    consumer.ref_resolver = lambda ref: BUILT  # origin/dev read lags the command

    accepted, _ = _process(consumer, _command(HEAD), 100, [])

    assert accepted is not None


def test_the_decision_is_the_fences_and_needs_both_proofs() -> None:
    cmd = _command()

    def decide(running: ModelRunningBuild | None, refs: list[str]) -> Any:
        return decide_lineage(
            cmd,
            read_running_build=lambda: running,
            contains=contains,
            resolve_ref=None,
            tracking_ref=TRACKING,
            published_at=None,
            overdue_references=lambda: refs,
        )

    carried = decide(ModelRunningBuild(infra_ref=BUILT), [BUILT])
    assert carried.verdict is EnumLineageVerdict.CONTAINED
    assert carried.verdict.supersedes
    assert (
        decide(ModelRunningBuild(infra_ref=BEFORE), [BUILT]).verdict
        is EnumLineageVerdict.ROLLBACK_DECLARED
    )
    assert (
        decide(ModelRunningBuild(infra_ref=BUILT), []).verdict
        is EnumLineageVerdict.ROLLBACK_DECLARED
    )
