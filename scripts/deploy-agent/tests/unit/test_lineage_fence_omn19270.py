# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent never rebuilds what runs, nor rebuilds backwards (OMN-19270).

THE INCIDENT
------------
On 2026-09-23 the .201 dev lane was running infra ``0edf5c914`` from job
``c009462c`` (lab-health triage, ``scope=runtime``, ``git_ref=origin/dev``,
accepted 13:50:25Z). Seven full rebuilds followed. Each was published by a CI
trigger between 10:59Z and 13:52Z and accepted hours later:

* ``ab27aedd`` at ``c159b7118`` rebuilt the lane BACKWARDS at 14:46:42Z.
* ``e074126b`` at ``533b19c2`` was behind ``0edf5c914`` too.
* ``602b1d61``, ``5ce1f33e``, ``f645854b`` and ``98bb76df`` were all at
  ``0edf5c914``, each re-delivering an omnimarket merge. ``98bb76df`` failed
  post-deploy verification and left the runtime container in ``Created``.
* ``cb5275c0`` at infra ``6fe05c35`` was the one new infra deploy.

WHAT THESE TESTS PIN
--------------------
* The measured sequence with the fence active throughout: monotonic, five
  commands superseded, ``98bb76df`` built because pr-2808 merged after
  ``c009462c`` staged, and ``cb5275c0`` built.
* The same commands against the REAL history of running builds: after the
  backward roll, ``602b1d61`` builds (a past success is not the running build),
  and ``5ce1f33e``, ``f645854b`` and ``98bb76df`` are superseded by the running
  workspace build that started after each was requested.
* The mixed-history case: after a backward roll, an equal-ref command
  requested after the running build started is NOT superseded.
* An equal ref, a strict ancestor, a diverged ref, the rollback override, a
  symbolic ref, and missing provenance, each at decision and consumer level.
* The running build's producing job is found by its ``build_time``, the
  publish time falls back to the broker timestamp, and a supersession is a
  durable ``superseded`` record plus a published event.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest
from deploy_agent.coalesce import (
    EnumCoalesceRefusal,
    GitAncestryResolver,
    ModelQueuedCommand,
    plan_coalesce,
)
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    EnumLineageVerdict,
    EnumRejectionReason,
    EnumRuntimeLane,
    ModelLineageDecision,
    ModelRebuildRequested,
    ModelRejectionNotice,
    ModelRollbackDeclaration,
)
from deploy_agent.job_state import JobState, JobStore
from deploy_agent.lineage_fence import (
    BUILD_PROVENANCE_PATH,
    DockerProvenanceReader,
    GitRefResolver,
    ModelRunningBuild,
    ci_source_repository,
    decide_lineage,
    is_workspace_rebuild,
)

TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
TRACKING = "origin/dev"

# omnibase_infra dev, first parent, 2026-09-23.
INFRA_C159 = "c159b711808d1eca01accabe8e3b1f3db3f3cdf9"  # 10:32:14Z
INFRA_533B = "533b19c23b630067d8b53855ce1bd96392aa4412"  # 12:09:47Z
INFRA_0EDF = "0edf5c9145876dbe22f6caf7a06b507c5d0fc7d3"  # 13:00:08Z
INFRA_6FE0 = "6fe05c3567d8fb546a821f50ae891869e9494da1"  # 14:05:15Z
#: A commit off the tracking branch, branched from INFRA_533B.
FEATURE = "f" * 40

_INFRA_LINE = [INFRA_C159, INFRA_533B, INFRA_0EDF, INFRA_6FE0]
#: origin/dev resolves to the infra tip.
_REFS = {TRACKING: INFRA_6FE0}


def contains(earlier: str, later: str) -> bool | None:
    """``git merge-base --is-ancestor earlier later`` over the graph above."""
    earlier = _REFS.get(earlier, earlier)
    later = _REFS.get(later, later)
    if earlier == later:
        return True
    if FEATURE in (earlier, later):
        if later == FEATURE:
            return earlier in _INFRA_LINE[: _INFRA_LINE.index(INFRA_533B) + 1]
        return False
    if earlier not in _INFRA_LINE or later not in _INFRA_LINE:
        return None
    return _INFRA_LINE.index(earlier) <= _INFRA_LINE.index(later)


def at(hms: str) -> datetime:
    """A 2026-09-23 UTC moment, ``HH:MM:SS``."""
    return datetime.fromisoformat(f"2026-09-23T{hms}+00:00")


def _running(
    infra: str,
    started: str | None = "13:50:25",
    *,
    workspace: bool = True,
    job: str = "c009462c",
) -> ModelRunningBuild:
    return ModelRunningBuild(
        infra_ref=infra,
        started_at=at(started) if started else None,
        workspace_sourced=workspace,
        producing_job=job,
    )


def _command(
    git_ref: str,
    *,
    requested_by: str = "gha/omnibase_infra/pr-4002",
    requested_at: datetime | None = None,
    rollback: ModelRollbackDeclaration | None = None,
    correlation_id: str | None = None,
    **overrides: Any,
) -> ModelRebuildRequested:
    fields: dict[str, Any] = {
        "correlation_id": correlation_id or str(uuid4()),
        "requested_by": requested_by,
        "scope": "full",
        "runtime_lane": "dev",
        "build_source": "workspace",
        "git_ref": git_ref,
        "rollback": rollback,
        "requested_at": requested_at,
    }
    fields.update(overrides)
    return ModelRebuildRequested.model_validate(fields)


def _omnimarket(pr: int, git_ref: str, requested: str) -> ModelRebuildRequested:
    return _command(
        git_ref,
        requested_by=f"gha/omnimarket/pr-{pr}",
        requested_at=at(requested),
    )


def _decide(
    cmd: ModelRebuildRequested,
    running: ModelRunningBuild | None,
    *,
    resolve_ref: Any = None,
    published_at: datetime | None = None,
) -> ModelLineageDecision:
    return decide_lineage(
        cmd,
        read_running_build=lambda: running,
        contains=contains,
        resolve_ref=resolve_ref,
        tracking_ref=TRACKING,
        published_at=published_at or cmd.requested_at,
    )


# ---------------------------------------------------------------------------
# The measured sequence.
# ---------------------------------------------------------------------------

#: (job, command, accepted_at) in the order the agent accepted them. The
#: requested_at of each is its CI trigger's redeploy-start publish time, read
#: from the trigger run logs; the omnimarket PR that produced each is named.
_SEQUENCE: list[tuple[str, ModelRebuildRequested, str]] = [
    ("ab27aedd", _omnimarket(2802, INFRA_C159, "10:59:36"), "14:32:21"),
    ("e074126b", _omnimarket(2804, INFRA_533B, "12:31:36"), "15:50:24"),
    ("602b1d61", _omnimarket(2806, INFRA_0EDF, "13:38:08"), "17:02:18"),
    ("5ce1f33e", _omnimarket(2805, INFRA_0EDF, "13:38:09"), "17:24:57"),
    ("f645854b", _omnimarket(2801, INFRA_0EDF, "13:39:50"), "17:45:21"),
    ("98bb76df", _omnimarket(2808, INFRA_0EDF, "13:52:44"), "18:39:01"),
    (
        "cb5275c0",
        _command(INFRA_6FE0, requested_at=at("19:18:00")),
        "19:20:38",
    ),
]

#: The running build that c009462c left: origin/dev resolved to 0edf5c914,
#: a workspace ``scope=runtime`` job accepted at 13:50:25Z, which staged
#: omnimarket BEFORE pr-2808 merged at 13:50:56Z.
_C009 = _running(INFRA_0EDF, "13:50:25", job="c009462c")


@pytest.mark.unit
def test_with_the_fence_active_throughout_five_are_superseded_and_none_go_back() -> (
    None
):
    running = _C009
    outcomes = []
    for job, cmd, accepted in _SEQUENCE:
        decision = _decide(cmd, running)
        assert not decision.verdict.refuses, (job, decision.journal_line())
        if not decision.verdict.supersedes:
            assert contains(running.infra_ref, decision.build_ref), (
                f"{job} would build {decision.build_ref}, behind the running "
                f"{running.infra_ref}"
            )
            running = _running(decision.build_ref, accepted, job=job)
        outcomes.append((job, decision.verdict, decision.build_ref))

    assert outcomes == [
        ("ab27aedd", EnumLineageVerdict.CONTAINED, INFRA_0EDF),
        ("e074126b", EnumLineageVerdict.CONTAINED, INFRA_0EDF),
        ("602b1d61", EnumLineageVerdict.CONTAINED, INFRA_0EDF),
        ("5ce1f33e", EnumLineageVerdict.CONTAINED, INFRA_0EDF),
        ("f645854b", EnumLineageVerdict.CONTAINED, INFRA_0EDF),
        # pr-2808 merged after c009462c staged, and nothing since has built it.
        ("98bb76df", EnumLineageVerdict.EQUAL, INFRA_0EDF),
        ("cb5275c0", EnumLineageVerdict.DESCENDANT, INFRA_6FE0),
    ]


#: The build each command in _SEQUENCE actually found running when it was
#: accepted, from the image tags and job records.
_REAL_RUNNING = {
    "ab27aedd": _C009,
    "e074126b": _running(INFRA_C159, "14:32:21", job="ab27aedd"),
    "602b1d61": _running(INFRA_533B, "15:50:24", job="e074126b"),
    "5ce1f33e": _running(INFRA_0EDF, "17:02:18", job="602b1d61"),
    "f645854b": _running(INFRA_0EDF, "17:24:57", job="5ce1f33e"),
    "98bb76df": _running(INFRA_0EDF, "17:45:21", job="f645854b"),
    "cb5275c0": _running(INFRA_0EDF, "18:39:01", job="98bb76df"),
}


@pytest.mark.unit
def test_against_the_real_history_the_fence_follows_the_running_build() -> None:
    """A past success is not the running build: after the roll, 602b1d61 builds."""
    verdicts = {
        job: _decide(cmd, _REAL_RUNNING[job]).verdict for job, cmd, _ in _SEQUENCE
    }

    assert verdicts == {
        "ab27aedd": EnumLineageVerdict.CONTAINED,
        "e074126b": EnumLineageVerdict.DESCENDANT,
        "602b1d61": EnumLineageVerdict.DESCENDANT,
        "5ce1f33e": EnumLineageVerdict.CONTAINED,
        "f645854b": EnumLineageVerdict.CONTAINED,
        "98bb76df": EnumLineageVerdict.CONTAINED,
        "cb5275c0": EnumLineageVerdict.DESCENDANT,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("requested", "verdict"),
    [
        # Requested after the rolled-back build started: that build may not
        # carry its sibling merge, so it builds.
        ("14:40:00", EnumLineageVerdict.EQUAL),
        # Requested before it started: the build staged the merge.
        ("14:00:00", EnumLineageVerdict.CONTAINED),
    ],
)
def test_after_a_backward_roll_an_equal_ref_is_judged_by_the_running_start(
    requested: str, verdict: EnumLineageVerdict
) -> None:
    after_the_roll = _running(INFRA_C159, "14:32:21", job="ab27aedd")

    decision = _decide(_omnimarket(2999, INFRA_C159, requested), after_the_roll)

    assert decision.verdict is verdict


# ---------------------------------------------------------------------------
# An equal ref.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_an_infra_merge_at_the_running_ref_is_superseded_on_containment() -> None:
    decision = _decide(_command(INFRA_0EDF, requested_at=at("23:00:00")), _C009)

    assert decision.verdict is EnumLineageVerdict.CONTAINED


@pytest.mark.unit
@pytest.mark.parametrize(
    ("running", "fragment"),
    [
        (_running(INFRA_0EDF, None), "could not be identified"),
        (_running(INFRA_0EDF, workspace=False), "did not rebuild the whole"),
    ],
)
def test_a_sibling_merge_is_not_superseded_without_a_known_workspace_start(
    running: ModelRunningBuild, fragment: str
) -> None:
    decision = _decide(_omnimarket(2805, INFRA_0EDF, "13:38:09"), running)

    assert decision.verdict is EnumLineageVerdict.EQUAL
    assert fragment in decision.detail


@pytest.mark.unit
def test_a_sibling_merge_with_no_publish_time_builds() -> None:
    cmd = _command(INFRA_0EDF, requested_by="gha/omnimarket/pr-2805")

    decision = _decide(cmd, _C009)

    assert decision.verdict is EnumLineageVerdict.EQUAL
    assert "no publish time" in decision.detail


@pytest.mark.unit
@pytest.mark.parametrize("requested_by", ["operator-manual", "lab-health-triage"])
def test_a_deliberate_request_at_the_running_ref_is_never_superseded(
    requested_by: str,
) -> None:
    """A same-ref rebuild is how a wedged lane is recovered."""
    cmd = _command(INFRA_0EDF, requested_by=requested_by, requested_at=at("09:00:00"))

    decision = _decide(cmd, _C009)

    assert decision.verdict is EnumLineageVerdict.EQUAL
    assert "never superseded" in decision.detail


# ---------------------------------------------------------------------------
# A strict ancestor, a descendant, a diverged ref.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_strict_ancestor_that_must_build_is_raised_to_the_running_ref() -> None:
    cmd = _command(INFRA_C159, requested_by="operator-manual")

    decision = _decide(cmd, _C009)

    assert decision.verdict is EnumLineageVerdict.RAISED
    assert decision.build_ref == INFRA_0EDF


@pytest.mark.unit
def test_a_descendant_builds_its_own_ref() -> None:
    decision = _decide(_command(INFRA_6FE0), _C009)

    assert decision.verdict is EnumLineageVerdict.DESCENDANT
    assert decision.build_ref == INFRA_6FE0


@pytest.mark.unit
def test_a_diverged_ref_off_the_tracking_branch_is_refused() -> None:
    assert _decide(_command(FEATURE), _C009).verdict is EnumLineageVerdict.DIVERGENT


@pytest.mark.unit
def test_a_lane_on_an_off_branch_build_returns_to_its_tracking_branch() -> None:
    """A lane left on a PR-head proof build must still take the next dev deploy."""
    decision = _decide(_command(INFRA_6FE0), _running(FEATURE))

    assert decision.verdict is EnumLineageVerdict.RETURNS_TO_TRACKING
    assert decision.build_ref == INFRA_6FE0


# ---------------------------------------------------------------------------
# The rollback override.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("target", [INFRA_C159, FEATURE])
def test_a_declared_rollback_builds_exactly_what_it_asked_for(target: str) -> None:
    reads: list[int] = []
    cmd = _command(
        target,
        rollback=ModelRollbackDeclaration(
            actor="operator", reason="0edf5c914 broke the projection writers"
        ),
    )

    decision = decide_lineage(
        cmd,
        read_running_build=lambda: reads.append(1) or _C009,  # type: ignore[func-returns-value]
        contains=contains,
        resolve_ref=None,
        tracking_ref=TRACKING,
        published_at=None,
    )

    assert decision.verdict is EnumLineageVerdict.ROLLBACK_DECLARED
    assert decision.build_ref == target
    assert reads == []


@pytest.mark.unit
@pytest.mark.parametrize("field", ["actor", "reason"])
def test_a_rollback_must_name_its_actor_and_reason(field: str) -> None:
    fields = {"actor": "operator", "reason": "bad build"} | {field: "   "}
    with pytest.raises(ValueError, match=field):
        ModelRollbackDeclaration(**fields)


@pytest.mark.unit
@pytest.mark.parametrize("rollback_first", [True, False])
def test_coalescing_never_folds_a_rollback(rollback_first: bool) -> None:
    rollback = ModelRollbackDeclaration(actor="operator", reason="bad build")
    plain = _command(INFRA_0EDF)
    rolled = _command(INFRA_533B, rollback=rollback)
    head, candidate = (rolled, plain) if rollback_first else (plain, rolled)

    plan = plan_coalesce(
        [
            ModelQueuedCommand(command=head, partition=0, offset=1),
            ModelQueuedCommand(command=candidate, partition=0, offset=2),
        ],
        contains=lambda earlier, later: True,
    )

    assert plan.superseded == ()
    assert plan.stop_reason is EnumCoalesceRefusal.ROLLBACK_DECLARED
    assert plan.runner.command == head


# ---------------------------------------------------------------------------
# A symbolic ref, and missing provenance.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_symbolic_ref_is_resolved_at_accept_and_built_at_that_sha() -> None:
    """``c009462c`` asked for origin/dev and its record never said which commit."""
    cmd = _command(TRACKING, requested_by="lab-health-triage")

    decision = _decide(cmd, _C009, resolve_ref=_REFS.get)

    assert decision.verdict is EnumLineageVerdict.DESCENDANT
    assert decision.resolved_ref == INFRA_6FE0
    assert decision.build_ref == INFRA_6FE0


@pytest.mark.unit
def test_an_unresolvable_symbolic_ref_builds_as_requested() -> None:
    decision = _decide(_command(TRACKING), _C009, resolve_ref=lambda ref: None)

    assert decision.verdict is EnumLineageVerdict.UNPROVEN
    assert decision.build_ref == TRACKING


@pytest.mark.unit
def test_an_unreadable_running_build_builds_as_requested() -> None:
    """A lane that is down has no container to read, and must accept a rebuild."""
    decision = _decide(_command(INFRA_C159), None)

    assert decision.verdict is EnumLineageVerdict.UNPROVEN
    assert decision.build_ref == INFRA_C159


@pytest.mark.unit
def test_an_unanswerable_ancestry_builds_as_requested() -> None:
    decision = decide_lineage(
        _command(INFRA_C159),
        read_running_build=lambda: _C009,
        contains=lambda earlier, later: None,
        resolve_ref=None,
        tracking_ref=TRACKING,
        published_at=None,
    )

    assert decision.verdict is EnumLineageVerdict.UNPROVEN


@pytest.mark.unit
def test_a_pinned_image_is_not_the_fence_s() -> None:
    cmd = _command(INFRA_C159, image_digest="sha256:" + "0" * 64)

    assert _decide(cmd, _C009).verdict is EnumLineageVerdict.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# The consumer: the producing job, the publish-time fallback, supersession.
# ---------------------------------------------------------------------------


def _message(
    cmd: ModelRebuildRequested, offset: int = 100, timestamp_ms: int | None = None
) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json") | {"_signature": "a" * 64}
    return SimpleNamespace(
        value=payload,
        topic=TOPIC,
        partition=0,
        offset=offset,
        key=None,
        timestamp=timestamp_ms,
    )


def _job(
    store: JobStore,
    *,
    accepted: str,
    completed: str | None,
    git_ref: str = INFRA_0EDF,
    scope: str = "full",
    status: str = "success",
) -> JobState:
    job = JobState(
        correlation_id=uuid4(),
        command={
            "scope": scope,
            "services": [],
            "build_source": "workspace",
            "git_ref": git_ref,
            "image_ref": None,
            "image_digest": None,
        },
        accepted_at=at(accepted),
        completed_at=at(completed) if completed else None,
        status=status,  # type: ignore[arg-type]
    )
    store._save(job)
    return job


def _consumer(
    running: ModelRunningBuild | None,
    store: JobStore | None = None,
    *,
    duplicate: bool = False,
) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock(wraps=store) if store else Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = duplicate
    if store is None:
        consumer.job_store.job_covering.return_value = None
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    consumer.ancestry_resolver = contains
    consumer.running_build = Mock(return_value=running)
    consumer.ref_resolver = None
    consumer.tracking_ref = TRACKING
    consumer.notices = []
    consumer.on_rejected = consumer.notices.append
    return consumer


def _process(
    consumer: DeployConsumer, msg: SimpleNamespace, lookahead: Any = None
) -> tuple[ModelRebuildRequested | None, str | None]:
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        return consumer._process_message(msg, lookahead=lookahead)


def _committed(consumer: DeployConsumer) -> list[int]:
    return [
        meta.offset
        for call in consumer.consumer.commit.call_args_list
        for meta in call.args[0].values()
    ]


@pytest.mark.unit
def test_the_consumer_names_the_job_that_built_the_running_image(
    tmp_path: Path,
) -> None:
    """98bb76df found f645854b's image running, and was superseded by it."""
    store = JobStore(tmp_path)
    _job(store, accepted="17:24:57", completed="17:39:08")
    producer = _job(store, accepted="17:45:21", completed="18:06:16")
    running = ModelRunningBuild(infra_ref=INFRA_0EDF, build_time=at("17:51:29"))
    consumer = _consumer(running, store)
    cmd = _omnimarket(2808, INFRA_0EDF, "13:52:44")

    accepted, reason = _process(consumer, _message(cmd))

    assert accepted is None
    assert reason == EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD.value
    record = consumer.job_store.record_superseded_by_running_build
    lineage = record.call_args.kwargs["lineage"]
    assert lineage.verdict is EnumLineageVerdict.CONTAINED
    assert str(producer.correlation_id) in lineage.detail
    assert _committed(consumer) == [101]
    assert consumer.notices == [
        ModelRejectionNotice(
            reason=EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD,
            correlation_id=cmd.correlation_id,
            scope=cmd.scope,
        )
    ]


@pytest.mark.unit
def test_an_image_no_job_built_leaves_the_start_unknown(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    _job(store, accepted="17:24:57", completed="17:39:08")
    running = ModelRunningBuild(infra_ref=INFRA_0EDF, build_time=at("17:51:29"))
    consumer = _consumer(running, store)
    cmd = _omnimarket(2808, INFRA_0EDF, "13:52:44")

    accepted, reason = _process(consumer, _message(cmd))

    assert reason is None
    assert accepted == cmd


@pytest.mark.unit
def test_a_covering_job_that_built_another_commit_is_not_trusted(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    _job(store, accepted="17:45:21", completed="18:06:16", git_ref=INFRA_533B)
    running = ModelRunningBuild(infra_ref=INFRA_0EDF, build_time=at("17:51:29"))
    consumer = _consumer(running, store)

    accepted, reason = _process(
        consumer, _message(_omnimarket(2808, INFRA_0EDF, "13:52:44"))
    )

    assert reason is None
    assert accepted is not None


@pytest.mark.unit
def test_without_requested_at_the_broker_timestamp_decides_and_it_builds() -> None:
    """The forwarding hop stamps the record just before the agent reads it."""
    consumer = _consumer(_C009)
    cmd = _command(INFRA_0EDF, requested_by="gha/omnimarket/pr-2805")
    forwarded_ms = int(at("17:24:57").timestamp() * 1000)

    accepted, reason = _process(consumer, _message(cmd, timestamp_ms=forwarded_ms))

    assert reason is None
    assert accepted == cmd
    lineage = consumer.job_store.accept.call_args.kwargs["lineage"]
    assert lineage.verdict is EnumLineageVerdict.EQUAL


@pytest.mark.unit
def test_a_raised_head_builds_alone_at_the_running_ref() -> None:
    consumer = _consumer(_C009)
    cmd = _command(INFRA_C159, requested_by="operator-manual")
    later = _message(_command(INFRA_6FE0), offset=101)

    accepted, reason = _process(consumer, _message(cmd), lookahead=[later])

    assert reason is None
    assert accepted is not None
    assert accepted.git_ref == INFRA_0EDF
    kwargs = consumer.job_store.accept.call_args.kwargs
    assert kwargs["lineage"].verdict is EnumLineageVerdict.RAISED
    assert kwargs["superseded_correlation_ids"] == []


@pytest.mark.unit
def test_the_consumer_refuses_a_diverged_ref_and_commits_past_it() -> None:
    consumer = _consumer(_C009)

    accepted, reason = _process(consumer, _message(_command(FEATURE)))

    assert accepted is None
    assert reason == EnumRejectionReason.DIVERGENT_REF.value
    consumer.job_store.accept.assert_not_called()
    consumer.job_store.record_superseded_by_running_build.assert_not_called()
    assert _committed(consumer) == [101]


@pytest.mark.unit
def test_an_equal_ref_already_seen_is_the_established_duplicate() -> None:
    consumer = _consumer(_C009, duplicate=True)

    accepted, reason = _process(consumer, _message(_command(INFRA_0EDF)))

    assert accepted is None
    assert reason == EnumRejectionReason.DUPLICATE.value


@pytest.mark.unit
def test_a_consumer_without_a_reader_runs_no_fence() -> None:
    consumer = _consumer(_C009)
    consumer.running_build = None
    cmd = _command(INFRA_C159)

    accepted, reason = _process(consumer, _message(cmd))

    assert reason is None
    assert accepted == cmd


# ---------------------------------------------------------------------------
# The job store, the record, the event.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_job_covering_finds_the_one_job_running_at_a_moment(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    first = _job(store, accepted="17:24:57", completed="17:39:08")
    second = _job(store, accepted="17:45:21", completed=None, status="in_progress")
    superseded = JobState(
        correlation_id=uuid4(),
        command={},
        accepted_at=at("17:30:00"),
        completed_at=at("17:30:00"),
        status="superseded",
        superseded_by_sha=INFRA_0EDF,
        superseded_by_running_build=True,
    )
    store._save(superseded)

    assert store.job_covering(at("17:30:00")) == first
    assert store.job_covering(at("17:50:00")) == second
    assert store.job_covering(at("17:42:00")) is None


@pytest.mark.unit
def test_job_covering_refuses_to_choose_between_overlapping_jobs(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    _job(store, accepted="17:00:00", completed="18:00:00")
    _job(store, accepted="17:30:00", completed="18:30:00")

    assert store.job_covering(at("17:45:00")) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ({"scope": "runtime", "services": [], "build_source": "workspace"}, True),
        ({"scope": "full", "services": [], "build_source": "workspace"}, True),
        ({"scope": "core", "services": [], "build_source": "workspace"}, False),
        (
            {"scope": "runtime", "services": ["x"], "build_source": "workspace"},
            False,
        ),
        ({"scope": "full", "services": [], "build_source": "release"}, False),
        (
            {
                "scope": "full",
                "services": [],
                "build_source": "workspace",
                "image_digest": "sha256:" + "0" * 64,
            },
            False,
        ),
    ],
)
def test_only_a_whole_workspace_rebuild_restages_every_sibling(
    command: dict[str, Any], expected: bool
) -> None:
    assert is_workspace_rebuild(command) is expected


@pytest.mark.unit
def test_the_superseded_record_names_the_running_build_and_owes_its_event(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    cmd = _omnimarket(2805, INFRA_0EDF, "13:38:09")
    lineage = _decide(cmd, _C009)

    store.record_superseded_by_running_build(
        cmd.correlation_id, command=cmd.model_dump(mode="json"), lineage=lineage
    )

    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "superseded"
    assert job.superseded_by_running_build is True
    assert job.superseded_by_sha == INFRA_0EDF
    assert job.superseded_by_correlation_id is None
    assert job.lineage == lineage
    assert job.result_publish_pending is True
    assert not store.has_active_job()


@pytest.mark.unit
def test_a_running_build_supersession_must_not_name_a_command() -> None:
    with pytest.raises(ValueError, match="running build"):
        JobState(
            correlation_id=uuid4(),
            command={},
            status="superseded",
            superseded_by_sha=INFRA_0EDF,
            superseded_by_correlation_id=uuid4(),
            superseded_by_running_build=True,
        )


def _agent_with(published: list[Any], *, lands: bool) -> Any:
    from deploy_agent import agent as agent_mod

    instance = agent_mod.DeployAgent.__new__(agent_mod.DeployAgent)
    instance.job_store = Mock()

    def publish(event: Any) -> bool:
        published.append(event)
        return lands

    instance._publish_rejection_event = publish
    return instance


@pytest.mark.unit
@pytest.mark.parametrize("lands", [True, False])
def test_the_notice_clears_the_record_s_publish_debt_only_when_it_lands(
    lands: bool,
) -> None:
    published: list[Any] = []
    agent = _agent_with(published, lands=lands)
    cid = uuid4()

    agent._publish_rejection_notice(
        ModelRejectionNotice(
            reason=EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD,
            correlation_id=cid,
            scope=_command(INFRA_0EDF).scope,
        )
    )

    assert [event.reason for event in published] == [
        EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD
    ]
    assert published[0].superseded_by_sha is None
    if lands:
        agent.job_store.mark_published.assert_called_once_with(cid)
    else:
        agent.job_store.mark_published.assert_not_called()


@pytest.mark.unit
def test_the_retry_loop_republishes_a_running_build_supersession(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    cmd = _omnimarket(2805, INFRA_0EDF, "13:38:09")
    store.record_superseded_by_running_build(
        cmd.correlation_id,
        command=cmd.model_dump(mode="json"),
        lineage=_decide(cmd, _C009),
    )
    job = store.load(cmd.correlation_id)
    published: list[Any] = []

    assert _agent_with(published, lands=True)._publish_superseded_for_job(job)

    assert len(published) == 1
    assert published[0].reason is EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD
    assert published[0].superseded_by_correlation_id is None


# ---------------------------------------------------------------------------
# The command's publish time and requester.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_requested_at_must_carry_an_offset() -> None:
    with pytest.raises(ValueError, match="requested_at"):
        _command(INFRA_0EDF, requested_at=datetime(2026, 9, 23, 13, 52, 44))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("requested_by", "source"),
    [
        ("gha/omnimarket/pr-2808", "omnimarket"),
        ("gha/omnibase_infra/pr-4002", "omnibase_infra"),
        ("operator-manual", None),
        ("gha-redeploy", None),
    ],
)
def test_the_ci_source_repository_is_read_from_requested_by(
    requested_by: str, source: str | None
) -> None:
    assert ci_source_repository(requested_by) == source


# ---------------------------------------------------------------------------
# Direction and resolution, on real commits.
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _real_repo(tmp_path: Path) -> dict[str, Any]:
    """``old <- mid <- new`` on an origin's ``dev``, and ``side`` off ``old``, cloned."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "dev")
    _git(origin, "config", "user.email", "t@example.invalid")
    _git(origin, "config", "user.name", "t")
    shas: dict[str, Any] = {}
    for name in ("old", "mid", "new"):
        _git(origin, "commit", "-q", "--allow-empty", "-m", name)
        shas[name] = _git(origin, "rev-parse", "HEAD")
    _git(origin, "checkout", "-q", "-b", "side", shas["old"])
    _git(origin, "commit", "-q", "--allow-empty", "-m", "side")
    shas["side"] = _git(origin, "rev-parse", "HEAD")
    _git(origin, "checkout", "-q", "dev")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)],
        check=True,
        capture_output=True,
    )
    shas["clone"] = clone
    return shas


@pytest.mark.unit
def test_direction_against_a_real_repository(tmp_path: Path) -> None:
    repo = _real_repo(tmp_path)
    resolver = GitAncestryResolver(str(repo["clone"]))

    def verdict(requested: str, running: str) -> EnumLineageVerdict:
        return decide_lineage(
            _command(requested, requested_by="operator-manual"),
            read_running_build=lambda: ModelRunningBuild(infra_ref=running),
            contains=resolver,
            resolve_ref=None,
            tracking_ref=TRACKING,
            published_at=None,
        ).verdict

    assert verdict(repo["old"], repo["mid"]) is EnumLineageVerdict.RAISED
    assert verdict(repo["new"], repo["mid"]) is EnumLineageVerdict.DESCENDANT
    assert verdict(repo["side"], repo["mid"]) is EnumLineageVerdict.DIVERGENT
    assert verdict(repo["new"], repo["side"]) is EnumLineageVerdict.RETURNS_TO_TRACKING


@pytest.mark.unit
def test_the_ref_resolver_resolves_against_a_fetched_clone(tmp_path: Path) -> None:
    repo = _real_repo(tmp_path)

    resolver = GitRefResolver(str(repo["clone"]))

    assert resolver("origin/dev") == repo["new"]
    assert resolver("origin/no-such-branch") is None


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


@pytest.mark.unit
def test_the_ref_resolver_refuses_to_resolve_after_a_failed_fetch() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(128, stderr="fatal: unable to access")

    assert GitRefResolver("/nowhere", run=run)("origin/dev") is None
    assert [argv[3] for argv in calls] == ["fetch"]


# ---------------------------------------------------------------------------
# The running build is read from the image's own provenance.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_reader_returns_the_infra_ref_and_the_build_time() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(
            0,
            json.dumps(
                {
                    "build_source": "workspace",
                    "build_time": "2026-09-23T19:20:53Z",
                    "infra_vcs_ref": INFRA_6FE0,
                }
            ),
        )

    reader = DockerProvenanceReader(lambda lane: "omninode-runtime", run=run)

    assert reader(EnumRuntimeLane.DEV) == ModelRunningBuild(
        infra_ref=INFRA_6FE0, build_time=at("19:20:53")
    )
    assert calls == [
        ["docker", "exec", "omninode-runtime", "cat", BUILD_PROVENANCE_PATH]
    ]


@pytest.mark.unit
@pytest.mark.parametrize("build_time", ["unknown", "2026-09-23T19:20:53", 17])
def test_an_unreadable_build_time_is_absent_not_guessed(build_time: object) -> None:
    body = json.dumps({"infra_vcs_ref": INFRA_6FE0, "build_time": build_time})
    reader = DockerProvenanceReader(
        lambda lane: "omninode-runtime", run=lambda argv, timeout: _completed(0, body)
    )

    running = reader(EnumRuntimeLane.DEV)

    assert running is not None
    assert running.build_time is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "result",
    [
        _completed(1, stderr="Error: No such container: omninode-runtime"),
        _completed(0, "not json"),
        _completed(0, json.dumps(["a list"])),
        _completed(0, json.dumps({"infra_vcs_ref": "unknown"})),
        _completed(0, json.dumps({"infra_vcs_ref": INFRA_0EDF[:12]})),
    ],
)
def test_the_reader_returns_none_for_anything_it_cannot_trust(
    result: subprocess.CompletedProcess[str],
) -> None:
    reader = DockerProvenanceReader(
        lambda lane: "omninode-runtime", run=lambda argv, timeout: result
    )

    assert reader(EnumRuntimeLane.DEV) is None


@pytest.mark.unit
def test_the_reader_returns_none_when_docker_itself_fails() -> None:
    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, timeout)

    reader = DockerProvenanceReader(lambda lane: "omninode-runtime", run=run)

    assert reader(EnumRuntimeLane.DEV) is None


# ---------------------------------------------------------------------------
# The operator entry point.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_trigger_signs_a_rollback_and_a_publish_time_the_agent_verifies() -> None:
    from deploy_agent.auth import verify_command
    from deploy_agent.trigger import build_rebuild_command, command_to_signed_envelope

    secret = "test-secret"
    command = build_rebuild_command(
        git_ref=INFRA_533B,
        runtime_lane=EnumRuntimeLane.DEV,
        scope=_command(INFRA_533B).scope,
        build_source=_command(INFRA_533B).build_source,
        requested_by="operator-manual",
        correlation_id=UUID(int=7),
        services=[],
        rollback=ModelRollbackDeclaration(actor="operator", reason="bad build"),
        requested_at=datetime(2026, 9, 23, 22, 0, tzinfo=UTC),
    )
    envelope = command_to_signed_envelope(command, secret)

    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": secret}):
        assert verify_command(envelope)
    body = {k: v for k, v in envelope.items() if k != "_signature"}
    parsed = ModelRebuildRequested.model_validate(body)
    assert parsed.rollback == command.rollback
    assert parsed.requested_at == command.requested_at


@pytest.mark.unit
def test_the_trigger_refuses_half_a_rollback_declaration(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from deploy_agent.trigger import main

    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", "test-secret")

    code = main(
        [
            "--git-ref",
            INFRA_533B,
            "--runtime-lane",
            "dev",
            "--rollback-actor",
            "operator",
            "--dry-run",
        ]
    )

    assert code == 1
    assert "--rollback-reason" in capsys.readouterr().err
