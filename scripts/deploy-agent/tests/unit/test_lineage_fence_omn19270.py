# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent never rebuilds what runs, nor rebuilds backwards (OMN-19270).

THE INCIDENT
------------
On 2026-09-23 the .201 dev lane was running infra ``0edf5c914``, vendoring
omnimarket ``c73f6e12``. Seven full rebuilds followed, and each was accepted
hours after it was published:

* ``ab27aedd`` at ``c159b7118`` rebuilt the lane BACKWARDS at 14:46:42Z.
* ``e074126b`` at ``533b19c2`` was behind ``0edf5c914`` too.
* ``602b1d61``, ``5ce1f33e``, ``f645854b`` and ``98bb76df`` were all at
  ``0edf5c914``, each re-delivering an omnimarket merge. ``98bb76df`` failed
  post-deploy verification and left the runtime container in ``Created``.
* ``cb5275c0`` at infra ``6fe05c35`` was the one genuinely new infra deploy.

Under the rule, that sequence is two builds. ``98bb76df`` delivers the only
omnimarket commit the lane did not already vendor, and ``cb5275c0`` delivers
the infra commit. The other five are recorded ``superseded``.

WHAT THESE TESTS PIN
--------------------
* The measured sequence, ``ab27aedd`` through ``cb5275c0``, both with sibling
  refs named, which gives two builds, and without them, as today's producer
  sends, where nothing builds backwards.
* An equal ref, a strict ancestor, a diverged ref (refused, or accepted on the
  way back to the tracking branch), a declared rollback, a symbolic ref, and
  missing provenance.
* A supersession is a durable ``superseded`` job record plus a published
  ``superseded_by_running_build`` event, and it is committed past.
* The provenance reader, the ref resolver and the sibling clone resolver,
  each against what they read.
"""

from __future__ import annotations

import json
import subprocess
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
    SiblingCloneAncestry,
    ci_source_repository,
    decide_lineage,
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

# omnimarket dev, first parent, 2026-09-23: the merge shas of the PRs whose
# rebuilds are in the sequence, in merge order.
MKT_2802 = "8c7e4ccd23b19a84f768dcf1ce4cda38aa2469d1"
MKT_2804 = "3cf62552f4c4410bfa8ab8c8cc57dfaffa161928"
MKT_2805 = "6f4934d0337b91ac823b26b886cefc522697eb11"
MKT_2806 = "e7fa2509d418b8d821f6a26f6b32961d88c46281"
MKT_2801 = "c73f6e1246d1a832358d8e77545d54e5f1808757"
MKT_2808 = "607cce198d85b51bbd5a107533491f723af372e0"

_INFRA_LINE = [INFRA_C159, INFRA_533B, INFRA_0EDF, INFRA_6FE0]
_MKT_LINE = [MKT_2802, MKT_2804, MKT_2805, MKT_2806, MKT_2801, MKT_2808]
#: origin/dev resolves to the infra tip.
_REFS = {TRACKING: INFRA_6FE0}


def _linear_contains(line: list[str]) -> Any:
    def contains(earlier: str, later: str) -> bool | None:
        earlier = _REFS.get(earlier, earlier)
        later = _REFS.get(later, later)
        if earlier == later:
            return True
        if FEATURE in (earlier, later):
            # FEATURE branches off INFRA_533B.
            if later == FEATURE:
                return earlier in line[: line.index(INFRA_533B) + 1]
            return False
        if earlier not in line or later not in line:
            return None
        return line.index(earlier) <= line.index(later)

    return contains


contains = _linear_contains(_INFRA_LINE)
_mkt_contains = _linear_contains(_MKT_LINE)


def contains_sibling(repo: str, earlier: str, later: str) -> bool | None:
    return _mkt_contains(earlier, later) if repo == "omnimarket" else None


def _running(infra: str, omnimarket: str = MKT_2801) -> ModelRunningBuild:
    return ModelRunningBuild(infra_ref=infra, sibling_refs={"omnimarket": omnimarket})


def _command(
    git_ref: str,
    *,
    requested_by: str = "gha/omnibase_infra/pr-4002",
    sibling_refs: dict[str, str] | None = None,
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
        "sibling_refs": sibling_refs or {},
    }
    fields.update(overrides)
    return ModelRebuildRequested.model_validate(fields)


def _omnimarket(
    pr: int, merge_sha: str, git_ref: str, **kw: Any
) -> ModelRebuildRequested:
    return _command(
        git_ref,
        requested_by=f"gha/omnimarket/pr-{pr}",
        sibling_refs={"omnimarket": merge_sha},
        **kw,
    )


def _decide(
    cmd: ModelRebuildRequested,
    running: ModelRunningBuild | None,
    *,
    resolve_ref: Any = None,
) -> ModelLineageDecision:
    return decide_lineage(
        cmd,
        read_running_build=lambda: running,
        contains=contains,
        contains_sibling=contains_sibling,
        resolve_ref=resolve_ref,
        tracking_ref=TRACKING,
    )


def _message(cmd: ModelRebuildRequested, offset: int = 100) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json") | {"_signature": "a" * 64}
    return SimpleNamespace(
        value=payload, topic=TOPIC, partition=0, offset=offset, key=None
    )


def _consumer(
    running: ModelRunningBuild | None,
    *,
    resolver: Any = contains,
    duplicate: bool = False,
    ref_resolver: Any = None,
) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = duplicate
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    consumer.ancestry_resolver = resolver
    consumer.running_build = Mock(return_value=running)
    consumer.sibling_ancestry = contains_sibling
    consumer.ref_resolver = ref_resolver
    consumer.tracking_ref = TRACKING
    consumer.notices = []
    consumer.on_rejected = consumer.notices.append
    return consumer


def _process(
    consumer: DeployConsumer, cmd: ModelRebuildRequested, lookahead: Any = None
) -> tuple[ModelRebuildRequested | None, str | None]:
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        return consumer._process_message(_message(cmd), lookahead=lookahead)


def _committed(consumer: DeployConsumer) -> list[int]:
    return [
        meta.offset
        for call in consumer.consumer.commit.call_args_list
        for meta in call.args[0].values()
    ]


# ---------------------------------------------------------------------------
# The measured sequence.
# ---------------------------------------------------------------------------

#: (job, command) in the order the agent accepted them on 2026-09-23.
_SEQUENCE: list[tuple[str, ModelRebuildRequested]] = [
    ("ab27aedd", _omnimarket(2802, MKT_2802, INFRA_C159)),
    ("e074126b", _omnimarket(2804, MKT_2804, INFRA_533B)),
    ("602b1d61", _omnimarket(2806, MKT_2806, INFRA_0EDF)),
    ("5ce1f33e", _omnimarket(2805, MKT_2805, INFRA_0EDF)),
    ("f645854b", _omnimarket(2801, MKT_2801, INFRA_0EDF)),
    ("98bb76df", _omnimarket(2808, MKT_2808, INFRA_0EDF)),
    ("cb5275c0", _command(INFRA_6FE0, requested_by="gha/omnibase_infra/pr-4002")),
]


def _replay(
    sequence: list[tuple[str, ModelRebuildRequested]],
) -> list[tuple[str, EnumLineageVerdict, str]]:
    """Run the sequence against a lane whose build moves as the rule says.

    The lane starts where it was: infra ``0edf5c914`` from ``c009462c``,
    vendoring omnimarket ``c73f6e12``. A workspace build stages omnimarket's
    dev head, which by the time any of these ran carried ``607cce19``.
    """
    running = _running(INFRA_0EDF, MKT_2801)
    outcomes = []
    for job, cmd in sequence:
        decision = _decide(cmd, running)
        assert not decision.verdict.refuses, (job, decision.journal_line())
        if not decision.verdict.supersedes:
            assert contains(running.infra_ref, decision.build_ref), (
                f"{job} would build {decision.build_ref}, behind the running "
                f"{running.infra_ref}"
            )
            running = _running(decision.build_ref, MKT_2808)
        outcomes.append((job, decision.verdict, decision.build_ref))
    return outcomes


@pytest.mark.unit
def test_the_measured_sequence_is_two_builds_when_sibling_refs_are_named() -> None:
    outcomes = _replay(_SEQUENCE)

    assert outcomes == [
        ("ab27aedd", EnumLineageVerdict.CONTAINED, INFRA_0EDF),
        ("e074126b", EnumLineageVerdict.CONTAINED, INFRA_0EDF),
        ("602b1d61", EnumLineageVerdict.CONTAINED, INFRA_0EDF),
        ("5ce1f33e", EnumLineageVerdict.CONTAINED, INFRA_0EDF),
        ("f645854b", EnumLineageVerdict.CONTAINED, INFRA_0EDF),
        # The one omnimarket commit the lane did not vendor yet.
        ("98bb76df", EnumLineageVerdict.EQUAL, INFRA_0EDF),
        ("cb5275c0", EnumLineageVerdict.DESCENDANT, INFRA_6FE0),
    ]


@pytest.mark.unit
def test_the_measured_sequence_never_builds_backwards_without_sibling_refs() -> None:
    """Today's producer names no sibling ref: nothing is superseded, nothing rolls back."""
    bare = [
        (job, cmd.model_copy(update={"sibling_refs": {}})) for job, cmd in _SEQUENCE
    ]

    outcomes = _replay(bare)

    verdicts = [verdict for _, verdict, _ in outcomes]
    assert EnumLineageVerdict.CONTAINED not in verdicts
    assert outcomes[0] == ("ab27aedd", EnumLineageVerdict.RAISED, INFRA_0EDF)
    assert outcomes[1] == ("e074126b", EnumLineageVerdict.RAISED, INFRA_0EDF)
    assert outcomes[-1] == ("cb5275c0", EnumLineageVerdict.DESCENDANT, INFRA_6FE0)


# ---------------------------------------------------------------------------
# Supersession: durable, acknowledged, published.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_contained_command_is_recorded_superseded_and_committed_past() -> None:
    consumer = _consumer(_running(INFRA_0EDF))
    cmd = _omnimarket(2805, MKT_2805, INFRA_0EDF)

    accepted, reason = _process(consumer, cmd)

    assert accepted is None
    assert reason == EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD.value
    consumer.job_store.accept.assert_not_called()
    record = consumer.job_store.record_superseded_by_running_build.call_args
    assert record.args[0] == cmd.correlation_id
    assert record.kwargs["lineage"].verdict is EnumLineageVerdict.CONTAINED
    assert record.kwargs["lineage"].running_ref == INFRA_0EDF
    assert _committed(consumer) == [101]
    assert consumer.notices == [
        ModelRejectionNotice(
            reason=EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD,
            correlation_id=cmd.correlation_id,
            scope=cmd.scope,
        )
    ]


@pytest.mark.unit
def test_the_superseded_record_names_the_running_build_and_owes_its_event(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    cmd = _omnimarket(2805, MKT_2805, INFRA_0EDF)
    lineage = _decide(cmd, _running(INFRA_0EDF))

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
    cmd = _omnimarket(2805, MKT_2805, INFRA_0EDF)
    store.record_superseded_by_running_build(
        cmd.correlation_id,
        command=cmd.model_dump(mode="json"),
        lineage=_decide(cmd, _running(INFRA_0EDF)),
    )
    job = store.load(cmd.correlation_id)
    published: list[Any] = []

    assert _agent_with(published, lands=True)._publish_superseded_for_job(job)

    assert len(published) == 1
    assert published[0].correlation_id == cmd.correlation_id
    assert published[0].reason is EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD
    assert published[0].superseded_by_correlation_id is None


# ---------------------------------------------------------------------------
# An equal ref.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_an_equal_infra_ref_from_an_infra_merge_is_superseded() -> None:
    decision = _decide(_command(INFRA_0EDF), _running(INFRA_0EDF))

    assert decision.verdict is EnumLineageVerdict.CONTAINED


@pytest.mark.unit
def test_an_equal_ref_carrying_an_undelivered_sibling_builds() -> None:
    cmd = _omnimarket(2808, MKT_2808, INFRA_0EDF)

    decision = _decide(cmd, _running(INFRA_0EDF, MKT_2801))

    assert decision.verdict is EnumLineageVerdict.EQUAL
    assert decision.build_ref == INFRA_0EDF
    assert "607cce19" in decision.detail


@pytest.mark.unit
def test_a_sibling_merge_that_names_no_sibling_ref_is_never_superseded() -> None:
    """Superseding on the infra ref alone would strand a sibling merge on a quiet infra day."""
    cmd = _command(INFRA_0EDF, requested_by="gha/omnimarket/pr-2808")

    decision = _decide(cmd, _running(INFRA_0EDF, MKT_2808))

    assert decision.verdict is EnumLineageVerdict.EQUAL
    assert "names no omnimarket ref" in decision.detail


@pytest.mark.unit
@pytest.mark.parametrize("requested_by", ["operator-manual", "lab-health-triage"])
def test_a_deliberate_request_at_the_running_ref_is_never_superseded(
    requested_by: str,
) -> None:
    """A same-ref rebuild is how a wedged lane is recovered."""
    consumer = _consumer(_running(INFRA_0EDF))
    cmd = _command(INFRA_0EDF, requested_by=requested_by)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd
    lineage = consumer.job_store.accept.call_args.kwargs["lineage"]
    assert lineage.verdict is EnumLineageVerdict.EQUAL


@pytest.mark.unit
def test_an_equal_ref_already_seen_is_the_established_duplicate() -> None:
    consumer = _consumer(_running(INFRA_0EDF), duplicate=True)

    accepted, reason = _process(consumer, _command(INFRA_0EDF))

    assert accepted is None
    assert reason == EnumRejectionReason.DUPLICATE.value


# ---------------------------------------------------------------------------
# A strict ancestor.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_strict_ancestor_from_an_infra_merge_is_superseded() -> None:
    decision = _decide(_command(INFRA_533B), _running(INFRA_0EDF))

    assert decision.verdict is EnumLineageVerdict.CONTAINED
    assert decision.build_ref == INFRA_0EDF


@pytest.mark.unit
def test_a_strict_ancestor_that_must_build_is_raised_to_the_running_ref() -> None:
    consumer = _consumer(_running(INFRA_0EDF))
    cmd = _command(INFRA_C159, requested_by="operator-manual")
    later = _message(_command(INFRA_6FE0), offset=101)

    accepted, reason = _process(consumer, cmd, lookahead=[later])

    assert reason is None
    assert accepted is not None
    assert accepted.git_ref == INFRA_0EDF
    assert accepted.correlation_id == cmd.correlation_id
    lineage = consumer.job_store.accept.call_args.kwargs["lineage"]
    assert lineage.verdict is EnumLineageVerdict.RAISED
    assert lineage.requested_ref == INFRA_C159
    assert lineage.build_ref == INFRA_0EDF
    # A raised head builds alone; the newer command is not folded into it.
    assert (
        consumer.job_store.accept.call_args.kwargs["superseded_correlation_ids"] == []
    )


@pytest.mark.unit
def test_a_descendant_builds_its_own_ref() -> None:
    consumer = _consumer(_running(INFRA_0EDF))
    cmd = _command(INFRA_6FE0)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd


# ---------------------------------------------------------------------------
# A diverged ref.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_diverged_ref_off_the_tracking_branch_is_refused() -> None:
    consumer = _consumer(_running(INFRA_0EDF))

    accepted, reason = _process(consumer, _command(FEATURE))

    assert accepted is None
    assert reason == EnumRejectionReason.DIVERGENT_REF.value
    consumer.job_store.accept.assert_not_called()
    consumer.job_store.record_superseded_by_running_build.assert_not_called()
    assert _committed(consumer) == [101]


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
    consumer = _consumer(_running(INFRA_0EDF))
    cmd = _command(
        target,
        requested_by="operator-manual",
        rollback=ModelRollbackDeclaration(
            actor="operator", reason="0edf5c914 broke the projection writers"
        ),
    )

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd
    # A rollback is decided on the declaration alone; the lane is not read.
    consumer.running_build.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("field", ["actor", "reason"])
def test_a_rollback_must_name_its_actor_and_reason(field: str) -> None:
    fields = {"actor": "operator", "reason": "bad build"} | {field: "   "}
    with pytest.raises(ValueError, match=field):
        ModelRollbackDeclaration(**fields)


@pytest.mark.unit
@pytest.mark.parametrize("rollback_first", [True, False])
def test_coalescing_never_folds_a_rollback(rollback_first: bool) -> None:
    """A newer command must not run in place of a deliberate rollback, or vice versa."""
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
# A symbolic ref.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_symbolic_ref_is_resolved_at_accept_and_built_at_that_sha() -> None:
    """``c009462c`` asked for origin/dev and its record never said which commit."""
    consumer = _consumer(_running(INFRA_0EDF), ref_resolver=_REFS.get)
    cmd = _command(TRACKING, requested_by="lab-health-triage")

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted is not None
    assert accepted.git_ref == INFRA_6FE0
    lineage = consumer.job_store.accept.call_args.kwargs["lineage"]
    assert lineage.requested_ref == TRACKING
    assert lineage.resolved_ref == INFRA_6FE0
    assert lineage.build_ref == INFRA_6FE0


@pytest.mark.unit
def test_an_unresolvable_symbolic_ref_builds_as_requested() -> None:
    consumer = _consumer(_running(INFRA_0EDF), ref_resolver=lambda ref: None)
    cmd = _command(TRACKING, requested_by="lab-health-triage")

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd
    consumer.running_build.assert_not_called()


@pytest.mark.unit
def test_the_ref_resolver_resolves_against_a_fetched_clone(tmp_path: Path) -> None:
    repo = _real_repo(tmp_path)

    resolver = GitRefResolver(str(repo["clone"]))

    assert resolver("origin/dev") == repo["new"]
    assert resolver("origin/no-such-branch") is None


@pytest.mark.unit
def test_the_ref_resolver_refuses_to_resolve_after_a_failed_fetch() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(128, stderr="fatal: unable to access")

    assert GitRefResolver("/nowhere", run=run)("origin/dev") is None
    assert [argv[3] for argv in calls] == ["fetch"]


# ---------------------------------------------------------------------------
# Missing provenance fails open.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_an_unreadable_running_build_builds_as_requested() -> None:
    """A lane that is down has no container to read, and must accept a rebuild."""
    consumer = _consumer(None)
    cmd = _command(INFRA_C159)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd
    lineage = consumer.job_store.accept.call_args.kwargs["lineage"]
    assert lineage.verdict is EnumLineageVerdict.UNPROVEN


@pytest.mark.unit
def test_an_unanswerable_ancestry_builds_as_requested() -> None:
    consumer = _consumer(_running(INFRA_0EDF), resolver=lambda earlier, later: None)
    cmd = _command(INFRA_C159)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd


@pytest.mark.unit
def test_a_sibling_the_running_build_did_not_record_is_not_contained() -> None:
    cmd = _omnimarket(2805, MKT_2805, INFRA_0EDF)

    decision = _decide(cmd, ModelRunningBuild(infra_ref=INFRA_0EDF))

    assert decision.verdict is EnumLineageVerdict.EQUAL


@pytest.mark.unit
def test_a_pinned_image_never_reads_the_lane() -> None:
    consumer = _consumer(_running(INFRA_0EDF))
    cmd = _command(INFRA_C159, image_digest="sha256:" + "0" * 64)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd
    consumer.running_build.assert_not_called()


@pytest.mark.unit
def test_a_consumer_without_a_reader_runs_no_fence() -> None:
    """The pre-change behaviour, reached for every consumer built without one."""
    consumer = _consumer(_running(INFRA_0EDF))
    consumer.running_build = None
    cmd = _command(INFRA_C159)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd


# ---------------------------------------------------------------------------
# The command's sibling refs and its requester.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "sibling_refs",
    [
        {"omnimarket": MKT_2805[:12]},
        {"omnibase_infra": INFRA_0EDF},
        {"../etc": MKT_2805},
    ],
)
def test_sibling_refs_are_full_shas_of_sibling_repositories(
    sibling_refs: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="sibling_refs"):
        _command(INFRA_0EDF, sibling_refs=sibling_refs)


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
# Direction, on real commits.
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
    """The fence and the real resolver agree on direction, on real commits."""
    repo = _real_repo(tmp_path)
    resolver = GitAncestryResolver(str(repo["clone"]))

    def verdict(requested: str, running: str) -> EnumLineageVerdict:
        return decide_lineage(
            _command(requested, requested_by="operator-manual"),
            read_running_build=lambda: ModelRunningBuild(infra_ref=running),
            contains=resolver,
            contains_sibling=contains_sibling,
            resolve_ref=None,
            tracking_ref=TRACKING,
        ).verdict

    assert verdict(repo["old"], repo["mid"]) is EnumLineageVerdict.RAISED
    assert verdict(repo["new"], repo["mid"]) is EnumLineageVerdict.DESCENDANT
    assert verdict(repo["side"], repo["mid"]) is EnumLineageVerdict.DIVERGENT
    assert verdict(repo["new"], repo["side"]) is EnumLineageVerdict.RETURNS_TO_TRACKING


@pytest.mark.unit
def test_sibling_ancestry_asks_the_clone_the_build_stages_from(tmp_path: Path) -> None:
    repo = _real_repo(tmp_path)
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir()
    (repo["clone"]).rename(omni_home / "omnimarket")

    ancestry = SiblingCloneAncestry(str(omni_home))

    assert ancestry("omnimarket", repo["old"], repo["new"]) is True
    assert ancestry("omnimarket", repo["new"], repo["old"]) is False
    assert ancestry("omnibase_compat", repo["old"], repo["new"]) is None
    assert SiblingCloneAncestry(None)("omnimarket", repo["old"], repo["new"]) is None


# ---------------------------------------------------------------------------
# The running build is read from the image's own provenance.
# ---------------------------------------------------------------------------


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


#: The shape the .201 dev lane's image carried on 2026-09-23 (trimmed).
_PROVENANCE = {
    "build_source": "workspace",
    "infra_vcs_ref": INFRA_6FE0,
    "per_repo_vcs_provenance": {
        "siblings": {
            "omnimarket": {"vcs_ref": MKT_2808, "vcs_dirty": False},
            "omnibase_compat": {"vcs_ref": "9" * 40, "vcs_dirty": True},
            "omnibase_core": {"vcs_ref": "unknown", "vcs_dirty": False},
        }
    },
}


@pytest.mark.unit
def test_the_reader_returns_the_infra_ref_and_every_clean_sibling_ref() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(0, json.dumps(_PROVENANCE))

    reader = DockerProvenanceReader(lambda lane: "omninode-runtime", run=run)

    assert reader(EnumRuntimeLane.DEV) == ModelRunningBuild(
        infra_ref=INFRA_6FE0, sibling_refs={"omnimarket": MKT_2808}
    )
    assert calls == [
        ["docker", "exec", "omninode-runtime", "cat", BUILD_PROVENANCE_PATH]
    ]


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
# The operator entry point can publish a rollback.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_trigger_signs_a_rollback_the_agent_verifies() -> None:
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
    )
    envelope = command_to_signed_envelope(command, secret)

    with patch.dict("os.environ", {"DEPLOY_AGENT_HMAC_SECRET": secret}):
        assert verify_command(envelope)
    assert envelope["rollback"] == {"actor": "operator", "reason": "bad build"}
    body = {k: v for k, v in envelope.items() if k != "_signature"}
    assert ModelRebuildRequested.model_validate(body).rollback == command.rollback


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
