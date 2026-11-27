# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev deploy queue never runs a stale build (OMN-26637).

WHAT THIS COST, MEASURED
------------------------
2026-11-24: the dev deploy agent's queue held about twenty build commands
(``requested_by`` ``gha/omnibase_infra/pr-NNNN``, from 03:45Z) and ran them
oldest-first at about 28 minutes each; one stale build at 11:18Z walked dev six
merges backwards. Coalescing folded only a chain in which each pair is
ancestor-related, and omnibase_infra#6239 still keyed on containment.

OPERATOR RULING 2026-11-24, about 12:20Z: "As soon as a new build comes out, all
previous queued builds should be canceled, right?" For the dev lane, accepting a
request supersedes every older queued request, whatever git says about the pair,
and dev takes only merged dev-branch refs.

THREE HALVES, ONE TEST CLASS EACH
---------------------------------
* ``supersede_dev_queue``: the coalesce path folds every older queued request of
  the lane, each recorded with one reason naming the accepting request.
* ``dev_accepts_only_merged_refs``: the agent refuses a ref its tracking branch
  lacks, with the typed reason ``not_a_merged_ref``.
* ``pr_head_not_published_to_dev``: the publisher refuses a PR head, and both
  workflows that run it pass the pull-request event's own facts.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import yaml
from click.testing import CliRunner
from deploy_agent.coalesce import (
    EnumCoalesceRefusal,
    ModelQueuedCommand,
    plan_coalesce,
)
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    EnumLineageVerdict,
    EnumRejectionReason,
    EnumRuntimeLane,
    ModelRebuildRequested,
    ModelRollbackDeclaration,
)
from deploy_agent.job_state import JobStore
from deploy_agent.lane_policy import MERGED_REF_LANES, SUPERSEDING_LANES
from deploy_agent.ref_fence import EnumMergedRefVerdict, decide_merged_ref
from kafka import TopicPartition

from scripts import trigger_rebuild_on_merge as trigger

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_SOURCE = REPO_ROOT / "scripts" / "deploy-agent" / "deploy_agent" / "agent.py"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
TRACKING = "origin/dev"

#: Four squash-merge commits and a PR head. None contains another.
MERGE_1 = "1111111a" + "0" * 32
MERGE_2 = "2222222b" + "0" * 32
MERGE_3 = "3333333c" + "0" * 32
MERGE_4 = "4444444d" + "0" * 32
PR_HEAD = "9999999e" + "0" * 32
MERGED = {MERGE_1, MERGE_2, MERGE_3, MERGE_4}


def _cmd(sha: str, **overrides: Any) -> ModelRebuildRequested:
    payload: dict[str, Any] = {
        "correlation_id": uuid4(),
        "requested_by": "gha/omnibase_infra/pr-1",
        "scope": "full",
        "runtime_lane": "dev",
        "build_source": "workspace",
        "services": [],
        "git_ref": sha,
    }
    payload.update(overrides)
    return ModelRebuildRequested.model_validate(payload)


def _queued(cmd: ModelRebuildRequested, offset: int) -> ModelQueuedCommand:
    return ModelQueuedCommand(command=cmd, partition=0, offset=offset)


def _never_contained(earlier: str, later: str) -> bool | None:
    """Squash merges: no sha contains another one."""
    return earlier == later


def _tracking_contains_merges(ref: str, tracking: str) -> bool | None:
    assert tracking == TRACKING
    return ref in MERGED


def _message(cmd: ModelRebuildRequested, offset: int) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json")
    payload["_signature"] = "a" * 64
    return SimpleNamespace(value=payload, topic=TOPIC, partition=0, offset=offset)


def _consumer(store: JobStore, **attrs: Any) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = store
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    for key, value in attrs.items():
        setattr(consumer, key, value)
    return consumer


def _committed(consumer: DeployConsumer) -> list[int]:
    offsets: list[int] = []
    for call in consumer.consumer.commit.call_args_list:
        offsets.extend(meta.offset for meta in call.args[0].values())
    return offsets


class TestSupersedeDevQueue:
    def test_supersede_dev_queue_plan_folds_every_older_request_without_ancestry(
        self,
    ) -> None:
        cmds = [_cmd(sha) for sha in (MERGE_1, MERGE_2, MERGE_3, MERGE_4)]
        plan = plan_coalesce(
            [_queued(c, 290 + i) for i, c in enumerate(cmds)],
            contains=_never_contained,
            supersede_lanes=SUPERSEDING_LANES,
        )

        assert plan.runner.command.correlation_id == cmds[-1].correlation_id
        assert [s.superseded.command.correlation_id for s in plan.superseded] == [
            c.correlation_id for c in cmds[:3]
        ]
        assert all(not s.by_containment for s in plan.superseded)
        assert plan.stop_reason is None

    def test_supersede_dev_queue_control_without_the_lane_ancestry_still_ends_the_group(
        self,
    ) -> None:
        """Positive control: the same batch, lane not named, folds nothing."""
        cmds = [_cmd(sha) for sha in (MERGE_1, MERGE_2, MERGE_3)]
        plan = plan_coalesce(
            [_queued(c, 290 + i) for i, c in enumerate(cmds)],
            contains=_never_contained,
            supersede_lanes=frozenset(),
        )

        assert plan.superseded == ()
        assert plan.runner.command.correlation_id == cmds[0].correlation_id

    def test_supersede_dev_queue_another_lane_is_not_folded(self) -> None:
        head = _cmd(MERGE_1)
        other = _cmd(
            MERGE_2,
            runtime_lane=EnumRuntimeLane.STABILITY_TEST.value,
            build_source="release",
        )
        plan = plan_coalesce(
            [_queued(head, 1), _queued(other, 2)],
            contains=_never_contained,
            supersede_lanes=SUPERSEDING_LANES,
        )

        assert plan.superseded == ()

    @pytest.mark.parametrize("rollback_first", [True, False])
    def test_supersede_dev_queue_a_declared_rollback_is_never_folded(
        self, rollback_first: bool
    ) -> None:
        plain = _cmd(MERGE_1)
        rolled = _cmd(
            MERGE_2,
            rollback=ModelRollbackDeclaration(actor="operator", reason="bad build"),
        )
        head, candidate = (rolled, plain) if rollback_first else (plain, rolled)

        plan = plan_coalesce(
            [_queued(head, 1), _queued(candidate, 2)],
            contains=_never_contained,
            supersede_lanes=SUPERSEDING_LANES,
        )

        assert plan.superseded == ()
        assert plan.stop_reason is EnumCoalesceRefusal.ROLLBACK_DECLARED

    def test_supersede_dev_queue_each_job_record_names_the_accepting_request(
        self, tmp_path: Path
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        cmds = [_cmd(sha) for sha in (MERGE_1, MERGE_2, MERGE_3, MERGE_4)]
        announced: list[Any] = []
        consumer = _consumer(
            store,
            ancestry_resolver=_never_contained,
            on_superseded=announced.append,
            supersede_lanes=SUPERSEDING_LANES,
        )
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [_message(c, 290 + i) for i, c in enumerate(cmds)]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, reason = consumer.poll_and_accept()

        assert reason is None
        assert accepted is not None
        assert accepted.correlation_id == cmds[-1].correlation_id
        records = [store.load(c.correlation_id) for c in cmds[:3]]
        assert len(records) == 3, "positive control: three older requests were queued"
        for record in records:
            assert record is not None
            assert record.status == "superseded"
            assert record.superseded_by_correlation_id == accepted.correlation_id
            assert record.superseded_by_sha == MERGE_4
            assert len(record.errors) == 1, "exactly one reason"
            assert str(accepted.correlation_id) in record.errors[0]
            assert "whatever its git ancestry" in record.errors[0]
            assert "contains this command's ref" not in record.errors[0]
        assert len(announced) == 3
        assert _committed(consumer) == [294]

    def test_supersede_dev_queue_a_runner_the_running_build_already_carries_folds_nothing(
        self, tmp_path: Path
    ) -> None:
        """Publish order is not merge order: a stale newest request must not run."""
        store = JobStore(state_dir=tmp_path / "jobs")
        head, stale = _cmd(MERGE_3), _cmd(MERGE_1)
        consumer = _consumer(
            store,
            ancestry_resolver=_never_contained,
            supersede_lanes=SUPERSEDING_LANES,
        )
        calls: list[str] = []

        def _fence(cmd: ModelRebuildRequested, msg: Any) -> Any:
            calls.append(cmd.git_ref)
            if cmd.git_ref == MERGE_1:
                return SimpleNamespace(
                    verdict=EnumLineageVerdict.CONTAINED,
                    journal_line=lambda: "stale",
                    build_ref=MERGE_1,
                )
            return None

        consumer._lineage_decision = _fence
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [_message(head, 290), _message(stale, 291)]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, _ = consumer.poll_and_accept()

        assert accepted is not None
        assert accepted.correlation_id == head.correlation_id
        assert store.load(stale.correlation_id) is None
        assert MERGE_1 in calls, (
            "positive control: the fence was asked about the runner"
        )

    def test_supersede_dev_queue_is_wired_into_the_agent(self) -> None:
        keywords = _consumer_keywords()
        assert ast.unparse(keywords["supersede_lanes"]) == "SUPERSEDING_LANES"
        assert EnumRuntimeLane.DEV in SUPERSEDING_LANES


def _consumer_keywords() -> dict[str, ast.expr]:
    tree = ast.parse(AGENT_SOURCE.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DeployConsumer"
        ):
            return {k.arg: k.value for k in node.keywords if k.arg is not None}
    raise AssertionError("agent.py no longer constructs a DeployConsumer")


class TestDevAcceptsOnlyMergedRefs:
    @pytest.mark.parametrize(
        ("sha", "tracking_answer", "expected"),
        [
            (MERGE_1, True, EnumMergedRefVerdict.MERGED),
            (PR_HEAD, False, EnumMergedRefVerdict.NOT_MERGED),
            (PR_HEAD, None, EnumMergedRefVerdict.UNPROVEN),
        ],
    )
    def test_dev_accepts_only_merged_refs_verdicts(
        self, sha: str, tracking_answer: bool | None, expected: EnumMergedRefVerdict
    ) -> None:
        verdict = decide_merged_ref(
            _cmd(sha),
            lanes=MERGED_REF_LANES,
            tracking_ref=TRACKING,
            contains=lambda ref, tracking: tracking_answer,
        )

        assert verdict is expected
        assert verdict.refuses is (expected is EnumMergedRefVerdict.NOT_MERGED)

    def test_dev_accepts_only_merged_refs_other_lanes_and_promotions_are_not_judged(
        self,
    ) -> None:
        def _boom(ref: str, tracking: str) -> bool | None:
            raise AssertionError("must not be asked")

        stability = _cmd(
            PR_HEAD,
            runtime_lane=EnumRuntimeLane.STABILITY_TEST.value,
            build_source="release",
        )
        pinned = _cmd(
            PR_HEAD, image_ref="registry/x:1", image_digest="sha256:" + "a" * 64
        )
        for cmd in (stability, pinned):
            assert (
                decide_merged_ref(
                    cmd, lanes=MERGED_REF_LANES, tracking_ref=TRACKING, contains=_boom
                )
                is EnumMergedRefVerdict.NOT_APPLICABLE
            )
        assert (
            decide_merged_ref(
                _cmd(PR_HEAD),
                lanes=frozenset(),
                tracking_ref=TRACKING,
                contains=_boom,
            )
            is EnumMergedRefVerdict.NOT_APPLICABLE
        )

    def test_dev_accepts_only_merged_refs_a_pr_head_is_refused_with_its_typed_reason(
        self, tmp_path: Path
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        rejected: list[Any] = []
        consumer = _consumer(
            store,
            ancestry_resolver=_tracking_contains_merges,
            on_rejected=rejected.append,
            merged_ref_lanes=MERGED_REF_LANES,
            tracking_ref=TRACKING,
        )
        head = _cmd(PR_HEAD, requested_by="lane/some-pr-proof")
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [_message(head, 290)]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, reason = consumer.poll_and_accept()

        assert accepted is None
        assert (
            reason == EnumRejectionReason.NOT_A_MERGED_REF.value == "not_a_merged_ref"
        )
        assert store.load(head.correlation_id) is None, "nothing was accepted"
        assert _committed(consumer) == [291], "the refusal commits past the record"
        assert [n.reason for n in rejected] == [EnumRejectionReason.NOT_A_MERGED_REF]

    def test_dev_accepts_only_merged_refs_a_merged_ref_is_accepted(
        self, tmp_path: Path
    ) -> None:
        """Positive control for the refusal above: the same wiring accepts a merge."""
        store = JobStore(state_dir=tmp_path / "jobs")
        consumer = _consumer(
            store,
            ancestry_resolver=_tracking_contains_merges,
            merged_ref_lanes=MERGED_REF_LANES,
            tracking_ref=TRACKING,
        )
        merge = _cmd(MERGE_1)
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [_message(merge, 290)]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, reason = consumer.poll_and_accept()

        assert reason is None
        assert accepted is not None
        assert accepted.correlation_id == merge.correlation_id

    def test_dev_accepts_only_merged_refs_a_pr_head_behind_a_merge_never_becomes_the_runner(
        self, tmp_path: Path
    ) -> None:
        store = JobStore(state_dir=tmp_path / "jobs")
        first, second, pr_head = _cmd(MERGE_1), _cmd(MERGE_2), _cmd(PR_HEAD)
        consumer = _consumer(
            store,
            ancestry_resolver=lambda a, b: (
                _tracking_contains_merges(a, b) if b == TRACKING else a == b
            ),
            supersede_lanes=SUPERSEDING_LANES,
            merged_ref_lanes=MERGED_REF_LANES,
            tracking_ref=TRACKING,
        )
        consumer.consumer.poll.return_value = {
            TopicPartition(TOPIC, 0): [
                _message(first, 290),
                _message(second, 291),
                _message(pr_head, 292),
            ]
        }

        with patch("deploy_agent.consumer.verify_command", return_value=True):
            accepted, _ = consumer.poll_and_accept()

        assert accepted is not None
        assert accepted.correlation_id == second.correlation_id
        assert store.load(first.correlation_id) is not None
        assert store.load(pr_head.correlation_id) is None, "left for the head path"
        assert _committed(consumer) == [292], "commits through the runner only"

    def test_dev_accepts_only_merged_refs_is_wired_into_the_agent(self) -> None:
        keywords = _consumer_keywords()
        assert ast.unparse(keywords["merged_ref_lanes"]) == "MERGED_REF_LANES"
        assert EnumRuntimeLane.DEV in MERGED_REF_LANES


def _invoke(*extra: str, env: dict[str, str] | None = None) -> Any:
    return CliRunner().invoke(
        trigger.main,
        [
            "--changed-files",
            "src/omnibase_infra/runtime/x.py",
            "--labels",
            "runtime_change",
            "--base-branch",
            "dev",
            "--source-sha",
            MERGE_1,
            "--runtime-path-validator",
            str(REPO_ROOT / "scripts" / "runtime_change_classifier.py"),
            "--dry-run",
            *extra,
        ],
        env=env,
    )


class TestPrHeadNotPublishedToDev:
    def test_pr_head_not_published_to_dev_an_unmerged_pr_is_refused(self) -> None:
        result = _invoke("--pr-merged", "false")

        assert result.exit_code == 1
        assert "not merged" in result.output
        assert "pre-PR slot" in result.output

    def test_pr_head_not_published_to_dev_a_source_equal_to_the_pr_head_is_refused(
        self,
    ) -> None:
        result = _invoke("--pr-merged", "true", "--pr-head-sha", MERGE_1)

        assert result.exit_code == 1
        assert "head commit" in result.output

    def test_pr_head_not_published_to_dev_a_merge_commit_passes_the_guard(
        self,
    ) -> None:
        """Positive control: the guard is what refuses above, not a bad invocation."""
        result = _invoke("--pr-merged", "true", "--pr-head-sha", PR_HEAD)

        assert "not merged" not in result.output
        assert "head commit" not in result.output

    @pytest.mark.parametrize(
        "workflow",
        ["runtime-rebuild-trigger.yml", "runtime-rebuild-trigger-reusable.yml"],
    )
    def test_pr_head_not_published_to_dev_every_publisher_passes_the_event_facts(
        self, workflow: str
    ) -> None:
        text = (WORKFLOWS / workflow).read_text()
        document = yaml.safe_load(text)
        steps = [
            step
            for job in document["jobs"].values()
            for step in job.get("steps", [])
            if "trigger_rebuild_on_merge.py" in step.get("run", "")
        ]

        assert len(steps) == 1, "positive control: exactly one publisher step"
        step = steps[0]
        assert '--pr-merged "$PR_MERGED"' in step["run"]
        assert '--pr-head-sha "$PR_HEAD_SHA"' in step["run"]
        assert step["env"]["PR_MERGED"] == "${{ github.event.pull_request.merged }}"
        assert step["env"]["PR_HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"

    def test_pr_head_not_published_to_dev_the_own_publisher_runs_on_merged_prs_only(
        self,
    ) -> None:
        document = yaml.safe_load(
            (WORKFLOWS / "runtime-rebuild-trigger.yml").read_text()
        )
        job = document["jobs"]["trigger-rebuild"]

        assert job["if"] == "github.event.pull_request.merged == true"
        step = next(
            s for s in job["steps"] if "trigger_rebuild_on_merge.py" in s.get("run", "")
        )
        assert (
            step["env"]["PR_MERGE_SHA"]
            == "${{ github.event.pull_request.merge_commit_sha }}"
        )
        assert '--source-sha "$PR_MERGE_SHA"' in step["run"]


def test_the_two_readbacks_agree_on_the_shape_of_a_superseded_record(
    tmp_path: Path,
) -> None:
    """The record is plain JSON a reader can list from the jobs directory (AC4 shape)."""
    store = JobStore(state_dir=tmp_path / "jobs")
    older, newer = uuid4(), uuid4()
    store.record_superseded(
        older,
        {"git_ref": MERGE_1},
        superseded_by_sha=MERGE_2,
        superseded_by_correlation_id=newer,
        reason=f"superseded by {newer} at {MERGE_2}",
    )

    document = json.loads((tmp_path / "jobs" / f"{older}.json").read_text())

    assert document["status"] == "superseded"
    assert document["superseded_by_correlation_id"] == str(newer)
    assert len(document["errors"]) == 1
    assert str(newer) in document["errors"][0]
