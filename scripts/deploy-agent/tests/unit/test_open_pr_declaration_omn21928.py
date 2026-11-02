# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane builds a declared open pull request's branch, for that repository only (OMN-21928).

The signed rebuild command carries ``open_pr`` (``<repo>#<number>@<head sha>``) beside
the rollback declaration a deliberate off-lineage deploy already needs. The executor
turns it into the first-party audit's build context and declaration build arguments,
reading the policy data from the clone being built; the stability and prod lanes
refuse the command, and judge and collaborator are not deploy-agent lanes at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.coalesce import (
    EnumCoalesceRefusal,
    ModelQueuedCommand,
    plan_coalesce,
)
from deploy_agent.events import (
    BuildSource,
    EnumLineageVerdict,
    EnumRuntimeLane,
    ModelRebuildRequested,
    ModelRollbackDeclaration,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from deploy_agent.lineage_fence import ModelRunningBuild, decide_lineage
from deploy_agent.trigger import main as trigger_main
from pydantic import ValidationError

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
PR_HEAD = "a1" * 20
DECLARED = f"omnibase_infra#5564@{PR_HEAD}"
TRACKING = "origin/dev"
RUNNING = "b2" * 20
ROLLBACK = ModelRollbackDeclaration(actor="slot-fix", reason="build open PR 5564")


def _command(lane: str = "dev", **overrides: object) -> ModelRebuildRequested:
    fields: dict[str, object] = {
        "correlation_id": UUID(int=21928),
        "requested_by": "operator-manual",
        "scope": "runtime",
        "runtime_lane": lane,
        "build_source": "workspace",
        "git_ref": PR_HEAD,
        "open_pr": DECLARED,
        "rollback": ROLLBACK,
    }
    fields.update(overrides)
    return ModelRebuildRequested.model_validate(fields)


def _noop(phase: Phase, status: PhaseStatus) -> None:
    pass


@pytest.fixture(autouse=True)
def _checker_from_this_clone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The executor reads the policy and checker from the clone it builds."""
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(REPO_ROOT))


def test_the_dev_lane_command_carries_the_declaration() -> None:
    cmd = _command()
    assert cmd.open_pr == DECLARED
    assert cmd.runtime_lane is EnumRuntimeLane.DEV
    assert _command(open_pr="", rollback=None).open_pr == ""


@pytest.mark.parametrize(
    "text", ["5564", "omnibase_infra#0@" + "a" * 40, "omnibase_infra#5@abc"]
)
def test_a_malformed_declaration_is_refused(text: str) -> None:
    with pytest.raises(ValidationError, match="open_pr"):
        _command(open_pr=text)


@pytest.mark.parametrize(
    "lane", [EnumRuntimeLane.STABILITY_TEST.value, EnumRuntimeLane.PROD.value]
)
def test_stability_and_prod_commands_refuse_a_branch_build(lane: str) -> None:
    extra = {"image_digest": "sha256:" + "0" * 64} if lane == "prod" else {}
    with pytest.raises(ValidationError, match="open_pr needs runtime_lane dev"):
        _command(lane, **extra)


@pytest.mark.parametrize("lane", ["judge", "collaborator"])
def test_judge_and_collaborator_are_not_lanes_a_command_can_name(lane: str) -> None:
    with pytest.raises(ValidationError):
        _command(lane)


def test_the_declaration_must_name_the_commit_built_and_carry_a_rollback() -> None:
    with pytest.raises(ValidationError, match="open_pr needs runtime_lane dev"):
        _command(git_ref="c4" * 20)
    with pytest.raises(ValidationError, match="open_pr needs runtime_lane dev"):
        _command(git_ref=TRACKING)
    with pytest.raises(ValidationError, match="open_pr needs runtime_lane dev"):
        _command(rollback=None)


def _build_commands(executor: DeployExecutor, lane: EnumRuntimeLane) -> list[list[str]]:
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], timeout: int, **kwargs: object) -> object:
        captured.append(cmd)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        executor._compose_build(
            Scope.RUNTIME,
            "abc1234def56",
            _noop,
            build_source=BuildSource.RELEASE,
            runtime_lane=lane,
        )
    return [cmd for cmd in captured if "build" in cmd]


def test_the_dev_lane_build_names_the_open_pr_context_and_declaration() -> None:
    executor = DeployExecutor()
    executor.declared_pr = DECLARED

    (build_cmd,) = _build_commands(executor, EnumRuntimeLane.DEV)

    assert "FIRST_PARTY_BUILD_CONTEXT=dev-lane-open-pr" in build_cmd
    assert f"FIRST_PARTY_DECLARED_PR={DECLARED}" in build_cmd


def test_a_build_without_a_declaration_names_no_audit_context() -> None:
    """Positive control: the unchanged dev command audits in the default context."""
    (build_cmd,) = _build_commands(DeployExecutor(), EnumRuntimeLane.DEV)

    assert not [arg for arg in build_cmd if arg.startswith("FIRST_PARTY_")]


@pytest.mark.parametrize("lane", [EnumRuntimeLane.STABILITY_TEST, EnumRuntimeLane.PROD])
def test_the_executor_refuses_a_declaration_on_any_other_lane_before_building(
    lane: EnumRuntimeLane,
) -> None:
    executor = DeployExecutor()
    executor.declared_pr = DECLARED
    ran: list[list[str]] = []

    with (
        patch(
            "deploy_agent.executor._run",
            side_effect=lambda cmd, *a, **k: ran.append(cmd),
        ),
        pytest.raises(ValueError, match="cannot build a declared open pull request"),
    ):
        executor._compose_build(
            Scope.RUNTIME,
            "abc1234def56",
            _noop,
            build_source=BuildSource.RELEASE,
            runtime_lane=lane,
        )

    assert ran == []


def test_rebuild_scope_hands_the_declaration_to_every_build_and_clears_it() -> None:
    executor = DeployExecutor()
    seen: list[str] = []

    def fake_build(scope: Scope, sha: str, cb: object, **kwargs: object) -> None:
        seen.append(executor.declared_pr)

    def fake_up(*args: object, **kwargs: object) -> None:
        return None

    executor._compose_build = fake_build
    executor._compose_up = fake_up

    executor.rebuild_scope(
        Scope.FULL, [], _noop, git_sha="abc1234def56", declared_pr=DECLARED
    )
    assert seen
    assert all(item == DECLARED for item in seen)

    seen.clear()
    executor.rebuild_scope(Scope.FULL, [], _noop, git_sha="abc1234def56")
    assert seen
    assert all(item == "" for item in seen)


def test_the_lineage_fence_builds_the_declared_head_as_asked() -> None:
    """The open-PR head is off the lane's lineage; the rollback declaration it carries is the way."""
    decision = decide_lineage(
        _command(),
        read_running_build=lambda: ModelRunningBuild(infra_ref=RUNNING),
        contains=lambda earlier, later: earlier == later,
        resolve_ref=None,
        tracking_ref=TRACKING,
        published_at=None,
    )
    assert decision.verdict is EnumLineageVerdict.ROLLBACK_DECLARED
    assert decision.build_ref == PR_HEAD
    # Control: the same off-lineage head with no declaration at all is refused.
    undeclared = decide_lineage(
        _command(open_pr="", rollback=None),
        read_running_build=lambda: ModelRunningBuild(infra_ref=RUNNING),
        contains=lambda earlier, later: earlier == later,
        resolve_ref=None,
        tracking_ref=TRACKING,
        published_at=None,
    )
    assert undeclared.verdict is EnumLineageVerdict.DIVERGENT


def test_the_next_tracking_command_returns_the_lane_to_main_lineage() -> None:
    """After a declared branch build the lane runs the PR head; a dev merge brings it home."""
    dev_head = "d3" * 20

    def contains(earlier: str, later: str) -> bool:
        # The new dev head is on the tracking branch; the PR head is not an ancestor of it.
        return earlier == later or (earlier == dev_head and later == TRACKING)

    decision = decide_lineage(
        _command(open_pr="", rollback=None, git_ref=dev_head),
        read_running_build=lambda: ModelRunningBuild(infra_ref=PR_HEAD),
        contains=contains,
        resolve_ref=None,
        tracking_ref=TRACKING,
        published_at=None,
    )
    assert decision.verdict is EnumLineageVerdict.RETURNS_TO_TRACKING
    assert decision.build_ref == dev_head


@pytest.mark.parametrize("declared_first", [True, False])
def test_coalescing_never_folds_a_declared_open_pr(declared_first: bool) -> None:
    plain = _command(
        open_pr="",
        rollback=None,
        git_ref="d3" * 20,
        scope="full",
        correlation_id=UUID(int=2),
    )
    declared = _command(scope="full")
    head, candidate = (declared, plain) if declared_first else (plain, declared)

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


def test_the_trigger_signs_the_declaration_into_the_command(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", "test-secret")

    code = trigger_main(
        [
            "--runtime-lane",
            "dev",
            "--git-ref",
            PR_HEAD,
            "--open-pr",
            DECLARED,
            "--rollback-actor",
            "slot-fix",
            "--rollback-reason",
            "build open PR 5564",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert f"git_ref:        {PR_HEAD}" in out
    assert f'"open_pr": "{DECLARED}"' in out


def test_the_trigger_refuses_a_malformed_declaration_and_a_non_dev_lane(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", "test-secret")

    assert (
        trigger_main(["--runtime-lane", "dev", "--open-pr", "5564", "--dry-run"]) == 1
    )
    assert "open_pr" in capsys.readouterr().err

    code = trigger_main(
        [
            "--runtime-lane",
            EnumRuntimeLane.STABILITY_TEST.value,
            "--git-ref",
            PR_HEAD,
            "--open-pr",
            DECLARED,
            "--rollback-actor",
            "slot-fix",
            "--rollback-reason",
            "build open PR 5564",
            "--dry-run",
        ]
    )
    assert code == 1
    assert "open_pr needs runtime_lane dev" in capsys.readouterr().err
