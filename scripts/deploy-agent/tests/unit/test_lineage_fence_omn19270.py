# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent refuses a command whose ref is behind the running build (OMN-19270).

THE INCIDENT
------------
On 2026-09-23 the .201 dev lane was running infra ``0edf5c914``. Job
``e074126b`` carried ``git_ref=533b19c2``, an ancestor of that build, and the
agent accepted it and recreated the lane onto it at 16:01:15Z. The lane went
backwards. The accept protocol had no step that compared a command's ref with
what the lane was running.

WHAT THESE TESTS PIN
--------------------
* AC1: a strict ancestor of the running build is refused with the typed
  ``superseded_by_running_build`` reason, acknowledged past its offset, and no
  job is created. The incident's own shas are used.
* AC2: a strict descendant is accepted. This is checked once against a
  hand-built ancestry and once against a real git repository, so a fence that
  compared in the wrong direction fails.
* AC3: an equal ref is handled by the existing duplicate path. The same
  correlation id is refused ``duplicate``. A new correlation id at the same
  infra ref is a sibling-triggered rebuild, and it is accepted.
* The owner's design constraints: a ref that has diverged and is not on the
  tracking branch is refused with its own reason, and a signed rollback
  declaration gets through, and is never folded away by coalescing.
* Every comparison the host cannot make lets the command run as before.
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
    EnumRejectionReason,
    EnumRuntimeLane,
    ModelRebuildRequested,
    ModelRejectionNotice,
    ModelRollbackDeclaration,
)
from deploy_agent.lineage_fence import (
    BUILD_PROVENANCE_PATH,
    DockerProvenanceReader,
    EnumLineageVerdict,
    decide_lineage,
)

TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
TRACKING = "origin/dev"

#: The incident: the lane ran RUNNING, the replayed command asked for STALE.
STALE = "533b19c23b630067d8b53855ce1bd96392aa4412"
RUNNING = "0edf5c9145876dbe22f6caf7a06b507c5d0fc7d3"
#: A later dev commit, a descendant of RUNNING.
NEWER = "6fe05c3567d8fb546a821f50ae891869e9494da1"
#: A commit off the tracking branch, branched from STALE.
FEATURE = "f" * 40

#: child -> parent. STALE <- RUNNING <- NEWER is the tracking branch, and
#: FEATURE branches off STALE.
_PARENT: dict[str, str | None] = {
    STALE: None,
    RUNNING: STALE,
    NEWER: RUNNING,
    FEATURE: STALE,
}
_REFS = {TRACKING: NEWER}


def _ancestors_or_self(sha: str) -> set[str]:
    seen: set[str] = set()
    cursor: str | None = _REFS.get(sha, sha)
    while cursor is not None:
        seen.add(cursor)
        cursor = _PARENT[cursor]
    return seen


def contains(earlier: str, later: str) -> bool:
    """``git merge-base --is-ancestor earlier later``, over the graph above."""
    return _REFS.get(earlier, earlier) in _ancestors_or_self(later)


def _command(
    git_ref: str,
    *,
    correlation_id: str | None = None,
    rollback: ModelRollbackDeclaration | None = None,
    **overrides: Any,
) -> ModelRebuildRequested:
    fields: dict[str, Any] = {
        "correlation_id": correlation_id or str(uuid4()),
        "requested_by": "gha/omnimarket/pr-2804",
        "scope": "full",
        "runtime_lane": "dev",
        "build_source": "workspace",
        "git_ref": git_ref,
        "rollback": rollback,
    }
    fields.update(overrides)
    return ModelRebuildRequested.model_validate(fields)


def _message(cmd: ModelRebuildRequested, offset: int = 100) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json") | {"_signature": "a" * 64}
    return SimpleNamespace(
        value=payload, topic=TOPIC, partition=0, offset=offset, key=None
    )


def _consumer(
    running: str | None,
    *,
    resolver: Any = contains,
    duplicate: bool = False,
) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = duplicate
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    consumer.ancestry_resolver = resolver
    consumer.running_build_ref = Mock(return_value=running)
    consumer.tracking_ref = TRACKING
    consumer.notices = []
    consumer.on_rejected = consumer.notices.append
    return consumer


def _process(
    consumer: DeployConsumer, cmd: ModelRebuildRequested
) -> tuple[ModelRebuildRequested | None, str | None]:
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        return consumer._process_message(_message(cmd))


def _committed(consumer: DeployConsumer) -> list[int]:
    return [
        meta.offset
        for call in consumer.consumer.commit.call_args_list
        for meta in call.args[0].values()
    ]


# ---------------------------------------------------------------------------
# AC1: a strict ancestor is refused, typed, acknowledged, and never becomes a job.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_incident_command_is_refused_as_superseded_by_the_running_build() -> None:
    consumer = _consumer(RUNNING)
    cmd = _command(STALE)

    accepted, reason = _process(consumer, cmd)

    assert accepted is None
    assert reason == EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD.value
    consumer.job_store.accept.assert_not_called()
    # Acknowledged: committed past its own record, so it blocks nothing.
    assert _committed(consumer) == [101]
    assert consumer.notices == [
        ModelRejectionNotice(
            reason=EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD,
            correlation_id=cmd.correlation_id,
            scope=cmd.scope,
        )
    ]
    consumer.running_build_ref.assert_called_once_with(EnumRuntimeLane.DEV)


# ---------------------------------------------------------------------------
# AC2: a strict descendant runs.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_descendant_of_the_running_build_is_accepted() -> None:
    consumer = _consumer(RUNNING)
    cmd = _command(NEWER)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd
    consumer.job_store.accept.assert_called_once()
    assert consumer.notices == []


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.unit
def test_direction_against_a_real_repository(tmp_path: Path) -> None:
    """The fence and the real resolver agree on direction, on real commits.

    ``old <- mid <- new`` on ``dev``, with ``side`` branched from ``old``. A
    fence that swapped its arguments would accept ``old`` and refuse ``new``.
    """
    repo = tmp_path / "clone"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    shas: dict[str, str] = {}
    for name in ("old", "mid", "new"):
        _git(repo, "commit", "-q", "--allow-empty", "-m", name)
        shas[name] = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/dev", shas["new"])
    _git(repo, "checkout", "-q", "-b", "side", shas["old"])
    _git(repo, "commit", "-q", "--allow-empty", "-m", "side")
    shas["side"] = _git(repo, "rev-parse", "HEAD")

    resolver = GitAncestryResolver(str(repo))

    def verdict(requested: str, running: str) -> EnumLineageVerdict:
        return decide_lineage(
            _command(requested),
            read_running_ref=lambda: running,
            contains=resolver,
            tracking_ref=TRACKING,
        ).verdict

    assert verdict(shas["old"], shas["mid"]) is EnumLineageVerdict.STALE_ANCESTOR
    assert verdict(shas["new"], shas["mid"]) is EnumLineageVerdict.DESCENDANT
    assert verdict(shas["side"], shas["mid"]) is EnumLineageVerdict.DIVERGENT
    assert verdict(shas["new"], shas["side"]) is EnumLineageVerdict.RETURNS_TO_TRACKING


# ---------------------------------------------------------------------------
# AC3: an equal ref goes through the existing duplicate path, and only that.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_an_equal_ref_already_seen_is_the_established_duplicate() -> None:
    consumer = _consumer(RUNNING, duplicate=True)

    accepted, reason = _process(consumer, _command(RUNNING))

    assert accepted is None
    assert reason == EnumRejectionReason.DUPLICATE.value
    consumer.job_store.accept.assert_not_called()


@pytest.mark.unit
def test_an_equal_ref_under_a_new_correlation_id_still_runs() -> None:
    """A sibling-triggered rebuild publishes the infra HEAD the lane already runs.

    Its new code is the sibling's, staged at build time, so refusing an equal
    infra ref would strand every sibling merge that lands while infra is quiet.
    """
    consumer = _consumer(RUNNING)
    cmd = _command(RUNNING)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd
    assert (
        decide_lineage(
            cmd,
            read_running_ref=lambda: RUNNING,
            contains=contains,
            tracking_ref=TRACKING,
        ).verdict
        is EnumLineageVerdict.EQUAL
    )


# ---------------------------------------------------------------------------
# Divergence and deliberate rollback.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_diverged_ref_off_the_tracking_branch_is_refused_with_its_own_reason() -> (
    None
):
    consumer = _consumer(RUNNING)

    accepted, reason = _process(consumer, _command(FEATURE))

    assert accepted is None
    assert reason == EnumRejectionReason.DIVERGENT_REF.value
    consumer.job_store.accept.assert_not_called()
    assert _committed(consumer) == [101]


@pytest.mark.unit
def test_a_lane_on_an_off_branch_build_returns_to_its_tracking_branch() -> None:
    """After a build from off the branch, the next dev deploy is not locked out."""
    consumer = _consumer(FEATURE)
    cmd = _command(NEWER)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd


@pytest.mark.unit
def test_divergence_stands_when_the_tracking_check_cannot_be_answered() -> None:
    def partial(earlier: str, later: str) -> bool | None:
        if later == TRACKING:
            return None
        return contains(earlier, later)

    decision = decide_lineage(
        _command(FEATURE),
        read_running_ref=lambda: RUNNING,
        contains=partial,
        tracking_ref=TRACKING,
    )

    assert decision.verdict is EnumLineageVerdict.DIVERGENT


@pytest.mark.unit
def test_a_declared_rollback_to_an_ancestor_is_accepted() -> None:
    consumer = _consumer(RUNNING)
    cmd = _command(
        STALE,
        rollback=ModelRollbackDeclaration(
            actor="operator", reason="0edf5c914 broke the projection writers"
        ),
    )

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd
    consumer.job_store.accept.assert_called_once()
    # A rollback is decided on the declaration alone; the lane is not read.
    consumer.running_build_ref.assert_not_called()


@pytest.mark.unit
def test_a_declared_rollback_to_a_diverged_ref_is_accepted() -> None:
    consumer = _consumer(RUNNING)
    cmd = _command(
        FEATURE, rollback=ModelRollbackDeclaration(actor="lane-x", reason="bisect")
    )

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd


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
    plain = _command(RUNNING)
    rolled = _command(STALE, rollback=rollback)
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
# What the fence does not touch, and what it does when it cannot answer.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_an_unreadable_running_build_lets_the_command_run() -> None:
    """A lane that is down has no container to read, and must accept a rebuild."""
    consumer = _consumer(None)
    cmd = _command(STALE)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd


@pytest.mark.unit
def test_an_unanswerable_ancestry_lets_the_command_run() -> None:
    consumer = _consumer(RUNNING, resolver=lambda earlier, later: None)
    cmd = _command(STALE)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"git_ref": "origin/dev"},
        {"image_digest": "sha256:" + "0" * 64},
    ],
)
def test_aliases_and_pinned_images_never_read_the_lane(
    overrides: dict[str, str],
) -> None:
    consumer = _consumer(RUNNING)
    git_ref = overrides.pop("git_ref", STALE)
    cmd = _command(git_ref, **overrides)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd
    consumer.running_build_ref.assert_not_called()


@pytest.mark.unit
def test_a_consumer_without_a_reader_runs_no_fence() -> None:
    """The pre-change behaviour, reached for every consumer built without one."""
    consumer = _consumer(RUNNING)
    consumer.running_build_ref = None
    cmd = _command(STALE)

    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd


# ---------------------------------------------------------------------------
# The running build is read from the image's own provenance.
# ---------------------------------------------------------------------------


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


@pytest.mark.unit
def test_the_reader_returns_infra_vcs_ref_from_the_lane_container() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(
            0, json.dumps({"build_source": "workspace", "infra_vcs_ref": RUNNING})
        )

    reader = DockerProvenanceReader(lambda lane: "omninode-runtime", run=run)

    assert reader(EnumRuntimeLane.DEV) == RUNNING
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
        _completed(0, json.dumps({"infra_vcs_ref": RUNNING[:12]})),
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
        git_ref=STALE,
        runtime_lane=EnumRuntimeLane.DEV,
        scope=_command(STALE).scope,
        build_source=_command(STALE).build_source,
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
            STALE,
            "--runtime-lane",
            "dev",
            "--rollback-actor",
            "operator",
            "--dry-run",
        ]
    )

    assert code == 1
    assert "--rollback-reason" in capsys.readouterr().err
