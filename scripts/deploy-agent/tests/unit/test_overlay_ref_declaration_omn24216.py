# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The signed rebuild command carries an optional overlays revision, and the job mounts it (OMN-24216).

``overlay_ref`` is the 40-hex head of an overlays repository revision. A command without it
builds exactly as before; the model accepts it on the dev lane with a rollback declaration
and refuses it anywhere else; and the dev job renders it before any build, so a revision the
host copy lacks ends the job with nothing built. The job tests run the agent's own
``_execute_command`` with the executor stubbed, and read the environment at the moment the
build step runs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.auth import sign_envelope
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    ModelRollbackDeclaration,
    Scope,
)
from deploy_agent.executor import load_dev_boot_guard
from pydantic import ValidationError

from tests.unit.test_auto_rollback import _agent

pytestmark = pytest.mark.unit

MODULE = "nodes.node_setup_local_provision_effect.runtime_overlay_revision"
ROLLBACK = ModelRollbackDeclaration(actor="slot-fix", reason="mount overlays revision")
ABSENT = "0" * 40


def _declared() -> dict[str, str]:
    return {
        k: str(v)
        for k, v in load_dev_boot_guard(MODULE).declaration(executor_mod).items()
    }


def _command(
    lane: str = EnumRuntimeLane.DEV.value, **fields: object
) -> ModelRebuildRequested:
    return ModelRebuildRequested.model_validate(
        {
            "correlation_id": str(uuid4()),
            "requested_by": "test/omn24216",
            "scope": "runtime",
            "runtime_lane": lane,
            "git_ref": "a" * 40,
            **fields,
        }
    )


def test_a_command_without_the_field_defaults_to_no_revision() -> None:
    assert _command().overlay_ref == ""


def test_the_dev_lane_with_a_rollback_declaration_accepts_a_revision() -> None:
    assert _command(overlay_ref="ab" * 20, rollback=ROLLBACK).overlay_ref == "ab" * 20


def test_the_stability_lane_refuses_a_revision() -> None:
    with pytest.raises(ValidationError, match="overlay_ref needs runtime_lane dev"):
        _command(
            EnumRuntimeLane.STABILITY_TEST.value,
            overlay_ref="ab" * 20,
            rollback=ROLLBACK,
        )


def test_the_dev_lane_without_a_rollback_declaration_refuses_a_revision() -> None:
    with pytest.raises(ValidationError, match="overlay_ref needs runtime_lane dev"):
        _command(overlay_ref="ab" * 20)


@pytest.mark.parametrize("revision", ["not-a-sha", "AB" * 20, "ab" * 19])
def test_a_malformed_revision_is_refused(revision: str) -> None:
    with pytest.raises(ValidationError, match="overlay_ref"):
        _command(overlay_ref=revision, rollback=ROLLBACK)


def test_an_unrelated_extra_field_is_still_refused() -> None:
    with pytest.raises(ValidationError, match="overlay_refs"):
        _command(overlay_refs="ab" * 20, rollback=ROLLBACK)


@pytest.fixture
def overlays_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A real overlays clone on the host, named by the contract's clone variable; its head."""
    declared = _declared()
    repo = tmp_path / "overlays-clone"
    (repo / declared["overlay_dir"]).mkdir(parents=True)
    (repo / declared["overlay_dir"] / "lane.yaml").write_text("mode: on_demand\n")
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run(
        [*git, "-c", "user.email=lane@example.invalid", "-c", "user.name=lane"]
        + ["commit", "-q", "-m", "seed"],
        check=True,
    )
    monkeypatch.setenv(declared["clone_env"], str(repo))
    monkeypatch.delenv(declared["store_root_env"], raising=False)
    return subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _run_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cmd: ModelRebuildRequested
) -> tuple[object, list[str | None], object]:
    """Run the dev job; return its record, the store root seen at each build, the executor."""
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    monkeypatch.setattr(agent, "_self_update_post_terminal", lambda: None)
    seen: list[str | None] = []

    def build(*args: object, **kwargs: object) -> list[str]:
        seen.append(os.environ.get(_declared()["store_root_env"]))
        return ["omninode-runtime"]

    executor.rebuild_scope.side_effect = build
    agent._execute_command(cmd)
    return agent.job_store.load(cmd.correlation_id), seen, executor


def test_golden_a_dev_job_builds_with_the_revision_named_as_the_store_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlays_clone: str
) -> None:
    cmd = _command(overlay_ref=overlays_clone, rollback=ROLLBACK)
    record, seen, _ = _run_job(tmp_path, monkeypatch, cmd)
    declared = _declared()
    expected = (
        tmp_path
        / "agent-state"
        / declared["rendered_dir_name"]
        / overlays_clone
        / declared["overlay_dir"]
    )
    assert record.status == "success"
    assert seen == [str(expected)]
    assert (expected / "lane.yaml").read_text() == "mode: on_demand\n"


def test_golden_the_next_dev_job_without_a_revision_returns_to_the_host_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlays_clone: str
) -> None:
    _run_job(
        tmp_path, monkeypatch, _command(overlay_ref=overlays_clone, rollback=ROLLBACK)
    )
    _, seen, _ = _run_job(tmp_path, monkeypatch, _command())
    assert seen == [None]


def test_control_a_command_with_no_revision_never_sets_a_store_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlays_clone: str
) -> None:
    """AC1's failing control: an override here would name a store root nobody asked for."""
    record, seen, _ = _run_job(tmp_path, monkeypatch, _command())
    assert record.status == "success"
    assert seen == [None]
    assert not (tmp_path / "agent-state" / _declared()["rendered_dir_name"]).exists()


def test_error_a_revision_the_host_copy_lacks_ends_the_job_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlays_clone: str
) -> None:
    cmd = _command(overlay_ref=ABSENT, rollback=ROLLBACK)
    record, seen, executor = _run_job(tmp_path, monkeypatch, cmd)
    assert record.status == "failed"
    assert any("holds no" in error for error in record.errors)
    assert seen == []
    executor.rebuild_scope.assert_not_called()
    executor.git_pull.assert_not_called()


def test_seam_the_signers_command_reaches_the_jobs_store_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlays_clone: str
) -> None:
    """Signer envelope -> command model -> agent render, on the wire form.

    The envelope below is the wire shape ``node_branch_deploy_command_effect`` (omnimarket)
    builds and signs: the same keys, the same ``overlay_ref`` name. Fails when either end
    renames the field: the node's envelope key, the model's field, or the job's read of it.
    """
    built = {
        "correlation_id": str(uuid4()),
        "requested_by": "test/omn24216",
        "scope": Scope.RUNTIME.value,
        "runtime_lane": EnumRuntimeLane.DEV.value,
        "build_source": BuildSource.WORKSPACE.value,
        "git_ref": "a" * 40,
        "rollback": {"actor": ROLLBACK.actor, "reason": ROLLBACK.reason},
        "overlay_ref": overlays_clone,
    }
    envelope = sign_envelope(built, "0" * 64)
    wire = json.loads(json.dumps(envelope, separators=(",", ":")))
    cmd = ModelRebuildRequested.model_validate(
        {k: v for k, v in wire.items() if k != "_signature"}
    )
    record, seen, _ = _run_job(tmp_path, monkeypatch, cmd)
    declared = _declared()
    expected = (
        tmp_path
        / "agent-state"
        / declared["rendered_dir_name"]
        / overlays_clone
        / declared["overlay_dir"]
    )
    assert record.status == "success"
    assert seen == [str(expected)]
