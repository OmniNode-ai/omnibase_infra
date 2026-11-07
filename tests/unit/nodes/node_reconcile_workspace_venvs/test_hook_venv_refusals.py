# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Refuse unreadable installer intent and bounded host operations before sync."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.handlers.hook_venv_reconcile.handler_reconcile_workspace_venvs import (
    HandlerReconcileWorkspaceVenvs,
)
from omnibase_infra.models.model_hook_venv_reconcile_request import (
    ModelHookVenvReconcileRequest,
)
from omnibase_infra.nodes.node_reconcile_workspace_venvs.runtime_reconcile_workspace_venvs import (
    _bootstrap,
)

pytestmark = pytest.mark.unit
_CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/nodes/node_reconcile_workspace_venvs/contract.yaml"
)
_BEFORE = "a" * 40
_AFTER = "b" * 40


def _install(project: Path, *, commit: str = _BEFORE, duplicate: bool = False) -> None:
    for distribution in ["probe", "probe-copy"] if duplicate else ["probe"]:
        info = project / f".venv/lib/python3.12/site-packages/{distribution}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text("Name: probe\nVersion: 1.0\n")
        (info / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": project.as_uri(),
                    "vcs_info": {"vcs": "git", "commit_id": commit},
                }
            )
        )


def _handler() -> HandlerReconcileWorkspaceVenvs:
    return asyncio.run(_bootstrap(_CONTRACT)).get_service(
        HandlerReconcileWorkspaceVenvs
    )


def _request(project: Path) -> ModelHookVenvReconcileRequest:
    return ModelHookVenvReconcileRequest(
        workspace_root=project, project=project, uv_bin=Path("/uv")
    )


@pytest.mark.parametrize("stage", ["plan", "check", "sync", "ancestry"])
def test_process_timeout_is_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    _install(tmp_path)
    calls: list[list[str]] = []

    def bounded_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        is_git = args[0] == "git"
        selected = (
            (stage == "plan" and "--dry-run" in args)
            or (stage == "check" and "--check" in args)
            or (stage == "sync" and not is_git and args[-1] == str(tmp_path))
            or (stage == "ancestry" and is_git)
        )
        assert kwargs["timeout"] == (30 if is_git else 600)
        if selected:
            raise subprocess.TimeoutExpired(args, float(str(kwargs["timeout"])))
        plan = (
            f" - probe==1.0\n + probe==1.0 (from git+{tmp_path.as_uri()}@{_AFTER})"
            if stage == "ancestry"
            else "Would make no changes"
        )
        return subprocess.CompletedProcess(
            args, 1 if "--check" in args else 0, plan, ""
        )

    monkeypatch.setattr(subprocess, "run", bounded_run)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "refused"
    assert "timed out" in result.detail
    assert len(calls) == {"plan": 1, "check": 2, "sync": 3, "ancestry": 2}[stage]


@pytest.mark.parametrize("git_status", [2, 128])
def test_ancestry_error_never_reaches_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_status: int
) -> None:
    _install(tmp_path)
    calls: list[list[str]] = []

    def unreadable_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        plan = f" - probe==1.0\n + probe==1.0 (from git+{tmp_path.as_uri()}@{_AFTER})"
        return subprocess.CompletedProcess(
            args, git_status if args[0] == "git" else 0, plan, ""
        )

    monkeypatch.setattr(subprocess, "run", unreadable_run)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "refused"
    assert "ancestry is unreadable" in result.detail
    assert len(calls) == 2
    assert calls[-1][0] == "git"


@pytest.mark.parametrize("defect", ["short-commit", "duplicate-name"])
def test_ambiguous_installer_metadata_never_invokes_uv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    _install(
        tmp_path,
        commit="abc" if defect == "short-commit" else _BEFORE,
        duplicate=defect == "duplicate-name",
    )

    def forbidden_run(*args: object, **kwargs: object) -> None:
        pytest.fail("unreadable installer metadata must refuse before any process")

    monkeypatch.setattr(subprocess, "run", forbidden_run)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "refused"
    assert "unreadable or ambiguous Git revision" in result.detail


def test_tilde_plan_cannot_approve_a_git_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(tmp_path)
    calls: list[list[str]] = []

    def plan_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, " ~ probe==1.0", "")

    monkeypatch.setattr(subprocess, "run", plan_run)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "refused"
    assert "conflicting branch install would be replaced: probe" in result.detail
    assert len(calls) == 1
    assert "--dry-run" in calls[0]
