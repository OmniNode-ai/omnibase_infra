# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keep lock-listed extras on the sync plan without weakening the refusal guard."""

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


def test_installed_extra_is_kept_on_the_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project.optional-dependencies]\nprobeextra = ["probe>=1.0"]\n'
    )
    _install(tmp_path)
    calls: list[list[str]] = []

    def uv_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        # uv drops a package the project's default install does not list, unless its extra is named.
        calls.append(args)
        kept = "--extra" in args and args[args.index("--extra") + 1] == "probeextra"
        plan = " Would make no changes" if kept else " - probe==1.0"
        return subprocess.CompletedProcess(args, 0, plan, "")

    monkeypatch.setattr(subprocess, "run", uv_run)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "already_in_sync"
    assert calls
    assert all("--extra" in args for args in calls)


def test_extra_with_a_missing_requirement_is_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project.optional-dependencies]\npartial = ["probe>=1.0", "absent>=1"]\n'
    )
    _install(tmp_path)
    calls: list[list[str]] = []

    def planned_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, " Would make no changes", "")

    monkeypatch.setattr(subprocess, "run", planned_run)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "already_in_sync"
    assert calls
    for args in calls:
        assert "--extra" not in args


def test_no_pyproject_means_no_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def planned_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, " Would make no changes", "")

    monkeypatch.setattr(subprocess, "run", planned_run)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "already_in_sync"
    assert calls
    for args in calls:
        assert "--extra" not in args


def test_unlisted_git_install_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(tmp_path)
    calls: list[list[str]] = []

    def planned_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, " - probe==1.0", "")

    monkeypatch.setattr(subprocess, "run", planned_run)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "refused"
    assert "conflicting branch install would be replaced: probe" in result.detail
    assert len(calls) == 1
    assert "--dry-run" in calls[0]
