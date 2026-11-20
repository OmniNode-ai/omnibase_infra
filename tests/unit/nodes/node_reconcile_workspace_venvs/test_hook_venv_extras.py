# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keep lock-listed extras on the sync plan without weakening the refusal guard."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.handlers.hook_venv_reconcile.handler_reconcile_workspace_venvs import (
    HandlerReconcileWorkspaceVenvs,
)
from omnibase_infra.models.model_hook_venv_reconcile_request import (
    ModelHookVenvReconcileRequest,
)
from omnibase_infra.nodes.node_reconcile_workspace_venvs import (
    runtime_reconcile_workspace_venvs as runtime,
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


def _install(
    project: Path,
    *,
    commit: str = _BEFORE,
    duplicate: bool = False,
    metadata: str = "Name: probe\nVersion: 1.0\n",
) -> None:
    for distribution in ["probe", "probe-copy"] if duplicate else ["probe"]:
        info = project / f".venv/lib/python3.12/site-packages/{distribution}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(metadata)
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


def _project(
    project: Path, declared: tuple[str, ...], locked: tuple[str, ...] | None
) -> None:
    """Write a pyproject declaring extras and, unless locked is None, a lock listing some."""
    extras = "".join(f'{name} = ["probe>=1.0"]\n' for name in declared)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "probeproj"\n\n[project.optional-dependencies]\n' + extras
    )
    if locked is not None:
        listed = ", ".join(f'"{name}"' for name in locked)
        (project / "uv.lock").write_text(
            '[[package]]\nname = "probeproj"\n\n[package.metadata]\n'
            f"provides-extras = [{listed}]\n"
        )


def _extras_of(args: list[str]) -> list[str]:
    return [args[i + 1] for i, value in enumerate(args) if value == "--extra"]


def _uv_that_drops_unlisted(calls: list[list[str]]) -> object:
    """Behave like uv: the venv's extra package is dropped unless its extra is named."""

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        kept = "probeextra" in _extras_of(args)
        code = 0 if kept or "--check" not in args else 1
        return subprocess.CompletedProcess(
            args, code, " Would make no changes" if kept else " - probe==1.0", ""
        )

    return run


def test_installed_lock_listed_extra_is_kept_on_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project(tmp_path, ("probeextra",), ("probeextra",))
    _install(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _uv_that_drops_unlisted(calls))
    result = _handler().handle(_request(tmp_path))
    assert result.status == "already_in_sync"
    assert len(calls) == 2
    assert all(_extras_of(args) == ["probeextra"] for args in calls)


def test_extra_is_computed_once_for_plan_check_and_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project(tmp_path, ("probeextra",), ("probeextra",))
    _install(tmp_path)
    calls: list[list[str]] = []

    def writer_in_between(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        # Another writer changes the lock after the plan: a recompute would drop the extra.
        (tmp_path / "uv.lock").unlink(missing_ok=True)
        return subprocess.CompletedProcess(
            args, 1 if "--check" in args else 0, " Would make no changes", ""
        )

    monkeypatch.setattr(subprocess, "run", writer_in_between)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "reconciled"
    assert len(calls) == 3
    assert all(_extras_of(args) == ["probeextra"] for args in calls)


def test_stray_extra_is_not_kept_and_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project(tmp_path, ("probeextra",), ())
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
    assert all(not _extras_of(args) for args in calls)
    assert "extras not kept (not listed in uv.lock): probeextra" in result.output


def test_missing_lock_keeps_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project(tmp_path, ("probeextra",), None)
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
    assert all(not _extras_of(args) for args in calls)


def test_extra_listed_git_package_that_must_be_replaced_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project(tmp_path, ("probeextra",), ("probeextra",))
    _install(tmp_path)
    calls: list[list[str]] = []

    def replaced_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[0] == "git":
            return subprocess.CompletedProcess(args, 1, "", "")  # not an ancestor
        plan = f" - probe==1.0\n + probe==1.0 (from git+{tmp_path.as_uri()}@{'b' * 40})"
        return subprocess.CompletedProcess(args, 0, plan, "")

    monkeypatch.setattr(subprocess, "run", replaced_run)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "refused"
    assert "conflicting branch install would be replaced: probe" in result.detail
    assert [args for args in calls if args[0] != "git"] == calls[:1]
    assert "--dry-run" in calls[0]


def test_metadata_body_line_is_not_a_package_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project(tmp_path, ("probeextra",), ("probeextra",))
    _install(tmp_path, metadata="Name: other\nVersion: 1\n\nName: probe\n")
    calls: list[list[str]] = []

    def planned_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, " Would make no changes", "")

    monkeypatch.setattr(subprocess, "run", planned_run)
    result = _handler().handle(_request(tmp_path))
    assert result.status == "already_in_sync"
    assert all(not _extras_of(args) for args in calls)


@pytest.mark.parametrize(("code", "status"), [(0, "already_in_sync"), (1, "outdated")])
def test_check_mode_runs_one_check_with_the_kept_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, status: str
) -> None:
    _project(tmp_path, ("probeextra",), ("probeextra",))
    _install(tmp_path)
    calls: list[list[str]] = []

    def checked_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, code, "", "")

    monkeypatch.setattr(subprocess, "run", checked_run)
    request = _request(tmp_path).model_copy(update={"mode": "check"})
    result = _handler().handle(request)
    assert result.status == status
    assert len(calls) == 1
    assert "--check" in calls[0]
    assert "--dry-run" not in calls[0]
    assert _extras_of(calls[0]) == ["probeextra"]


@pytest.mark.parametrize(("code", "exit_code"), [(0, 0), (1, 1)])
def test_runtime_check_flag_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, exit_code: int
) -> None:
    _project(tmp_path, ("probeextra",), ("probeextra",))
    _install(tmp_path)

    def checked_run(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, code, "", "")

    monkeypatch.setattr(subprocess, "run", checked_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--workspace",
            str(tmp_path),
            "--project",
            str(tmp_path),
            "--uv",
            "/uv",
            "--contract",
            str(_CONTRACT),
            "--check",
        ],
    )
    assert runtime.main() == exit_code


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
