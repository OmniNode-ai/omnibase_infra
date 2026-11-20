# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the hook venv reconcile and its drift check through the real uv binary."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import zipfile
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

pytestmark = pytest.mark.integration
_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_reconcile_workspace_venvs/contract.yaml"
)


def _uv() -> str:
    uv = shutil.which("uv")
    assert uv is not None, "the real uv binary is required"
    return uv


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project whose only dependency is an optional extra, installed from a local wheel."""
    # The CI runner exports UV_PROJECT_ENVIRONMENT, which would send uv to another venv.
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_LINK_MODE", "copy")
    wheel = tmp_path / "probe-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("probe/__init__.py", "")
        archive.writestr(
            "probe-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: probe\nVersion: 1.0\n",
        )
        archive.writestr(
            "probe-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("probe-1.0.dist-info/RECORD", "")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "probeproj"\nversion = "0.1"\nrequires-python = ">=3.9"\n'
        "dependencies = []\n\n[project.optional-dependencies]\n"
        f'probeextra = ["probe @ {wheel.as_uri()}"]\n\n[tool.uv]\npackage = false\n'
    )
    for args in (["lock"], ["sync", "--frozen", "--extra", "probeextra"]):
        done = _run([_uv(), *args], tmp_path)
        assert done.returncode == 0, done.stderr
    return tmp_path


def _handler() -> HandlerReconcileWorkspaceVenvs:
    return asyncio.run(_bootstrap(_CONTRACT)).get_service(
        HandlerReconcileWorkspaceVenvs
    )


def _request(project: Path, mode: str) -> ModelHookVenvReconcileRequest:
    return ModelHookVenvReconcileRequest(
        mode=mode, workspace_root=project, project=project, uv_bin=Path(_uv())
    )


def _probe_installed(project: Path) -> bool:
    return any(project.glob(".venv/lib/python*/site-packages/probe-1.0.dist-info"))


def test_plain_frozen_check_reads_a_kept_extra_as_outdated(project: Path) -> None:
    """The control: the drift check this change replaces says outdated, forever."""
    assert _run([_uv(), "sync", "--frozen", "--check"], project).returncode != 0
    named = _run(
        [_uv(), "sync", "--frozen", "--check", "--extra", "probeextra"], project
    )
    assert named.returncode == 0


def test_reconcile_keeps_the_installed_extra_through_real_uv(project: Path) -> None:
    result = _handler().handle(_request(project, "reconcile"))
    assert result.status == "already_in_sync"
    assert _probe_installed(project)


def test_check_mode_reads_in_sync_through_real_uv(project: Path) -> None:
    result = _handler().handle(_request(project, "check"))
    assert result.status == "already_in_sync"


def test_check_mode_reads_outdated_when_the_lock_no_longer_lists_the_extra(
    project: Path,
) -> None:
    lock = project / "uv.lock"
    lock.write_text(
        "\n".join(
            line
            for line in lock.read_text().splitlines()
            if "provides-extras" not in line
        )
        + "\n"
    )
    result = _handler().handle(_request(project, "check"))
    assert result.status == "outdated"
    assert "extras not kept (not listed in uv.lock): probeextra" in result.output
