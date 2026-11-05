# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Replay the base-only hook sync that missed intelligence 0.25.3.

Only the hook project's uv operations are real; other workspace surfaces use
the existing offline harness. The wheel and lock are local, with no network.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_workspace_venvs import _SCRIPT, _Workspace

pytestmark = pytest.mark.unit


def _run(
    *argv: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv, env=env, capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture
def hook_workspace(tmp_path: Path) -> _Workspace:
    workspace = _Workspace(tmp_path / "workspace")
    project = workspace.omniclaude
    wheelhouse = project / "wheels"
    wheelhouse.mkdir()
    dist = "omninode_intelligence-0.25.3.dist-info"
    with zipfile.ZipFile(
        wheelhouse / "omninode_intelligence-0.25.3-py3-none-any.whl", "w"
    ) as wheel:
        wheel.writestr(
            f"{dist}/METADATA",
            "Metadata-Version: 2.1\nName: omninode-intelligence\nVersion: 0.25.3\n",
        )
        wheel.writestr(
            f"{dist}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr(f"{dist}/RECORD", "")
    (project / "pyproject.toml").write_text(
        '[project]\nname = "omninode-claude"\nversion = "0.1.0"\n'
        'requires-python = ">=3.12"\ndependencies = []\n'
        '[project.optional-dependencies]\nintelligence = ["omninode-intelligence==0.25.3"]\n'
        'full = ["omninode-intelligence==0.25.3"]\n'
        "[tool.uv]\npackage = false\n",
        encoding="utf-8",
    )
    uv = shutil.which("uv")
    assert uv, "uv is required to reproduce the recorded sync behavior"
    env = workspace.env()
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    # The inherited process may itself run inside a venv for another project.
    env.pop("VIRTUAL_ENV", None)
    (project / "uv.lock").unlink()
    shutil.rmtree(project / ".venv")
    _run(
        uv,
        "lock",
        "--offline",
        "--project",
        str(project),
        "--find-links",
        str(wheelhouse),
        "--python",
        sys.executable,
        env=env,
    )
    _run(uv, "venv", str(project / ".venv"), "--python", sys.executable, env=env)
    _run(uv, "sync", "--frozen", "--offline", "--project", str(project), env=env)
    # Positive control: the recorded base-only check is green with no package.
    _run(
        uv,
        "sync",
        "--frozen",
        "--check",
        "--offline",
        "--project",
        str(project),
        env=env,
    )
    assert _installed_version(workspace) == "absent"

    # Route the production delegate's hook operations to real uv. All other
    # operations retain the shared harness's shims and existing assertions.
    shim = workspace.bin_dir / "uv"
    shim.rename(workspace.bin_dir / "uv-other-surfaces")
    shim.write_text(
        '#!/usr/bin/env bash\nprevious=""\n'
        'for arg in "$@"; do\n'
        f'  if [[ "$previous" == "--project" && "$arg" == {shlex.quote(str(project))} ]]; then\n'
        f'    exec {shlex.quote(uv)} "$@" --offline\n'
        '  fi\n  previous="$arg"\ndone\n'
        f'exec {shlex.quote(str(workspace.bin_dir / "uv-other-surfaces"))} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return workspace


def _installed_version(workspace: _Workspace) -> str:
    return _run(
        str(workspace.omniclaude / ".venv/bin/python"),
        "-c",
        "from importlib.metadata import version, PackageNotFoundError\n"
        "try: print(version('omninode-intelligence'))\n"
        "except PackageNotFoundError: print('absent')\n",
    ).stdout.strip()


def _reconcile(workspace: _Workspace, *args: str) -> subprocess.CompletedProcess[str]:
    env = workspace.env()
    env["ONEX_RECONCILE_UV_BIN"] = str(workspace.bin_dir / "uv")
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_check_detects_missing_intelligence_without_writing(
    hook_workspace: _Workspace,
) -> None:
    result = _reconcile(hook_workspace, "--check")
    assert "DRIFT: hook venv" in result.stdout, result.stdout + result.stderr
    assert result.returncode != 0
    assert _installed_version(hook_workspace) == "absent"


def test_repair_installs_locked_intelligence_and_next_hook_check_passes(
    hook_workspace: _Workspace,
) -> None:
    result = _reconcile(hook_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _installed_version(hook_workspace) == "0.25.3"
    result = _reconcile(hook_workspace, "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRIFT: hook venv" not in result.stdout, result.stdout + result.stderr


def test_hook_sync_still_removes_undeclared_packages(
    hook_workspace: _Workspace,
) -> None:
    result = _reconcile(hook_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    python = hook_workspace.omniclaude / ".venv/bin/python"
    site = Path(
        _run(
            str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"
        ).stdout.strip()
    )
    pollution = site / "undeclared_probe-1.0.dist-info"
    pollution.mkdir()
    (pollution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: undeclared-probe\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (pollution / "RECORD").write_text(
        f"{pollution.name}/METADATA,,\n{pollution.name}/RECORD,,\n", encoding="utf-8"
    )
    result = _reconcile(hook_workspace, "--check")
    assert "DRIFT: hook venv" in result.stdout, result.stdout + result.stderr
    result = _reconcile(hook_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not pollution.exists()
    assert _installed_version(hook_workspace) == "0.25.3"
