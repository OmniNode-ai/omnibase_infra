# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the local bootstrap with real offline uv and installed Git wheels."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME = "omnibase_infra.nodes.node_reconcile_workspace_venvs.runtime_reconcile_workspace_venvs"
_CONTRACT = (
    _ROOT / "src/omnibase_infra/nodes/node_reconcile_workspace_venvs/contract.yaml"
)


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = scrub_git_location_env(os.environ)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env.pop("PYTHONPATH", None)
    env["UV_OFFLINE"] = "true"
    return subprocess.run(
        args, cwd=cwd, env=env, capture_output=True, text=True, check=False, timeout=90
    )


def _git(repo: Path, *args: str) -> str:
    result = _run(["git", *args], repo)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _commit(repo: Path, revision: str) -> str:
    (repo / "probe.py").write_text(f'REV = "{revision}"\n')
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", revision)
    return _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    "forward", [False, True], ids=["refuse-rewind", "allow-forward"]
)
def test_bootstrap_reads_back_installer_revision_with_real_uv(
    tmp_path: Path, forward: bool
) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q", "-b", "dev")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "Bootstrap integration")
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires=[]\nbuild-backend="_backend"\nbackend-path=["."]\n'
        '[project]\nname="branch-probe"\nversion="1.0"\n'
    )
    (source / "_backend.py").write_text(
        "from pathlib import Path\nimport zipfile\n"
        "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
        '    name = "branch_probe-1.0-py3-none-any.whl"\n'
        '    with zipfile.ZipFile(Path(wheel_directory) / name, "w") as z:\n'
        '        z.writestr("probe.py", Path("probe.py").read_text())\n'
        '        z.writestr("branch_probe-1.0.dist-info/METADATA", '
        '"Metadata-Version: 2.1\\nName: branch-probe\\nVersion: 1.0\\n")\n'
        '        z.writestr("branch_probe-1.0.dist-info/WHEEL", '
        '"Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n")\n'
        '        z.writestr("branch_probe-1.0.dist-info/RECORD", "")\n'
        "    return name\n"
    )
    locked = _commit(source, "locked")
    _git(source, "switch", "-qc", "installer-branch")
    installed = _commit(source, "installed")
    approved = _commit(source, "forward") if forward else locked
    project = tmp_path / "hook-project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname="hook-project"\nversion="1.0"\nrequires-python=">=3.12"\n'
        f'dependencies=["branch-probe @ git+{source.as_uri()}@{approved}"]\n'
    )
    result = _run(
        [uv, "sync", "--python", sys.executable, "--project", str(project)], project
    )
    assert result.returncode == 0, result.stdout + result.stderr
    python = project / ".venv/bin/python"
    result = _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            f"git+{source.as_uri()}@{installed}",
        ],
        project,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    direct_url = next(
        (project / ".venv").glob("lib/python*/site-packages/*/direct_url.json")
    )
    before = direct_url.read_bytes()
    assert json.loads(before)["vcs_info"]["commit_id"] == installed

    result = _run(
        [
            sys.executable,
            "-m",
            _RUNTIME,
            "--workspace",
            str(tmp_path),
            "--project",
            str(project),
            "--uv",
            uv,
            "--contract",
            str(_CONTRACT),
        ],
        _ROOT,
    )
    assert result.returncode == (0 if forward else 2), result.stdout + result.stderr
    expected = approved if forward else installed
    assert json.loads(direct_url.read_text())["vcs_info"]["commit_id"] == expected
    readback = _run([str(python), "-c", "import probe; print(probe.REV)"], project)
    assert readback.returncode == 0, readback.stderr
    assert readback.stdout.strip() == ("forward" if forward else "installed")
    if forward:
        assert "hook venv: reconciled" in result.stdout
    else:
        assert "conflicting branch install would be replaced" in result.stderr
        assert direct_url.read_bytes() == before
