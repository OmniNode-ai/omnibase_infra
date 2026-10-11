# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20687: ``resolve_runtime_deploy`` finds ``onex-runtime-deploy``.

The command is a console script of the omnibase_internal project, installed in
that clone's ``.venv`` and on no host's PATH. The lane refresh scripts used to
default ``DEPLOY_RUNTIME`` to the bare name and exit 64 when PATH lacked it.
Each branch of the resolution order is pinned here: the explicit
``DEPLOY_RUNTIME``, the omnibase_internal clone (declared home, sibling as
written, sibling of the resolved target), then PATH, and the loud failure that
names every path looked in.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"
MANIFEST = RUNTIME_BUILD / "sibling_clone_manifest.sh"
CUT_LAB_REF = RUNTIME_BUILD / "cut-lab-ref.sh"

_COMMAND = "onex-runtime-deploy"


def _path_without_the_command() -> str:
    """The ambient PATH minus any directory that already carries the command."""
    kept = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and not (Path(entry) / _COMMAND).exists()
    ]
    return os.pathsep.join(kept)


def _install(home: Path, *, mode: int = 0o755) -> Path:
    command = home / ".venv" / "bin" / _COMMAND
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    command.chmod(mode)
    return command


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"DEPLOY_RUNTIME", "OMNIBASE_INTERNAL_HOME", "OMNI_HOME"}
    }
    env["PATH"] = _path_without_the_command()
    env.update(overrides)
    return env


def _resolve(env: dict[str, str]) -> tuple[str, str]:
    """Run the resolver in a fresh shell; return (DEPLOY_RUNTIME, TRIED)."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail; source "{MANIFEST}"; resolve_runtime_deploy; '
            'printf "%s\\n%s\\n" "${DEPLOY_RUNTIME}" "${DEPLOY_RUNTIME_TRIED}"',
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    resolved, tried = result.stdout.splitlines()
    return resolved, tried


@pytest.fixture
def omni_home(tmp_path: Path) -> Path:
    home = tmp_path / "workspace" / "omni_home"
    home.mkdir(parents=True)
    return home


@pytest.mark.unit
def test_explicit_deploy_runtime_wins_over_the_clone(
    tmp_path: Path, omni_home: Path
) -> None:
    _install(omni_home.parent / "omnibase_internal")
    resolved, tried = _resolve(
        _env(tmp_path, OMNI_HOME=str(omni_home), DEPLOY_RUNTIME="/opt/relocated/deploy")
    )
    assert resolved == "/opt/relocated/deploy"
    assert tried == "DEPLOY_RUNTIME=/opt/relocated/deploy"


@pytest.mark.unit
def test_declared_internal_home_wins_over_the_sibling(
    tmp_path: Path, omni_home: Path
) -> None:
    _install(omni_home.parent / "omnibase_internal")
    declared = _install(tmp_path / "elsewhere" / "omnibase_internal")
    resolved, _ = _resolve(
        _env(
            tmp_path,
            OMNI_HOME=str(omni_home),
            OMNIBASE_INTERNAL_HOME=str(declared.parents[2]),
        )
    )
    assert resolved == str(declared)


@pytest.mark.unit
def test_a_declared_internal_home_is_not_searched_past(
    tmp_path: Path, omni_home: Path
) -> None:
    """An empty declaration is not a licence to run the sibling clone."""
    _install(omni_home.parent / "omnibase_internal")
    empty = tmp_path / "declared-but-empty"
    empty.mkdir()
    resolved, tried = _resolve(
        _env(tmp_path, OMNI_HOME=str(omni_home), OMNIBASE_INTERNAL_HOME=str(empty))
    )
    assert resolved == _COMMAND
    assert tried == f"{empty}/.venv/bin/{_COMMAND} PATH"


@pytest.mark.unit
def test_the_sibling_of_omni_home_is_used(tmp_path: Path, omni_home: Path) -> None:
    command = _install(omni_home.parent / "omnibase_internal")
    resolved, tried = _resolve(_env(tmp_path, OMNI_HOME=str(omni_home)))
    assert resolved == str(command)
    assert tried == str(command)


@pytest.mark.unit
def test_a_stale_sibling_beside_a_symlinked_omni_home_is_passed_over(
    tmp_path: Path,
) -> None:
    """h201's shape: the link's neighbour is stale, the target's neighbour is live."""
    real = tmp_path / "data"
    (real / "omni_home").mkdir(parents=True)
    live = _install(real / "omnibase_internal")
    link_dir = tmp_path / "home"
    link_dir.mkdir()
    link = link_dir / "omni_home"
    link.symlink_to(real / "omni_home")
    stale = link_dir / "omnibase_internal"
    stale.mkdir()

    resolved, tried = _resolve(_env(tmp_path, OMNI_HOME=str(link)))

    assert resolved == str(live)
    assert tried == f"{stale}/.venv/bin/{_COMMAND} {live}"


@pytest.mark.unit
def test_the_sibling_as_written_wins_when_it_holds_the_command(
    tmp_path: Path,
) -> None:
    real = tmp_path / "data"
    (real / "omni_home").mkdir(parents=True)
    _install(real / "omnibase_internal")
    link_dir = tmp_path / "home"
    link_dir.mkdir()
    link = link_dir / "omni_home"
    link.symlink_to(real / "omni_home")
    written = _install(link_dir / "omnibase_internal")

    resolved, _ = _resolve(_env(tmp_path, OMNI_HOME=str(link)))

    assert resolved == str(written)


@pytest.mark.unit
def test_a_non_executable_command_is_not_used(tmp_path: Path, omni_home: Path) -> None:
    command = _install(omni_home.parent / "omnibase_internal", mode=0o644)
    resolved, tried = _resolve(_env(tmp_path, OMNI_HOME=str(omni_home)))
    assert resolved == _COMMAND
    assert tried == f"{command} PATH"


@pytest.mark.unit
def test_path_is_the_last_resort(tmp_path: Path, omni_home: Path) -> None:
    on_path = tmp_path / "bin"
    on_path.mkdir()
    stub = on_path / _COMMAND
    stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    env = _env(tmp_path, OMNI_HOME=str(omni_home))
    env["PATH"] = f"{on_path}{os.pathsep}{env['PATH']}"

    resolved, tried = _resolve(env)

    assert resolved == _COMMAND
    assert tried.endswith(" PATH")
    found = subprocess.run(
        ["bash", "-c", f"command -v {resolved}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert found.stdout.strip() == str(stub)


@pytest.mark.unit
def test_unset_omni_home_searches_only_path(tmp_path: Path) -> None:
    resolved, tried = _resolve(_env(tmp_path))
    assert resolved == _COMMAND
    assert tried == "PATH"


def _cut_lab_ref(
    tmp_path: Path, omni_home: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(CUT_LAB_REF), "--ref", "dev", "--lane", "dev", *args],
        capture_output=True,
        text=True,
        check=False,
        env=_env(tmp_path, OMNI_HOME=str(omni_home)),
    )


@pytest.mark.unit
def test_the_plan_names_the_clone_command(tmp_path: Path, omni_home: Path) -> None:
    command = _install(omni_home.parent / "omnibase_internal")
    result = _cut_lab_ref(tmp_path, omni_home)
    assert result.returncode == 0, result.stderr
    assert f"{command} --execute --force" in result.stderr


@pytest.mark.unit
def test_execute_with_no_command_fails_loud_with_every_path_looked_in(
    tmp_path: Path, omni_home: Path
) -> None:
    result = _cut_lab_ref(tmp_path, omni_home, "--execute")
    assert result.returncode == 1
    assert "onex-runtime-deploy not found; looked in:" in result.stderr
    assert f"{omni_home.parent}/omnibase_internal/.venv/bin/{_COMMAND}" in result.stderr
    assert "PATH" in result.stderr
