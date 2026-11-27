# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20687, OMN-25330: ``resolve_runtime_deploy`` finds ``onex-runtime-deploy``.

The command is a console script in one first-party clone's own ``.venv`` and on
no host's PATH. The lane refresh scripts used to look in one fixed clone; after
the command moved to another clone, a refresh run on h201 exited 64 until the
operator named the command by hand. The lookup now names no clone: it takes the
first clone of ``SIBLING_CLONE_MANIFEST`` under OMNI_HOME whose venv holds the
command. Each branch of the resolution order is pinned here: the explicit
``DEPLOY_RUNTIME``, the declared ``DEPLOY_RUNTIME_HOME``, the manifest clones in
order, then PATH, and the loud failure that names every path looked in.
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


def _manifest_clones() -> list[str]:
    """The clone directory names the resolver searches, in its own order."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail; source "{MANIFEST}"; '
            'printf "%s\\n" "${SIBLING_CLONE_MANIFEST[@]}"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split()


_CLONES = _manifest_clones()
_FIRST, _LAST = _CLONES[0], _CLONES[-1]


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
        if key not in {"DEPLOY_RUNTIME", "DEPLOY_RUNTIME_HOME", "OMNI_HOME"}
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
def registry_root(tmp_path: Path) -> Path:
    home = tmp_path / "workspace" / "registry_root"
    home.mkdir(parents=True)
    return home


@pytest.mark.unit
def test_manifest_holds_more_than_one_clone() -> None:
    assert len(_CLONES) > 1
    assert _FIRST != _LAST


@pytest.mark.unit
def test_explicit_deploy_runtime_wins_over_the_clone(
    tmp_path: Path, registry_root: Path
) -> None:
    _install(registry_root / _LAST)
    resolved, tried = _resolve(
        _env(
            tmp_path,
            OMNI_HOME=str(registry_root),
            DEPLOY_RUNTIME="/opt/relocated/deploy",
        )
    )
    assert resolved == "/opt/relocated/deploy"
    assert tried == "DEPLOY_RUNTIME=/opt/relocated/deploy"


@pytest.mark.unit
def test_declared_home_wins_over_the_manifest_clones(
    tmp_path: Path, registry_root: Path
) -> None:
    _install(registry_root / _FIRST)
    declared = _install(tmp_path / "elsewhere" / "clone")
    resolved, _ = _resolve(
        _env(
            tmp_path,
            OMNI_HOME=str(registry_root),
            DEPLOY_RUNTIME_HOME=str(declared.parents[2]),
        )
    )
    assert resolved == str(declared)


@pytest.mark.unit
def test_a_declared_home_is_not_searched_past(
    tmp_path: Path, registry_root: Path
) -> None:
    """An empty declaration is not a licence to run a clone under OMNI_HOME."""
    _install(registry_root / _FIRST)
    empty = tmp_path / "declared-but-empty"
    empty.mkdir()
    resolved, tried = _resolve(
        _env(tmp_path, OMNI_HOME=str(registry_root), DEPLOY_RUNTIME_HOME=str(empty))
    )
    assert resolved == _COMMAND
    assert tried == f"{empty}/.venv/bin/{_COMMAND} PATH"


@pytest.mark.unit
def test_the_clone_that_holds_the_command_is_found_wherever_it_sits(
    tmp_path: Path, registry_root: Path
) -> None:
    """h201's shape: the command lives in a later clone, earlier ones lack it."""
    command = _install(registry_root / _LAST)
    resolved, tried = _resolve(_env(tmp_path, OMNI_HOME=str(registry_root)))
    assert resolved == str(command)
    assert tried.split()[-1] == str(command)
    assert len(tried.split()) == len(_CLONES)


@pytest.mark.unit
def test_the_first_manifest_clone_holding_the_command_wins(
    tmp_path: Path, registry_root: Path
) -> None:
    first = _install(registry_root / _FIRST)
    _install(registry_root / _LAST)
    resolved, tried = _resolve(_env(tmp_path, OMNI_HOME=str(registry_root)))
    assert resolved == str(first)
    assert tried == str(first)


@pytest.mark.unit
def test_a_directory_outside_the_manifest_is_not_searched(
    tmp_path: Path, registry_root: Path
) -> None:
    _install(registry_root / "not-a-manifest-clone")
    _install(registry_root.parent / _FIRST)
    resolved, tried = _resolve(_env(tmp_path, OMNI_HOME=str(registry_root)))
    assert resolved == _COMMAND
    assert tried.endswith(" PATH")
    assert "not-a-manifest-clone" not in tried


@pytest.mark.unit
def test_a_non_executable_command_is_not_used(
    tmp_path: Path, registry_root: Path
) -> None:
    command = _install(registry_root / _LAST, mode=0o644)
    resolved, tried = _resolve(_env(tmp_path, OMNI_HOME=str(registry_root)))
    assert resolved == _COMMAND
    assert tried.split()[-2:] == [str(command), "PATH"]


@pytest.mark.unit
def test_path_is_the_last_resort(tmp_path: Path, registry_root: Path) -> None:
    on_path = tmp_path / "bin"
    on_path.mkdir()
    stub = on_path / _COMMAND
    stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    env = _env(tmp_path, OMNI_HOME=str(registry_root))
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
def test_unset_registry_root_searches_only_path(tmp_path: Path) -> None:
    resolved, tried = _resolve(_env(tmp_path))
    assert resolved == _COMMAND
    assert tried == "PATH"


def _cut_lab_ref(
    tmp_path: Path, registry_root: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(CUT_LAB_REF), "--ref", "dev", "--lane", "dev", *args],
        capture_output=True,
        text=True,
        check=False,
        env=_env(tmp_path, OMNI_HOME=str(registry_root)),
    )


@pytest.mark.unit
def test_the_plan_names_the_clone_command(tmp_path: Path, registry_root: Path) -> None:
    command = _install(registry_root / _LAST)
    result = _cut_lab_ref(tmp_path, registry_root)
    assert result.returncode == 0, result.stderr
    assert f"{command} --execute --force" in result.stderr


@pytest.mark.unit
def test_execute_with_no_command_fails_loud_with_every_path_looked_in(
    tmp_path: Path, registry_root: Path
) -> None:
    result = _cut_lab_ref(tmp_path, registry_root, "--execute")
    assert result.returncode == 1
    assert "onex-runtime-deploy not found; looked in:" in result.stderr
    assert f"{registry_root}/{_LAST}/.venv/bin/{_COMMAND}" in result.stderr
    assert "PATH" in result.stderr
