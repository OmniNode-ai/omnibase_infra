# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Adversarial source-target tests for the architecture-layer script (OMN-17793)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "check_architecture.sh"
CANONICAL_ORIGIN = "git@github.com:OmniNode-ai/omnibase_core.git"


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True)


def _source_tree(
    parent: Path,
    *,
    origin_url: str = CANONICAL_ORIGIN,
    make_commit: bool = True,
    track_source: bool = True,
    git_root: Path | None = None,
) -> Path:
    """Create a minimal candidate source tree and return its package directory."""
    project_root = parent / "omnibase_core"
    package = project_root / "src" / "omnibase_core"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__all__ = []\n")
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "omnibase_core"\nversion = "0.0.0"\n'
    )

    repository = git_root or project_root
    repository.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    _git(repository, "remote", "add", "origin", origin_url)
    if make_commit:
        _git(repository, "add", ".")
        _git(
            repository,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        )
    if not track_source:
        _git(
            repository,
            "rm",
            "--cached",
            "-q",
            "src/omnibase_core/__init__.py",
        )
    return package


def _environment(**updates: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(updates)
    return env


def _run(
    path: Path, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--no-color", "--path", str(path)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )


def _run_without_path(
    cwd: Path, *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--no-color"],
        capture_output=True,
        check=False,
        cwd=cwd,
        env=env,
        text=True,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "origin_url",
    [
        CANONICAL_ORIGIN,
        "https://github.com/OmniNode-ai/omnibase_core.git",
        "ssh://git@github.com/OmniNode-ai/omnibase_core.git",
    ],
)
def test_canonical_local_origin_source_package_is_accepted(
    tmp_path: Path, origin_url: str
) -> None:
    package = _source_tree(tmp_path, origin_url=origin_url)

    result = _run(package)

    assert result.returncode == 0, result.stderr
    assert f"Target: {package}" in result.stdout


@pytest.mark.unit
def test_physical_source_target_is_reported_for_logical_symlink(tmp_path: Path) -> None:
    physical = _source_tree(tmp_path / "physical")
    logical = tmp_path / "logical-source"
    logical.symlink_to(physical, target_is_directory=True)

    result = _run(logical)

    assert result.returncode == 0, result.stderr
    assert f"Target: {physical}" in result.stdout
    assert f"Target: {logical}" not in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize(
    "forbidden_root", ["site-packages", "dist-packages", ".venv", "venv"]
)
def test_installed_or_venv_source_target_is_rejected(
    tmp_path: Path, forbidden_root: str
) -> None:
    package = _source_tree(tmp_path / forbidden_root)

    result = _run(package)

    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    ("make_commit", "track_source"), [(False, True), (True, False)]
)
def test_unborn_or_untracked_source_tree_is_rejected(
    tmp_path: Path, make_commit: bool, track_source: bool
) -> None:
    package = _source_tree(
        tmp_path,
        make_commit=make_commit,
        track_source=track_source,
    )

    result = _run(package)

    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
def test_nested_project_that_is_not_git_toplevel_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    package = _source_tree(repository / "nested", git_root=repository)

    result = _run(package)

    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "spoof_environment",
    [
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "remote.origin.url",
            "GIT_CONFIG_VALUE_0": CANONICAL_ORIGIN,
        },
        {
            "GIT_CONFIG_PARAMETERS": f"'remote.origin.url={CANONICAL_ORIGIN}'",
        },
    ],
)
def test_command_config_cannot_spoof_decoy_origin(
    tmp_path: Path, spoof_environment: dict[str, str]
) -> None:
    decoy = _source_tree(
        tmp_path,
        origin_url="git@github.com:attacker/omnibase_core.git",
    )

    result = _run(decoy, env=_environment(**spoof_environment))

    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
def test_numbered_command_config_cannot_spoof_second_origin(tmp_path: Path) -> None:
    decoy = _source_tree(
        tmp_path,
        origin_url="git@github.com:attacker/omnibase_core.git",
    )
    env = _environment(
        GIT_CONFIG_COUNT="2",
        GIT_CONFIG_KEY_0="core.filemode",
        GIT_CONFIG_VALUE_0="false",
        GIT_CONFIG_KEY_1="remote.origin.url",
        GIT_CONFIG_VALUE_1=CANONICAL_ORIGIN,
    )

    result = _run(decoy, env=env)

    assert result.returncode == 2


@pytest.mark.unit
def test_caller_path_cannot_select_forged_git_binary(tmp_path: Path) -> None:
    decoy = _source_tree(
        tmp_path,
        origin_url="git@github.com:attacker/omnibase_core.git",
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-C" ]; then core_root="$2"; shift 2; continue; fi\n'
        '  remaining="$remaining $1"; shift\n'
        "done\n"
        'case "$remaining" in\n'
        "  *show-toplevel*) /bin/printf '%s\\n' \"$core_root\" ;;\n"
        "  *HEAD\\^\\{commit\\}*) /bin/printf '%s\\n' forged-head ;;\n"
        "  *remote.origin.url*) /bin/printf '%s\\n' 'git@github.com:OmniNode-ai/omnibase_core.git' ;;\n"
        "  *ls-files*) /bin/printf '%s\\n' src/omnibase_core/__init__.py ;;\n"
        "esac\n"
    )
    fake_git.chmod(0o755)
    env = _environment(PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    result = _run(decoy, env=env)

    assert result.returncode == 2


@pytest.mark.unit
def test_global_url_rewrite_cannot_spoof_local_decoy_origin(tmp_path: Path) -> None:
    decoy = _source_tree(tmp_path, origin_url="mirror:omnibase_core")
    home = tmp_path / "attacker-home"
    home.mkdir()
    (home / ".gitconfig").write_text(
        f'[url "{CANONICAL_ORIGIN}"]\n\tinsteadOf = mirror:\n'
    )

    result = _run(decoy, env=_environment(HOME=str(home)))

    assert result.returncode == 2


@pytest.mark.unit
def test_repository_selector_and_object_store_environment_cannot_hide_unborn_head(
    tmp_path: Path,
) -> None:
    canonical = _source_tree(tmp_path / "canonical")
    decoy = _source_tree(tmp_path / "decoy", make_commit=False)
    canonical_root = canonical.parents[1]
    env = _environment(
        GIT_DIR=str(canonical_root / ".git"),
        GIT_WORK_TREE=str(canonical_root),
        GIT_INDEX_FILE=str(canonical_root / ".git" / "index"),
        GIT_OBJECT_DIRECTORY=str(canonical_root / ".git" / "objects"),
        GIT_ALTERNATE_OBJECT_DIRECTORIES=str(canonical_root / ".git" / "objects"),
        GIT_QUARANTINE_PATH=str(canonical_root / ".git" / "objects"),
    )

    result = _run(decoy, env=env)

    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
def test_exactly_one_local_origin_url_is_required(tmp_path: Path) -> None:
    package = _source_tree(tmp_path)
    repository = package.parents[1]
    _git(repository, "config", "--add", "remote.origin.url", CANONICAL_ORIGIN)

    result = _run(package)

    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
def test_environment_override_fails_closed_without_fallback(tmp_path: Path) -> None:
    invalid = tmp_path / "not-a-source-tree"
    invalid.mkdir()
    env = _environment(OMNIBASE_CORE_PATH=str(invalid))

    result = _run_without_path(tmp_path, env=env)

    assert result.returncode == 2
    assert "OMNIBASE_CORE_PATH" in result.stderr


@pytest.mark.unit
def test_linked_worktree_discovers_canonical_sibling_source(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    core = _source_tree(workspace)
    infra = workspace / "omnibase_infra"
    infra.mkdir()
    subprocess.run(["git", "init", "-q", str(infra)], check=True)
    _git(
        infra,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "init",
    )
    linked = workspace / "infra-linked"
    _git(infra, "worktree", "add", "--detach", "-q", str(linked), "HEAD")
    env = _environment()
    env.pop("OMNI_HOME", None)
    env.pop("OMNIBASE_CORE_PATH", None)

    result = _run_without_path(linked, env=env)

    assert result.returncode == 0, result.stderr
    assert f"Target: {core}" in result.stdout


@pytest.mark.unit
def test_pathless_linked_worktree_ignores_caller_path_git(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    core = _source_tree(workspace)
    infra = workspace / "omnibase_infra"
    infra.mkdir()
    subprocess.run(["git", "init", "-q", str(infra)], check=True)
    _git(
        infra,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "init",
    )
    linked = workspace / "infra-linked"
    _git(infra, "worktree", "add", "--detach", "-q", str(linked), "HEAD")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    invoked = tmp_path / "fake-git-invoked"
    fake_git.write_text(f"#!/bin/sh\n/bin/touch {invoked}\nexit 1\n")
    fake_git.chmod(0o755)
    env = _environment(PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    env.pop("OMNI_HOME", None)
    env.pop("OMNIBASE_CORE_PATH", None)

    result = _run_without_path(linked, env=env)

    assert result.returncode == 0, result.stderr
    assert f"Target: {core}" in result.stdout
    assert not invoked.exists()


@pytest.mark.unit
def test_zero_python_file_guard_is_fail_closed() -> None:
    text = SCRIPT.read_text()

    assert "No Python files found in source target" in text
    assert "JSON_EXIT_CODE=2" in text
