# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Adversarial source-target tests for the architecture-layer script (OMN-17793)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "check_architecture.sh"


def _source_tree(
    tmp_path: Path,
    *,
    project_name: str = "omnibase_core",
    origin_url: str = "git@github.com:OmniNode-ai/omnibase_core.git",
    literal_name: bool = False,
    make_commit: bool = True,
    pyproject_toml: str | None = None,
) -> Path:
    project_root = tmp_path / "omnibase_core"
    package = project_root / "src" / "omnibase_core"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__all__ = []\n")
    name_quote = "'" if literal_name else '"'
    (project_root / "pyproject.toml").write_text(
        pyproject_toml
        or f'[project]\nname = {name_quote}{project_name}{name_quote}\nversion = "0.0.0"\n'
    )
    subprocess.run(["git", "init", "-q", str(project_root)], check=True)
    subprocess.run(
        ["git", "-C", str(project_root), "remote", "add", "origin", origin_url],
        check=True,
    )
    if make_commit:
        subprocess.run(["git", "-C", str(project_root), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
    return package


def _run(
    path: Path, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--no-color", "--path", str(path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _run_without_path(
    cwd: Path, *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--no-color"],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        check=False,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "origin_url",
    [
        "git@github.com:OmniNode-ai/omnibase_core.git",
        "https://github.com/OmniNode-ai/omnibase_core.git",
        "ssh://git@github.com/OmniNode-ai/omnibase_core.git",
    ],
)
def test_valid_source_package_is_accepted(tmp_path: Path, origin_url: str) -> None:
    result = _run(_source_tree(tmp_path, origin_url=origin_url))
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_valid_literal_toml_project_name_is_accepted(tmp_path: Path) -> None:
    result = _run(_source_tree(tmp_path, literal_name=True))
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_unborn_canonical_checkout_is_rejected(tmp_path: Path) -> None:
    result = _run(_source_tree(tmp_path, make_commit=False))
    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "pyproject_toml",
    [
        '[project]\nname = "omnibase_corex"\nversion = "0.0.0"\n',
        '[project]\nname = "omnibase_core" trailing\nversion = "0.0.0"\n',
    ],
)
def test_near_or_malformed_project_name_is_rejected(
    tmp_path: Path, pyproject_toml: str
) -> None:
    result = _run(_source_tree(tmp_path, pyproject_toml=pyproject_toml))
    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
def test_valid_symlink_scans_the_physical_source_target(tmp_path: Path) -> None:
    """A valid symlink must not reintroduce a logical scan target after validation."""
    physical = _source_tree(tmp_path / "physical")
    logical = tmp_path / "logical-source"
    logical.symlink_to(physical, target_is_directory=True)

    result = _run(logical)

    assert result.returncode == 0, result.stderr
    assert f"Target: {physical}" in result.stdout
    assert f"Target: {logical}" not in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "arbitrary",
        "empty",
        "site-packages",
        "dist-packages",
        ".venv",
        "venv",
    ],
)
def test_explicit_invalid_targets_fail_without_fallback(
    tmp_path: Path, kind: str
) -> None:
    if kind == "missing":
        target = tmp_path / "missing"
    elif kind == "arbitrary":
        target = tmp_path / "arbitrary"
        target.mkdir()
    elif kind == "empty":
        target = tmp_path / "omnibase_core" / "src" / "omnibase_core"
        target.mkdir(parents=True)
    else:
        if kind in {"site-packages", "dist-packages"}:
            target = tmp_path / kind / "omnibase_core"
        else:
            target = tmp_path / kind / "src" / "omnibase_core"
        target.mkdir(parents=True)
        (target / "__init__.py").write_text("\n")

    result = _run(target)
    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
def test_invalid_environment_override_does_not_fall_back(tmp_path: Path) -> None:
    invalid = tmp_path / "empty"
    invalid.mkdir()
    env = os.environ.copy()
    env["OMNIBASE_CORE_PATH"] = str(invalid)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--no-color"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2
    assert "OMNIBASE_CORE_PATH" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "origin_url",
    [
        "git@github.com:OmniNode-ai/not-omnibase-core.git",
        "git@evilgithub.com:OmniNode-ai/omnibase_core.git",
        "https://notgithub.com/OmniNode-ai/omnibase_core.git",
    ],
)
def test_environment_override_rejects_noncanonical_checkout(
    tmp_path: Path, origin_url: str
) -> None:
    decoy = _source_tree(
        tmp_path,
        origin_url=origin_url,
    )
    env = os.environ.copy()
    env["OMNIBASE_CORE_PATH"] = str(decoy)

    result = _run_without_path(tmp_path, env=env)

    assert result.returncode == 2
    assert "OMNIBASE_CORE_PATH" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "forbidden_root", ["site-packages", "dist-packages", ".venv", "venv"]
)
def test_symlinked_forbidden_source_root_is_rejected(
    tmp_path: Path, forbidden_root: str
) -> None:
    """An intermediate link cannot hide an installed/venv target."""
    physical = tmp_path / forbidden_root / "src" / "omnibase_core"
    physical.mkdir(parents=True)
    (physical / "__init__.py").write_text("__all__ = []\n")
    (tmp_path / forbidden_root / "pyproject.toml").write_text(
        "[project]\nname = 'omnibase_core'\nversion = '0.0.0'\n"
    )
    logical_root = tmp_path / "linked"
    logical_root.symlink_to(tmp_path / forbidden_root, target_is_directory=True)
    logical = logical_root / "src" / "omnibase_core"

    result = _run(logical)

    assert result.returncode == 2
    assert "source package" in result.stderr


@pytest.mark.unit
def test_no_argument_resolution_uses_core_sibling_of_linked_worktree(
    tmp_path: Path,
) -> None:
    """Auto-detection follows the canonical sibling layout from a linked worktree."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    core = _source_tree(workspace)

    infra_repo = workspace / "omnibase_infra"
    infra_repo.mkdir()
    subprocess.run(["git", "init", "-q", str(infra_repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(infra_repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "init",
        ],
        check=True,
    )
    linked = workspace / "omnibase_infra-linked"
    subprocess.run(
        [
            "git",
            "-C",
            str(infra_repo),
            "worktree",
            "add",
            "--detach",
            "-q",
            str(linked),
            "HEAD",
        ],
        check=True,
    )

    env = os.environ.copy()
    env.pop("OMNIBASE_CORE_PATH", None)
    env.pop("OMNI_HOME", None)
    result = _run_without_path(linked, env=env)

    assert result.returncode == 0, result.stderr
    assert str(core) in result.stdout


@pytest.mark.unit
def test_no_argument_linked_source_precedes_omni_home_fallback(
    tmp_path: Path,
) -> None:
    """A valid OMNI_HOME source is a fallback, not an override of a linked tree."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    linked_core = _source_tree(workspace)

    infra_repo = workspace / "omnibase_infra"
    infra_repo.mkdir()
    subprocess.run(["git", "init", "-q", str(infra_repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(infra_repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "init",
        ],
        check=True,
    )
    linked = workspace / "omnibase_infra-linked"
    subprocess.run(
        [
            "git",
            "-C",
            str(infra_repo),
            "worktree",
            "add",
            "--detach",
            "-q",
            str(linked),
            "HEAD",
        ],
        check=True,
    )

    fallback_home = tmp_path / "fallback-home"
    _source_tree(fallback_home)
    env = os.environ.copy()
    env.pop("OMNIBASE_CORE_PATH", None)
    env["OMNI_HOME"] = str(fallback_home)
    # Pre-push exports infra Git metadata. The resolver must retain it for
    # linked-worktree discovery but not let it override the Core checkout.
    env["GIT_DIR"] = str(infra_repo / ".git")
    result = _run_without_path(linked, env=env)

    assert result.returncode == 0, result.stderr
    assert str(linked_core) in result.stdout


@pytest.mark.unit
def test_fallback_ignores_decoy_sibling_project(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _source_tree(workspace, project_name="not_omnibase_core")

    infra_repo = workspace / "omnibase_infra"
    infra_repo.mkdir()
    subprocess.run(["git", "init", "-q", str(infra_repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(infra_repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "init",
        ],
        check=True,
    )
    linked = workspace / "omnibase_infra-linked"
    subprocess.run(
        [
            "git",
            "-C",
            str(infra_repo),
            "worktree",
            "add",
            "--detach",
            "-q",
            str(linked),
            "HEAD",
        ],
        check=True,
    )
    fallback_home = tmp_path / "fallback-home"
    fallback_core = _source_tree(fallback_home)
    env = os.environ.copy()
    env.pop("OMNIBASE_CORE_PATH", None)
    env["OMNI_HOME"] = str(fallback_home)

    result = _run_without_path(linked, env=env)

    assert result.returncode == 0, result.stderr
    assert str(fallback_core) in result.stdout


@pytest.mark.unit
def test_zero_python_file_guard_is_fail_closed() -> None:
    text = SCRIPT.read_text()
    assert "JSON_EXIT_CODE=2" in text
    assert "No Python files found in source target" in text
    assert "exit 2" in text
