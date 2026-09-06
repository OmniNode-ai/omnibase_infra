# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for ARCH-004 baseline identity in descriptive worktrees."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_architecture_validator.validators.scanner_imperative_orchestrator_ratchet import (
    BASELINE_RELATIVE_PATH,
    load_baseline,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VALIDATE_PATH = _REPO_ROOT / "scripts/validate.py"

_ACCEPTED_H2_NODES = {
    "node_chain_orchestrator",
    "node_merge_sweep_workflow_orchestrator",
    "node_registration_orchestrator",
    "node_routing_orchestrator",
    "node_rsd_orchestrator",
    "node_runner_fleet_maintain_orchestrator",
    "node_scope_workflow_orchestrator",
}


@pytest.fixture
def validate_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the script without leaking module or import-path state between tests."""
    spec = importlib.util.spec_from_file_location(
        "omnibase_infra_validate", _VALIDATE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load validator from {_VALIDATE_PATH}")

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "omnibase_infra_validate", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize(
    "declared_name", ["omnibase_infra", "OmniBase-Infra", "omnibase.infra"]
)
def test_pep503_identity_recognizes_accepted_arch004_entries(
    validate_module: Any, tmp_path: Path, declared_name: str
) -> None:
    """PEP 503 package spellings must map to the underscore baseline keys."""
    descriptive_worktree = tmp_path / "linked-checkout-with-descriptive-name"
    descriptive_worktree.mkdir()
    (descriptive_worktree / "pyproject.toml").write_text(
        f"[project]\nname = {declared_name!r}\n", encoding="utf-8"
    )

    repo_name = validate_module._canonical_repository_name(descriptive_worktree)

    assert repo_name == "omnibase_infra"

    baseline = load_baseline(_REPO_ROOT / BASELINE_RELATIVE_PATH)
    assert {f"{repo_name}::{node}" for node in _ACCEPTED_H2_NODES}.issubset(baseline)


@pytest.mark.unit
def test_arch004_identity_uses_git_root_from_nested_worktree_directory(
    validate_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ARCH-004 runs from a nested directory against the worktree baseline."""
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "untrusted-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "untrusted-work-tree"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.chdir(_REPO_ROOT / "scripts")

    assert validate_module._repository_root(Path.cwd()) == _REPO_ROOT.resolve()

    assert validate_module.run_imperative_orchestrators(
        files=[
            "src/omnibase_infra/nodes/node_chain_orchestrator/handlers/"
            "handler_chain_replay_complete.py"
        ]
    )


@pytest.mark.unit
def test_arch004_identity_uses_path_resolved_git_with_sanitized_environment(
    validate_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Portable Git binaries keep the selector isolated from inherited state."""
    resolved_git = tmp_path / "portable" / "bin" / "git"
    captured: dict[str, object] = {}

    def run_git(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=f"{_REPO_ROOT.resolve()}\n",
        )

    monkeypatch.setenv("GIT_DIR", str(tmp_path / "untrusted-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "untrusted-work-tree"))
    monkeypatch.setattr(validate_module.shutil, "which", lambda _: str(resolved_git))
    monkeypatch.setattr(validate_module.subprocess, "run", run_git)

    assert (
        validate_module._repository_root(_REPO_ROOT / "scripts") == _REPO_ROOT.resolve()
    )
    assert captured["command"] == [
        str(resolved_git),
        "-C",
        str((_REPO_ROOT / "scripts").resolve()),
        "rev-parse",
        "--show-toplevel",
    ]
    assert captured["env"] == {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }


@pytest.mark.unit
def test_arch004_identity_fails_closed_without_a_resolved_git_executable(
    validate_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing portable Git binary remains a generic fail-closed error."""
    monkeypatch.setattr(validate_module.shutil, "which", lambda _: None)

    assert validate_module._repository_root(tmp_path) is None
    assert capsys.readouterr().out == (
        "Imperative Orchestrators: ERROR (git-worktree-root-unresolved)\n"
    )


@pytest.mark.unit
def test_arch004_identity_fails_closed_for_a_fake_nested_git_marker(
    validate_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A nested filesystem marker cannot replace the validator's Git root."""
    fake_checkout = tmp_path / "fake-checkout"
    (fake_checkout / ".git").mkdir(parents=True)
    (fake_checkout / "pyproject.toml").write_text(
        "[project]\nname = 'omnibase_infra'\n", encoding="utf-8"
    )
    monkeypatch.chdir(fake_checkout)

    assert validate_module._repository_root(Path.cwd()) is None


@pytest.mark.unit
def test_arch004_identity_fails_closed_for_a_genuine_foreign_git_root(
    validate_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real nested foreign repository cannot become this validator's root."""
    foreign_repo = tmp_path / "outer" / "nested-foreign-repository"
    foreign_repo.parent.mkdir(parents=True)
    subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            "PATH=/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM=1",
            "/usr/bin/git",
            "init",
            str(foreign_repo),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    monkeypatch.chdir(foreign_repo)

    assert validate_module._repository_root(Path.cwd()) is None


@pytest.mark.unit
def test_arch004_identity_does_not_expose_subprocess_failure_details(
    validate_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Git failures report a stable code rather than subprocess command details."""
    sensitive_detail = "do-not-expose-this-subprocess-command"

    def raise_git_failure(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=["git", sensitive_detail],
            stderr="do-not-expose-this-subprocess-stderr",
        )

    monkeypatch.setattr(validate_module.subprocess, "run", raise_git_failure)

    assert validate_module._repository_root(tmp_path) is None

    output = capsys.readouterr().out
    assert output == "Imperative Orchestrators: ERROR (git-worktree-root-unresolved)\n"
    assert sensitive_detail not in output


@pytest.mark.unit
def test_arch004_identity_fails_closed_without_project_name(
    validate_module: Any, tmp_path: Path
) -> None:
    """No worktree basename fallback is allowed when repository metadata is absent."""
    descriptive_worktree = tmp_path / "omnibase_infra-ticket-description"
    descriptive_worktree.mkdir()
    (descriptive_worktree / "pyproject.toml").write_text(
        "[project]\nversion = '0.0.0'\n", encoding="utf-8"
    )

    assert validate_module._canonical_repository_name(descriptive_worktree) is None


@pytest.mark.unit
def test_arch004_identity_fails_closed_for_malformed_metadata(
    validate_module: Any, tmp_path: Path
) -> None:
    """Malformed TOML must not produce a guessed architecture baseline key."""
    descriptive_worktree = tmp_path / "malformed-metadata-worktree"
    descriptive_worktree.mkdir()
    (descriptive_worktree / "pyproject.toml").write_text(
        "[project\nname = 'omnibase_infra'\n", encoding="utf-8"
    )

    assert validate_module._canonical_repository_name(descriptive_worktree) is None


@pytest.mark.unit
def test_arch004_identity_fails_closed_for_unreadable_metadata(
    validate_module: Any, tmp_path: Path
) -> None:
    """An unreadable metadata path must not fall back to a directory basename."""
    descriptive_worktree = tmp_path / "unreadable-metadata-worktree"
    descriptive_worktree.mkdir()
    (descriptive_worktree / "pyproject.toml").mkdir()

    assert validate_module._canonical_repository_name(descriptive_worktree) is None
