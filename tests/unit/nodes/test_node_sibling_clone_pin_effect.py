# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the sibling-clone pin resolution handler (OMN-23947).

Real git repos in pytest's tmp_path, no mocks. Each test builds a typed
``ModelSiblingClonePinRequest`` and calls ``HandlerSiblingClonePin.handle``
against a freshly initialised git repository, asserting the full
``ModelSiblingClonePinResult``.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_sibling_clone_pin import (
    HandlerSiblingClonePin,
)
from omnibase_infra.models.sibling_pin import (
    ModelSiblingClonePinRequest,
    ModelSiblingClonePinResult,
)

pytestmark = pytest.mark.unit

_STATIC = '[project]\nname = "omnibase_spi"\nversion = "0.20.6"\n'
_DYNAMIC = (
    "[project]\n"
    'name = "omnibase_core"\n'
    'dynamic = ["version"]\n\n'
    "[tool.hatch.version.raw-options]\n"
    'git_describe_command = ["git", "describe", "--dirty", "--tags", "--long", "--match", "v[0-9]*"]\n'
)
_DYNAMIC_NO_COMMAND = '[project]\nname = "omnibase_core"\ndynamic = ["version"]\n'
_DYNAMIC_BAD_COMMAND = (
    "[project]\n"
    'name = "omnibase_core"\n'
    'dynamic = ["version"]\n\n'
    "[tool.hatch.version.raw-options]\n"
    'git_describe_command = ["git", "rev-parse", "--short", "HEAD"]\n'
)
_NO_VERSION = '[project]\nname = "omnibase_spi"\n'


def _git(root: Path, *args: str) -> str:
    """Run a git command in *root* with a scrubbed environment."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=t@example.test",
            "-c",
            "user.name=t",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )
    return result.stdout.strip()


def _init_repo(root: Path, pyproject: str) -> None:
    """Create a git repo at *root* with a pyproject.toml and one commit."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "pyproject.toml")
    _git(root, "commit", "-q", "-m", "init")


def _commit(root: Path, message: str) -> None:
    """Create an empty commit in *root*."""
    _git(root, "commit", "-q", "--allow-empty", "-m", message)


def _handle(package: str, repo_root: Path) -> ModelSiblingClonePinResult:
    request = ModelSiblingClonePinRequest(package=package, repo_root=repo_root)
    return HandlerSiblingClonePin().handle(request)


class TestGoldenChain:
    def test_static_clone(self, tmp_path: Path) -> None:
        root = tmp_path / "static"
        _init_repo(root, _STATIC)
        head = _git(root, "rev-parse", "HEAD")
        result = _handle("omnibase_spi", root)
        assert result == ModelSiblingClonePinResult(
            package="omnibase_spi",
            version="0.20.6",
            version_derived=False,
            git_sha=head,
        )

    def test_dynamic_clone_with_tag_and_commits(self, tmp_path: Path) -> None:
        root = tmp_path / "dynamic"
        _init_repo(root, _DYNAMIC)
        _git(root, "tag", "v1.2.3")
        _commit(root, "second")
        _commit(root, "third")
        head = _git(root, "rev-parse", "HEAD")
        short = _git(root, "rev-parse", "--short", "HEAD")
        result = _handle("omnibase_core", root)
        assert result == ModelSiblingClonePinResult(
            package="omnibase_core",
            version=f"1.2.4.dev2+g{short}",
            version_derived=True,
            git_sha=head,
        )

    def test_dynamic_clone_detached_at_tag(self, tmp_path: Path) -> None:
        root = tmp_path / "detached"
        _init_repo(root, _DYNAMIC)
        _git(root, "tag", "v1.2.3")
        _git(root, "checkout", "-q", "v1.2.3")
        head = _git(root, "rev-parse", "HEAD")
        result = _handle("omnibase_core", root)
        assert result == ModelSiblingClonePinResult(
            package="omnibase_core",
            version="1.2.3",
            version_derived=True,
            git_sha=head,
        )


class TestErrorChain:
    def test_missing_pyproject(self, tmp_path: Path) -> None:
        root = tmp_path / "no_pyproject"
        root.mkdir()
        _git(root, "init", "-q")
        _git(root, "commit", "-q", "--allow-empty", "-m", "empty")
        with pytest.raises(FileNotFoundError, match=r"missing pyproject\.toml"):
            _handle("omnibase_spi", root)

    def test_no_version_and_no_dynamic(self, tmp_path: Path) -> None:
        root = tmp_path / "no_version"
        _init_repo(root, _NO_VERSION)
        with pytest.raises(ValueError, match=r"no project\.version"):
            _handle("omnibase_spi", root)

    def test_dynamic_no_command(self, tmp_path: Path) -> None:
        root = tmp_path / "no_command"
        _init_repo(root, _DYNAMIC_NO_COMMAND)
        _git(root, "tag", "v1.0.0")
        with pytest.raises(ValueError, match="no git_describe_command"):
            _handle("omnibase_core", root)

    def test_dynamic_no_tag(self, tmp_path: Path) -> None:
        root = tmp_path / "no_tag"
        _init_repo(root, _DYNAMIC)
        with pytest.raises(subprocess.CalledProcessError):
            _handle("omnibase_core", root)

    def test_dynamic_bad_command_output(self, tmp_path: Path) -> None:
        root = tmp_path / "bad_command"
        _init_repo(root, _DYNAMIC_BAD_COMMAND)
        with pytest.raises(ValueError, match="cannot parse git describe output"):
            _handle("omnibase_core", root)


class TestRequestModel:
    def test_rejects_extra_field(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            ModelSiblingClonePinRequest.model_validate(
                {"package": "pkg", "repo_root": tmp_path, "unexpected": "x"}
            )

    def test_rejects_empty_package(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            ModelSiblingClonePinRequest(package="", repo_root=tmp_path)


class TestContract:
    def test_contract_node_type_and_handler(self) -> None:
        import importlib.resources

        pkg = importlib.import_module(
            "omnibase_infra.nodes.node_sibling_clone_pin_effect"
        )
        contract_path = importlib.resources.files(pkg).joinpath("contract.yaml")
        data = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        assert data["node_type"] == "EFFECT_GENERIC"
        handler_entry = data["handler_routing"]["handlers"][0]
        module_name = handler_entry["handler"]["module"]
        class_name = handler_entry["handler"]["name"]
        module = importlib.import_module(module_name)
        assert getattr(module, class_name) is HandlerSiblingClonePin


_CLI = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_sibling_clone_pin_effect"
    / "runtime_sibling_clone_pin.py"
)
# An interpreter that holds pydantic and not omnibase_core, as the workspace build has.
_BARE = (
    "import runpy, sys; sys.modules['omnibase_core'] = None; "
    "sys.argv = [{cli!r}, *sys.argv[1:]]; "
    "runpy.run_path({cli!r}, run_name='__main__')"
)


class TestRuntimeCli:
    def test_cli_answers_derived_version_without_omnibase_core(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "dynamic"
        _init_repo(root, _DYNAMIC)
        _git(root, "tag", "v1.2.3")
        _commit(root, "second")
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _BARE.format(cli=str(_CLI)),
                "--package",
                "omnibase-core",
                "--repo-root",
                str(root),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        result = ModelSiblingClonePinResult.model_validate_json(proc.stdout)
        assert result.version_derived is True
        assert result.version.startswith("1.2.4.dev1+g")
        assert result.git_sha == _git(root, "rev-parse", "HEAD")

    def test_cli_refuses_with_exit_2_and_the_reason(self, tmp_path: Path) -> None:
        root = tmp_path / "no_pyproject"
        root.mkdir()
        proc = subprocess.run(
            [sys.executable, str(_CLI), "--package", "x", "--repo-root", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 2
        assert "missing pyproject.toml" in proc.stderr
