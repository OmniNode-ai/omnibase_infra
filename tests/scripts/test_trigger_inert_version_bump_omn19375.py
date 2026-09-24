# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""omnimarket's own post-release version bump does not rebuild the lane (OMN-19375).

MEASURED: after every omnimarket release the release-on-merge bot merges a PR
that changes ``[project].version`` in ``pyproject.toml`` and omnimarket's own
``version`` line in ``uv.lock``, and nothing else. Both files are lane state
since OMN-18671, so each bump published a full dev-lane rebuild: jobs
``0805d076`` (after the bump following omnimarket#2822), ``e4d36317`` (#2813)
and ``ff4dc5de`` (#2817). ``ff4dc5de`` recreated ``omninode-runtime`` about nine
minutes before scheduled C15 run 35971574919, which then failed.

WHY THE BUMP IS INERT FOR omnimarket (verified 2026-09-24, not assumed)
----------------------------------------------------------------------
* The lane installs omnimarket from the staged source tree,
  ``--no-deps`` (``docker/Dockerfile.runtime``); the running
  ``omninode-runtime`` container's ``direct_url.json`` reads
  ``file:///workspace/sibling-repos/omnimarket``. No registry version is
  resolved, so the version string selects nothing.
* ``omnimarket.__version__`` is the literal ``0.1.0``. The only runtime reader
  of the installed distribution's version, ``omnimarket.runtime.
  version_handshake.check_plugin_compat``, has no caller in omnimarket,
  omnibase_infra or omnibase_core, and no ``plugin-compat.yaml`` exists.
* The deploy agent's ``runtime_omnimarket_version`` check compares the k3s lab
  lane against the compose lane, which are built from one image, so a bump
  that rebuilds neither leaves them equal.
* The sibling-pin preflight compares omnimarket's lock entry with its own
  ``pyproject.toml`` version; the bump moves both together.

omnibase_infra's own version is NOT inert (``service_kernel.KERNEL_VERSION``,
``overlay_config_resolver`` and ``version_compatibility`` read it), which is
why the exemption names packages rather than applying to every repository.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"
TRAIN_PATH = REPO_ROOT / "scripts" / "ci" / "release_train.py"
RUNTIME_PATH_VALIDATOR = REPO_ROOT / "tests" / "fixtures" / "runtime_path_classifier.py"


def _load_trigger_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_trigger_rebuild_omn19375", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_trigger_rebuild_omn19375"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trigger_module() -> Any:
    return _load_trigger_module()


def _pyproject(
    version: str,
    *,
    name: str = "omnimarket",
    infra_floor: str = "0.38.31",
) -> str:
    return f"""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "{version}"
requires-python = ">=3.12"
dependencies = [
    # a comment the bot never touches
    "omnibase-core>=0.47.3",
    "omnibase-infra>={infra_floor}",
]

[tool.hatch.version]
version = "unrelated"
"""


def _lock(
    version: str,
    *,
    name: str = "omnimarket",
    infra_version: str = "0.38.31",
) -> str:
    return f"""version = 1
revision = 3
requires-python = ">=3.12"

[[package]]
name = "omnibase-infra"
version = "{infra_version}"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "{name}"
version = "{version}"
source = {{ editable = "." }}
dependencies = [
    {{ name = "omnibase-infra" }},
]
"""


def _reader(files: dict[str, tuple[str | None, str | None]]) -> Any:
    def _read(path: str) -> tuple[str | None, str | None]:
        return files.get(path, (None, None))

    return _read


BUMP = {
    "pyproject.toml": (_pyproject("0.4.214"), _pyproject("0.4.215")),
    "uv.lock": (_lock("0.4.214"), _lock("0.4.215")),
}


def _canonical_none(_changed_files: list[str]) -> list[str]:
    return []


@pytest.mark.unit
class TestInertVersionBump:
    def test_the_bot_bump_reads_no_rebuild(self, trigger_module: Any) -> None:
        """AC5, RED before OMN-19375: the bump's two files were both lane state."""
        changed = ["pyproject.toml", "uv.lock"]
        runtime_paths = trigger_module.classify_runtime_paths(
            changed, _canonical_none, manifest_reader=_reader(BUMP)
        )
        assert runtime_paths == []
        assert not trigger_module.should_trigger(runtime_paths, [])

    def test_the_inert_paths_are_named(self, trigger_module: Any) -> None:
        assert trigger_module.inert_version_bump_paths(
            ["pyproject.toml", "uv.lock"], _reader(BUMP)
        ) == ["pyproject.toml", "uv.lock"]

    def test_without_a_reader_nothing_is_exempt(self, trigger_module: Any) -> None:
        """A caller that cannot read the manifests keeps today's answer."""
        assert trigger_module.classify_runtime_paths(
            ["pyproject.toml", "uv.lock"], _canonical_none
        ) == ["pyproject.toml", "uv.lock"]


@pytest.mark.unit
class TestPositiveControls:
    """AC6: a manifest change that does reach the runtime still rebuilds."""

    def test_a_dependency_floor_change_is_runtime(self, trigger_module: Any) -> None:
        files = {
            "pyproject.toml": (
                _pyproject("0.4.214"),
                _pyproject("0.4.214", infra_floor="0.38.32"),
            ),
        }
        assert trigger_module.classify_runtime_paths(
            ["pyproject.toml"], _canonical_none, manifest_reader=_reader(files)
        ) == ["pyproject.toml"]

    def test_a_version_bump_riding_with_a_dependency_change_is_runtime(
        self, trigger_module: Any
    ) -> None:
        files = {
            "pyproject.toml": (
                _pyproject("0.4.214"),
                _pyproject("0.4.215", infra_floor="0.38.32"),
            ),
            "uv.lock": (
                _lock("0.4.214"),
                _lock("0.4.215", infra_version="0.38.32"),
            ),
        }
        assert trigger_module.classify_runtime_paths(
            ["pyproject.toml", "uv.lock"],
            _canonical_none,
            manifest_reader=_reader(files),
        ) == ["pyproject.toml", "uv.lock"]

    def test_any_other_lock_change_is_runtime(self, trigger_module: Any) -> None:
        files = {
            "pyproject.toml": (_pyproject("0.4.214"), _pyproject("0.4.214")),
            "uv.lock": (_lock("0.4.214"), _lock("0.4.214", infra_version="0.38.32")),
        }
        assert trigger_module.classify_runtime_paths(
            ["uv.lock"], _canonical_none, manifest_reader=_reader(files)
        ) == ["uv.lock"]

    def test_a_non_inert_package_bump_is_runtime(self, trigger_module: Any) -> None:
        """omnibase_infra reads its own version at runtime; its bump rebuilds."""
        files = {
            "pyproject.toml": (
                _pyproject("0.38.40", name="omnibase-infra"),
                _pyproject("0.38.41", name="omnibase-infra"),
            ),
            "uv.lock": (
                _lock("0.38.40", name="omnibase-infra"),
                _lock("0.38.41", name="omnibase-infra"),
            ),
        }
        assert trigger_module.classify_runtime_paths(
            ["pyproject.toml", "uv.lock"],
            _canonical_none,
            manifest_reader=_reader(files),
        ) == ["pyproject.toml", "uv.lock"]

    def test_a_reader_that_fails_is_runtime(self, trigger_module: Any) -> None:
        def _broken(_path: str) -> tuple[str | None, str | None]:
            raise OSError("no checkout")

        assert trigger_module.classify_runtime_paths(
            ["pyproject.toml", "uv.lock"], _canonical_none, manifest_reader=_broken
        ) == ["pyproject.toml", "uv.lock"]

    def test_an_unparsable_manifest_is_runtime(self, trigger_module: Any) -> None:
        files = {
            "pyproject.toml": (_pyproject("0.4.214"), "[project\nname = "),
            "uv.lock": BUMP["uv.lock"],
        }
        assert trigger_module.classify_runtime_paths(
            ["pyproject.toml", "uv.lock"],
            _canonical_none,
            manifest_reader=_reader(files),
        ) == ["pyproject.toml", "uv.lock"]

    def test_a_bump_with_a_code_change_still_rebuilds(
        self, trigger_module: Any
    ) -> None:
        """AC7: only the manifests are exempt, never the source riding with them."""
        changed = ["pyproject.toml", "src/omnimarket/models/x.py", "uv.lock"]
        assert trigger_module.classify_runtime_paths(
            changed, _canonical_none, manifest_reader=_reader(BUMP)
        ) == ["src/omnimarket/models/x.py"]

    def test_the_canonical_result_is_never_narrowed(self, trigger_module: Any) -> None:
        def _canonical_claims_the_manifest(changed: list[str]) -> list[str]:
            return [p for p in changed if p == "pyproject.toml"]

        assert trigger_module.classify_runtime_paths(
            ["pyproject.toml", "uv.lock"],
            _canonical_claims_the_manifest,
            manifest_reader=_reader(BUMP),
        ) == ["pyproject.toml"]


@pytest.fixture(autouse=True)
def _no_inherited_git_location(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader under test runs git itself; a hook's GIT_DIR must not reach it."""
    kept = scrub_git_location_env(os.environ)
    for key in list(os.environ):
        if key not in kept:
            monkeypatch.delenv(key)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def _commit_bump(tmp_path: Path, after: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "omnimarket"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "pyproject.toml").write_text(_pyproject("0.4.214"), encoding="utf-8")
    (repo / "uv.lock").write_text(_lock("0.4.214"), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "release 0.4.214")
    for name, text in after.items():
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "bump")
    return repo, _git(repo, "rev-parse", "HEAD")


@pytest.mark.unit
class TestGitManifestReader:
    def test_reads_both_sides_of_the_merge(
        self, trigger_module: Any, tmp_path: Path
    ) -> None:
        repo, sha = _commit_bump(
            tmp_path,
            {"pyproject.toml": _pyproject("0.4.215"), "uv.lock": _lock("0.4.215")},
        )
        read = trigger_module.git_manifest_reader(repo, sha)
        assert read("pyproject.toml") == (_pyproject("0.4.214"), _pyproject("0.4.215"))
        assert read("absent.toml") == (None, None)

    def test_an_unknown_commit_raises(
        self, trigger_module: Any, tmp_path: Path
    ) -> None:
        repo, _sha = _commit_bump(tmp_path, {"uv.lock": _lock("0.4.215")})
        read = trigger_module.git_manifest_reader(repo, "0" * 40)
        with pytest.raises(OSError):
            read("pyproject.toml")


def _invoke(trigger_module: Any, extra: list[str]) -> Any:
    return CliRunner().invoke(
        trigger_module.main,
        [
            "--changed-files",
            "pyproject.toml,uv.lock",
            "--runtime-path-validator",
            str(RUNTIME_PATH_VALIDATOR),
            "--base-branch",
            "dev",
            "--source-repo",
            "omnimarket",
            "--primary-ref",
            "18f9eaead1c0f1e6a1d4b0c9f2a3e5d7c8b9a0f1",
            "--dry-run",
            *extra,
        ],
    )


@pytest.mark.unit
class TestCli:
    def test_dry_run_on_a_bot_bump_reads_no_rebuild(
        self, trigger_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _explode(**_kwargs: object) -> None:
            raise AssertionError("dry-run must not publish")

        monkeypatch.setattr(trigger_module, "publish_redeploy_start_event", _explode)
        repo, sha = _commit_bump(
            tmp_path,
            {"pyproject.toml": _pyproject("0.4.215"), "uv.lock": _lock("0.4.215")},
        )
        result = _invoke(
            trigger_module, ["--source-sha", sha, "--source-checkout", str(repo)]
        )
        assert result.exit_code == 0, result.output
        assert "No rebuild trigger" in result.output
        assert "pyproject.toml, uv.lock" in result.output

    def test_dry_run_without_a_checkout_keeps_the_rebuild(
        self, trigger_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _explode(**_kwargs: object) -> None:
            raise AssertionError("dry-run must not publish")

        monkeypatch.setattr(trigger_module, "publish_redeploy_start_event", _explode)
        result = _invoke(trigger_module, ["--source-sha", "a" * 40])
        assert result.exit_code == 0, result.output
        assert "Runtime change detected" in result.output


def _load_train() -> Any:
    name = "_release_train_omn19375"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TRAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
class TestTrainWalksPastTheBump:
    """The proof-subject walk and the trigger give one answer (OMN-19318).

    If the trigger declined the bump while the walk still called it
    runtime-affecting, the walk would stop at a commit no lab pass was ever
    run for and demand a receipt that cannot exist.
    """

    def _predicate(self, tmp_path: Path, repo: Path) -> Any:
        validator = tmp_path / "validate_pr_deploy_required.py"
        validator.write_text(
            "def find_runtime_paths(changed_files, *a, **k):\n    return []\n",
            encoding="utf-8",
        )
        return _load_train().load_runtime_affecting(
            repo, validator, labels_for=lambda _sha: []
        )

    def test_the_bot_bump_is_not_runtime_affecting(self, tmp_path: Path) -> None:
        repo, sha = _commit_bump(
            tmp_path,
            {"pyproject.toml": _pyproject("0.4.215"), "uv.lock": _lock("0.4.215")},
        )
        assert self._predicate(tmp_path, repo)(sha) is False

    def test_a_dependency_change_is_runtime_affecting(self, tmp_path: Path) -> None:
        repo, sha = _commit_bump(
            tmp_path,
            {"pyproject.toml": _pyproject("0.4.214", infra_floor="0.38.32")},
        )
        assert self._predicate(tmp_path, repo)(sha) is True
