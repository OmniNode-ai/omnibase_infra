# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``apply`` renders the command's overlays revision for the dev lane's overlay store root (OMN-24216).

``runtime_overlay_revision.apply`` renders one revision of the overlays repository from the
agent host's clone into the agent's state directory and names the folder as the dev compose
environment's overlay store root. Every name, directory and timeout comes from the node
contract's ``overlay_revision_render`` block, which these tests read the way the module does.
The clone is a real throwaway git repository, never a stand-in for git.
"""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.executor import load_dev_boot_guard

pytestmark = pytest.mark.unit

MODULE = "nodes.node_setup_local_provision_effect.runtime_overlay_revision"
ABSENT = "0" * 40


def _module() -> ModuleType:
    return load_dev_boot_guard(MODULE)


def _declared() -> dict[str, str]:
    return {k: str(v) for k, v in _module().declaration(executor_mod).items()}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def clone(tmp_path: Path) -> tuple[Path, str]:
    """A throwaway overlays clone with one committed ``overlays/`` folder, and its head."""
    repo = tmp_path / "overlays-clone"
    folder = repo / _declared()["overlay_dir"]
    folder.mkdir(parents=True)
    (folder / "lane.yaml").write_text("mode: on_demand\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.email=lane@example.invalid",
        "-c",
        "user.name=lane",
        "commit",
        "-q",
        "-m",
        "seed",
    )
    return repo, _git(repo, "rev-parse", "HEAD")


def _environ(clone_path: Path) -> dict[str, str]:
    return {_declared()["clone_env"]: str(clone_path)}


def test_a_revision_the_clone_holds_is_rendered_and_named_as_the_store_root(
    tmp_path: Path, clone: tuple[Path, str]
) -> None:
    clone_path, head = clone
    declared = _declared()
    environ = _environ(clone_path)
    state_dir = tmp_path / "state"
    _module().apply(executor_mod, state_dir, environ, head)
    folder = state_dir / declared["rendered_dir_name"] / head / declared["overlay_dir"]
    assert environ[declared["store_root_env"]] == str(folder)
    assert (folder / "lane.yaml").read_text(encoding="utf-8") == "mode: on_demand\n"
    # the runtime container reads the folder as another user
    assert stat.S_IMODE(folder.parent.stat().st_mode) & stat.S_IROTH


def test_the_same_revision_applied_twice_renders_once(
    tmp_path: Path, clone: tuple[Path, str]
) -> None:
    clone_path, head = clone
    environ = _environ(clone_path)
    _module().apply(executor_mod, tmp_path, environ, head)
    first = environ[_declared()["store_root_env"]]
    _module().apply(executor_mod, tmp_path, environ, head)
    assert environ[_declared()["store_root_env"]] == first


def test_a_revision_the_clone_lacks_ends_before_anything_is_written(
    tmp_path: Path, clone: tuple[Path, str]
) -> None:
    """Failing control: a refusal that wrote first would leave a rendered folder or env value."""
    clone_path, _ = clone
    environ = _environ(clone_path)
    with pytest.raises(_module().OverlayRevisionError, match="holds no"):
        _module().apply(executor_mod, tmp_path / "state", environ, ABSENT)
    assert _declared()["store_root_env"] not in environ
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("revision", ["not-a-sha", "ABCDEF" * 6 + "ABCD", "a" * 39])
def test_a_malformed_revision_is_refused(
    tmp_path: Path, clone: tuple[Path, str], revision: str
) -> None:
    clone_path, _ = clone
    with pytest.raises(_module().OverlayRevisionError, match="40-character"):
        _module().apply(executor_mod, tmp_path, _environ(clone_path), revision)


def test_an_undeclared_clone_is_refused(
    tmp_path: Path, clone: tuple[Path, str]
) -> None:
    _, head = clone
    with pytest.raises(_module().OverlayRevisionError, match=_declared()["clone_env"]):
        _module().apply(executor_mod, tmp_path, {}, head)


def test_an_empty_revision_with_no_override_changes_nothing(tmp_path: Path) -> None:
    environ = {"UNRELATED": "kept"}
    _module().apply(executor_mod, tmp_path, environ, "")
    assert environ == {"UNRELATED": "kept"}


def test_an_empty_revision_clears_a_value_this_module_rendered(
    tmp_path: Path, clone: tuple[Path, str]
) -> None:
    clone_path, head = clone
    environ = _environ(clone_path)
    _module().apply(executor_mod, tmp_path, environ, head)
    _module().apply(executor_mod, tmp_path, environ, "")
    assert _declared()["store_root_env"] not in environ


def test_an_empty_revision_leaves_a_value_set_any_other_way(tmp_path: Path) -> None:
    store_root_env = _declared()["store_root_env"]
    environ = {store_root_env: "/srv/example/overlays"}
    _module().apply(executor_mod, tmp_path, environ, "")
    assert environ == {store_root_env: "/srv/example/overlays"}
