# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's overlay revision rendered into the agent's state dir (OMN-24216).

The deploy agent records one revision of the overlays repository and renders it into
its own state directory so compose can mount it. A deploy never runs on a revision
that is not rendered.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.lane_policy import ENV_ALLOWED_LANES

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

OverlayRevision = executor_mod.load_dev_boot_guard(
    "nodes.node_setup_local_provision_effect.runtime_overlay_revision"
)
OverlayRevisionError = OverlayRevision.OverlayRevisionError
DECLARED = OverlayRevision.declaration(executor_mod)
CLONE_ENV = str(DECLARED["clone_env"])
STATE_DIR_ENV = str(DECLARED["state_dir_env"])
RENDERED_DIR_NAME = str(DECLARED["rendered_dir_name"])
ACTIVE_FILE_NAME = str(DECLARED["active_file_name"])
STORE_ROOT_ENV = str(DECLARED["store_root_env"])
OVERLAY_DIR = str(DECLARED["overlay_dir"])


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )


def _make_overlay_repo(root: Path) -> tuple[Path, str, str]:
    """A git repo with two commits; commit 1 has overlays/runtime_contract_hotload/overlay.yaml with schema_version: 1 and lanes {}; commit 2 changes it to hold an example lane row."""
    repo = root / "overlay-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    overlay_path = repo / OVERLAY_DIR / "runtime_contract_hotload" / "overlay.yaml"
    overlay_path.parent.mkdir(parents=True)
    # Commit 1
    overlay_path.write_text(
        yaml.safe_dump({"schema_version": 1, "lanes": {}}, sort_keys=False),
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "commit 1")
    sha1 = _git_rev(repo)
    # Commit 2
    overlay_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "lanes": {
                    "example-lane": {
                        "enabled": True,
                        "mode": "on_demand",
                        "allowed_environments": ["local"],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "commit 2")
    sha2 = _git_rev(repo)
    return repo, sha1, sha2


def _git_rev(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )
    return result.stdout.strip()


def _environ(root: Path, clone: Path) -> dict[str, str]:
    state = root / "state"
    state.mkdir(exist_ok=True)
    return {
        CLONE_ENV: str(clone),
        STATE_DIR_ENV: str(state),
    }


def _active_file(environ: dict[str, str]) -> Path:
    return Path(environ[STATE_DIR_ENV]) / ACTIVE_FILE_NAME


def _rendered_folder(environ: dict[str, str], revision: str) -> Path:
    return (
        Path(environ[STATE_DIR_ENV])
        / RENDERED_DIR_NAME
        / revision
        / OVERLAY_DIR
    )


def _dev_environ(environ: dict[str, str]) -> dict[str, str]:
    """Environ with the dev fence active."""
    env = dict(environ)
    env[ENV_ALLOWED_LANES] = "dev"
    return env


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    repo, sha1, sha2 = _make_overlay_repo(tmp_path)
    environ = _environ(tmp_path, repo)
    return {
        "repo": repo,
        "sha1": sha1,
        "sha2": sha2,
        "environ": environ,
        "state_dir": Path(environ[STATE_DIR_ENV]),
    }


def test_golden_record_then_render_names_the_revision_folder(
    world: dict[str, Any],
) -> None:
    environ = _dev_environ(world["environ"])
    OverlayRevision.record(executor_mod, environ, world["sha2"])
    result = OverlayRevision.render_dev_lane(executor_mod, environ)
    expected_folder = _rendered_folder(environ, world["sha2"])
    assert result == {STORE_ROOT_ENV: str(expected_folder)}
    # The folder holds commit 2's text, not commit 1's.
    overlay_text = (expected_folder / "runtime_contract_hotload" / "overlay.yaml").read_text(
        encoding="utf-8"
    )
    doc = yaml.safe_load(overlay_text)
    assert doc["lanes"]["example-lane"]["mode"] == "on_demand"


def test_no_revision_declared_changes_nothing(world: dict[str, Any]) -> None:
    environ = _dev_environ(world["environ"])
    result = OverlayRevision.render_dev_lane(executor_mod, environ)
    assert result == {}


def test_empty_revision_clears_the_active_revision(world: dict[str, Any]) -> None:
    environ = _dev_environ(world["environ"])
    OverlayRevision.record(executor_mod, environ, world["sha2"])
    assert _active_file(environ).is_file()
    OverlayRevision.record(executor_mod, environ, "")
    assert not _active_file(environ).is_file()
    result = OverlayRevision.render_dev_lane(executor_mod, environ)
    assert result == {}


def test_error_revision_the_clone_lacks_is_refused_before_anything_is_written(
    world: dict[str, Any],
) -> None:
    environ = _dev_environ(world["environ"])
    bad_revision = "f" * 40
    with pytest.raises(OverlayRevisionError, match="refresh the overlays clone"):
        OverlayRevision.record(executor_mod, environ, bad_revision)
    # Nothing was written.
    assert not _active_file(environ).is_file()
    rendered_root = Path(environ[STATE_DIR_ENV]) / RENDERED_DIR_NAME
    if rendered_root.is_dir():
        assert not (rendered_root / bad_revision).is_dir()


def test_error_malformed_revision_is_refused(world: dict[str, Any]) -> None:
    environ = _dev_environ(world["environ"])
    with pytest.raises(OverlayRevisionError, match="not a 40-character"):
        OverlayRevision.record(executor_mod, environ, "not-a-sha")


def test_other_lane_takes_nothing(world: dict[str, Any]) -> None:
    environ = _dev_environ(world["environ"])
    OverlayRevision.record(executor_mod, environ, world["sha2"])
    # Set a non-dev fence.
    env = dict(world["environ"])
    env[ENV_ALLOWED_LANES] = EnumRuntimeLane.STABILITY_TEST.value
    result = OverlayRevision.render_dev_lane(executor_mod, env)
    assert result == {}


def test_error_active_revision_not_rendered_refuses(world: dict[str, Any]) -> None:
    environ = _dev_environ(world["environ"])
    # Write the active file by hand naming a revision with no folder.
    fake_revision = "a" * 40
    _active_file(environ).write_text(fake_revision, encoding="utf-8")
    with pytest.raises(RuntimeError, match="not rendered"):
        OverlayRevision.render_dev_lane(executor_mod, environ)
