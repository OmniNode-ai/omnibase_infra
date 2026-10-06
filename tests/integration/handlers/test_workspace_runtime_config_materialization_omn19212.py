# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Materialise a workspace's runtime config from a real ``origin`` (OMN-19212).

Unlike the unit tests, which plant ``refs/remotes/origin/main`` by hand, this
clones a real bare remote, so ``origin/main`` is produced by git itself. The
shared working tree then lags it, and the embedded runtime resolver must still
answer from the materialised copy, naming the full sha.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.handlers.handler_workspace_runtime_config_materializer import (
    SOURCE_PATH_IN_REPO,
    HandlerWorkspaceRuntimeConfigMaterializer,
)
from omnibase_infra.runtime.service_kernel import resolve_embedded_runtime_config

pytestmark = pytest.mark.integration

_TIER1 = 'event_bus:\n  type: "kafka"\n  profile: "local"\n  lane: "dev"\n'


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(cwd),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    )
    return result.stdout


def test_lagging_clone_resolves_transport_from_real_origin_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "-q", "--bare", "-b", "main")

    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "main")
    (seed / "notes.txt").write_text("base\n", encoding="utf-8")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "base")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-q", "origin", "main")

    workspace = tmp_path / "workspace"
    _git(tmp_path, "clone", "-q", str(origin), str(workspace))
    (workspace / ".git" / "info" / "exclude").write_text(
        ".onex_state/\n", encoding="utf-8"
    )

    # origin/main gains the tier-1 config after the workspace was cloned.
    config = seed / SOURCE_PATH_IN_REPO
    config.parent.mkdir(parents=True)
    config.write_text(_TIER1, encoding="utf-8")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "tier-1 config lands")
    _git(seed, "push", "-q", "origin", "main")
    sha = _git(seed, "rev-parse", "HEAD").strip()
    _git(workspace, "fetch", "-q", "origin")

    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(workspace))
    assert not (workspace / SOURCE_PATH_IN_REPO).exists()

    outcome = HandlerWorkspaceRuntimeConfigMaterializer().materialize(workspace)
    assert outcome.ok, outcome.detail
    assert outcome.sha == sha

    config_model, source = resolve_embedded_runtime_config(workspace_root=workspace)
    assert config_model.event_bus.type == "kafka"
    assert sha in source
    assert "materialised from origin/main" in source
    assert not (workspace / SOURCE_PATH_IN_REPO).exists()
