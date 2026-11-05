# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every container replacement the deploy agent performs runs the hot-patch gate.

OMN-19998. ``scripts/preflight_hotpatch_ledger.py`` refuses to
rebuild over a recorded hot-patch whose source fix is not merged into the build
ref. Only the shell deploy called it; the deploy agent's replacement paths --
``_compose_up`` (and the per-container recovery inside it) and
``run_compose_recreate`` (verify-path, one-shot and migration recreates) --
replaced containers without it.

These tests drive the REAL gate script against a recorded ledger row and a real
git clone; only ``docker`` is a stub (its ``exec`` prints the container's live
``.prepatch`` paths), and every compose command is captured, never run.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import (
    DeployExecutor,
    guard_hotpatch_ledger,
    run_compose_recreate,
)

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_deploy_agent_replacement import (
    HotpatchLedgerPreflightError,
)

_GATE = Path(__file__).resolve().parents[2] / "scripts" / "preflight_hotpatch_ledger.py"
_REAL_RUN = subprocess.run
_PREPATCH = "/app/src/omnibase_infra/x.py.prepatch"


def _git(repo: Path, *args: str) -> str:
    env = scrub_git_location_env(os.environ)
    env.update(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
    )
    return _REAL_RUN(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()


def _commit(repo: Path, content: str) -> str:
    (repo / "f.txt").write_text(content)
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", content)
    return _git(repo, "rev-parse", "HEAD")


class _Host:
    def __init__(self, root: Path) -> None:
        self.clones_root = root / "clones_root"
        self.repo = self.clones_root / "omnibase_infra"
        self.repo.mkdir(parents=True)
        _git(self.repo, "init", "-b", "dev")
        self.merged = _commit(self.repo, "merged fix")
        # A fix on a side branch: recorded in the ledger, never merged into dev.
        _git(self.repo, "checkout", "-b", "unmerged")
        self.unmerged = _commit(self.repo, "unmerged fix")
        _git(self.repo, "checkout", "dev")
        (self.repo / "scripts").mkdir()
        shutil.copy(_GATE, self.repo / "scripts" / _GATE.name)
        self.ledger = root / "ledger.yaml"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text(
            f'#!/bin/sh\nif [ "$1" = "exec" ]; then echo "{_PREPATCH}"; fi\nexit 0\n'
        )
        docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
        self.bin_dir = bin_dir

    def record(self, merge_commit: str) -> None:
        row = {
            "container": "omninode-runtime",
            "lane": "dev",
            "file": _PREPATCH.removesuffix(".prepatch"),
            "prepatch_path": _PREPATCH,
            "source_repo": "omnibase_infra",
            "source_pr": "OmniNode-ai/omnibase_infra#1",
            "merge_commit": merge_commit,
            "merged": True,
        }
        self.ledger.write_text(yaml.safe_dump({"schema": 1, "rows": [row]}))


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Host:
    h = _Host(tmp_path)
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(h.repo))
    monkeypatch.setenv("OMNI_HOME", str(h.clones_root))
    monkeypatch.setenv("HOTPATCH_LEDGER_PATH", str(h.ledger))
    monkeypatch.setenv("PATH", f"{h.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("HOTPATCH_PREFLIGHT_BYPASS", raising=False)
    return h


@pytest.fixture
def compose_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture compose commands; let git and the gate itself really run."""
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: Any, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if "compose" in cmd or (cmd and cmd[0] == "docker-compose"):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return _REAL_RUN(cmd, *args, **kwargs)

    monkeypatch.setattr(executor_mod.subprocess, "run", fake_run)
    return calls


@pytest.mark.unit
@pytest.mark.parametrize("lane", list(EnumRuntimeLane))
def test_dev_lane_name_matches_shell_rule(
    host: _Host, monkeypatch: pytest.MonkeyPatch, lane: EnumRuntimeLane
) -> None:
    host.record(host.merged)
    calls: list[list[str]] = []

    def capture(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        if str(_GATE.name) in " ".join(cmd):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return _REAL_RUN(cmd, *args, **kwargs)

    monkeypatch.setattr(executor_mod.subprocess, "run", capture)
    guard_hotpatch_ledger(lane)
    project = executor_mod.lane_config_for(lane).compose_project
    expected = project.removeprefix("omnibase-infra").removeprefix("-") or "dev"
    assert len(calls) == 1
    assert calls[0][calls[0].index("--lane") + 1] == expected


@pytest.mark.unit
def test_unmerged_recorded_row_refuses_the_gate(host: _Host) -> None:
    host.record(host.unmerged)
    with pytest.raises(HotpatchLedgerPreflightError, match="none of which is merged"):
        guard_hotpatch_ledger(EnumRuntimeLane.DEV)


@pytest.mark.unit
def test_merged_recorded_row_passes_the_gate(host: _Host) -> None:
    host.record(host.merged)
    guard_hotpatch_ledger(EnumRuntimeLane.DEV)


@pytest.mark.unit
def test_no_ledger_on_host_skips_the_gate(
    host: _Host, compose_calls: list[list[str]]
) -> None:
    assert not host.ledger.exists()
    result = run_compose_recreate(
        ["docker", "compose", "up", "-d", "--force-recreate", "omninode-runtime"],
        timeout=5,
        lane=EnumRuntimeLane.DEV,
        services=["omninode-runtime"],
    )
    assert result.returncode == 0
    assert len(compose_calls) == 1


@pytest.mark.unit
def test_recreate_refused_before_compose_runs(
    host: _Host, compose_calls: list[list[str]]
) -> None:
    host.record(host.unmerged)
    with pytest.raises(HotpatchLedgerPreflightError):
        run_compose_recreate(
            ["docker", "compose", "up", "-d", "--force-recreate", "omninode-runtime"],
            timeout=5,
            lane=EnumRuntimeLane.DEV,
            services=["omninode-runtime"],
        )
    assert compose_calls == []


@pytest.mark.unit
def test_recreate_runs_when_recorded_row_is_merged(
    host: _Host, compose_calls: list[list[str]]
) -> None:
    host.record(host.merged)
    result = run_compose_recreate(
        ["docker", "compose", "up", "-d", "--force-recreate", "omninode-runtime"],
        timeout=5,
        lane=EnumRuntimeLane.DEV,
        services=["omninode-runtime"],
    )
    assert result.returncode == 0
    assert len(compose_calls) == 1


@pytest.mark.unit
def test_compose_up_and_its_recovery_refused_before_any_up(
    host: _Host, compose_calls: list[list[str]]
) -> None:
    host.record(host.unmerged)
    updates: list[tuple[Phase, PhaseStatus]] = []
    with pytest.raises(HotpatchLedgerPreflightError):
        DeployExecutor()._compose_up(
            Phase.CORE,
            Scope.CORE,
            [],
            lambda phase, status, *a, **k: updates.append((phase, status)),
            lane=EnumRuntimeLane.DEV,
        )
    assert not [c for c in compose_calls if "up" in c]
