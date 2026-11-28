# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev-lane branch install by repository, branch and sha, with no pull request (OMN-26764)."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.enums.enum_dev_lane_branch_install_release_reason import (
    EnumDevLaneBranchInstallReleaseReason,
)
from omnibase_infra.handlers import handler_dev_lane_branch_install as bi
from omnibase_infra.handlers.handler_dev_lane_branch_install import (
    DevLaneBranchInstallStore,
    HandlerDevLaneBranchInstall,
    Runner,
    git_merged_probe,
    register,
    release,
)
from omnibase_infra.models.model_dev_lane_branch_install import (
    ModelDevLaneBranchInstall,
)

pytestmark = pytest.mark.unit

HEAD = "8020dbdcc9a4961cdc5684d14c63b3bc024eef30"
PATH = "/app/.venv/lib/python3.12/site-packages/omnimarket/x/handler.py"
NOW = datetime(2026, 11, 27, 23, 0, tzinfo=UTC)


def _declare(
    tmp_path: Path, store: DevLaneBranchInstallStore, **kw: object
) -> ModelDevLaneBranchInstall:
    src = tmp_path / "h.py"
    src.write_bytes(b"# fix\n")
    args: dict[str, Any] = {
        "lane": "dev",
        "install_id": None,
        "repo": "omnimarket",
        "pr": None,
        "branch": "omn-26764-fix",
        "head": HEAD,
        "ticket": "OMN-26764",
        "registered_by": "lane-x",
        "services": ["omninode-runtime"],
        "sources": {PATH: src},
        "expires_at": NOW + timedelta(hours=24),
        "now": NOW,
    }
    args.update(kw)
    return register(store, **args)


def _git(*, in_running: bool) -> Runner:
    def run(argv: list[str], timeout: float = 0.0) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "git"
        if "log" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[-1] == "origin/dev":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0 if in_running else 1, "", "")

    return run


def test_dev_lane_branch_install_no_pr_records_repo_branch_and_sha(
    tmp_path: Path,
) -> None:
    store = DevLaneBranchInstallStore(tmp_path / "state")

    install = _declare(tmp_path, store)

    assert install.pr is None
    assert (install.repo, install.branch, install.head) == (
        "omnimarket",
        "omn-26764-fix",
        HEAD,
    )
    assert install.id == f"omnimarket-{HEAD[:12]}"
    assert store.load().installs == (install,)


def test_dev_lane_branch_install_no_pr_command_line_needs_no_pr_flag(
    tmp_path: Path,
) -> None:
    src = tmp_path / "h.py"
    src.write_bytes(b"# fix\n")
    argv = [
        "register",
        "--state-dir",
        str(tmp_path / "state"),
        "--lane",
        "dev",
        "--repo",
        "omnimarket",
        "--branch",
        "b",
        "--head",
        HEAD,
        "--ticket",
        "OMN-26764",
        "--registered-by",
        "lane-x",
        "--service",
        "omninode-runtime",
        "--expires-in-hours",
        "24",
        "--file",
        f"{PATH}={src}",
    ]

    assert bi.main(argv) == bi.EXIT_OK
    assert DevLaneBranchInstallStore(tmp_path / "state").load().installs[0].pr is None


def test_dev_lane_branch_install_removed_on_roll_when_build_contains_the_sha(
    tmp_path: Path,
) -> None:
    store = DevLaneBranchInstallStore(tmp_path / "state")
    _declare(tmp_path, store)
    probe = git_merged_probe(
        lambda repo: Path("/clone") / repo,
        lambda repo: "c" * 40,
        run=_git(in_running=True),
    )
    handler = HandlerDevLaneBranchInstall(
        store,
        "omnibase-infra",
        run=_git(in_running=True),
        clock=lambda: NOW,
        merged_probe=probe,
    )

    released = handler.release_due()

    assert [r.reason for r in released] == [
        EnumDevLaneBranchInstallReleaseReason.MERGED
    ]
    assert "omnimarket@" + HEAD[:12] in released[0].detail
    assert store.load().installs == ()


def test_dev_lane_branch_install_removed_on_roll_keeps_it_while_build_lacks_sha(
    tmp_path: Path,
) -> None:
    store = DevLaneBranchInstallStore(tmp_path / "state")
    _declare(tmp_path, store)
    probe = git_merged_probe(
        lambda repo: Path("/clone") / repo,
        lambda repo: "c" * 40,
        run=_git(in_running=False),
    )
    handler = HandlerDevLaneBranchInstall(
        store, "omnibase-infra", clock=lambda: NOW, merged_probe=probe
    )

    assert handler.release_due() == ()
    assert len(store.load().installs) == 1


def test_dev_lane_branch_install_removed_on_roll_operator_release_by_key(
    tmp_path: Path,
) -> None:
    store = DevLaneBranchInstallStore(tmp_path / "state")
    install = _declare(tmp_path, store)

    dropped = release(
        store, install.id, EnumDevLaneBranchInstallReleaseReason.OPERATOR, "done"
    )

    assert dropped is not None and store.load().installs == ()
