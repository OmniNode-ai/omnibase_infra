# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev branch installer creates the parent directory of a new file (OMN-26765).

omnimarket#6933 adds one MCP server node, a directory the installed package does not
have. ``docker cp`` needs the destination's parent directory, so the install creates
it, and only under a declared root (the installed package root or the overlay root).
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_dev_lane_branch_install import (
    DevLaneBranchInstallRefusedError,
    DevLaneBranchInstallStore,
    HandlerDevLaneBranchInstall,
    register,
)

pytestmark = pytest.mark.unit

PROJECT = "omnibase-infra"
SERVICE = "omninode-runtime"
PKG = "/app/.venv/lib/python3.12/site-packages/omnimarket/nodes"
NEW_NODE_FILE = f"{PKG}/node_mcp_registry_server/handlers/handler_mcp_serve.py"
NEW_NODE_PARENT = f"{PKG}/node_mcp_registry_server/handlers"
HEAD = "8020dbdcc9a4961cdc5684d14c63b3bc024eef30"
NOW = datetime(2026, 11, 28, 1, 0, tzinfo=UTC)
BRANCH_COPY = b"# the branch's new node handler\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _done(
    argv: list[str], code: int = 0, out: str = "", err: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, code, out, err)


class _Lane:
    """A fake Docker: one container whose package tree lacks the new node directory."""

    def __init__(self) -> None:
        self.container = f"{1:064x}"
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {"/app", PKG}
        self.mkdirs: list[str] = []

    def __call__(
        self, argv: list[str], timeout: float = 0.0
    ) -> subprocess.CompletedProcess[str]:
        if argv[:2] == ["docker", "ps"]:
            return _done(argv, out=self.container + "\n")
        if argv[:2] == ["docker", "exec"]:
            rest = argv[2:]
            if rest[:2] == ["-u", "0"]:
                rest = rest[2:]
            command = rest[1:]
            if command[:3] == ["mkdir", "-p", "--"]:
                self.mkdirs.append(command[3])
                self.dirs.add(command[3])
                return _done(argv)
            if command[:2] == ["sha256sum", "--"]:
                path = command[2]
                if path not in self.files:
                    return _done(
                        argv, code=1, err="sha256sum: No such file or directory"
                    )
                return _done(argv, out=f"{_sha(self.files[path])}  {path}\n")
            if command[:1] == ["stat"]:
                return _done(argv, code=1)
            if command[:1] in (["chown"], ["chmod"]):
                return _done(argv)
            raise AssertionError(f"unexpected exec {command}")
        if argv[:2] == ["docker", "cp"]:
            path = argv[3].partition(":")[2]
            if str(Path(path).parent) not in self.dirs:
                return _done(argv, code=1)
            self.files[path] = Path(argv[2]).read_bytes()
            return _done(argv)
        if argv[:2] == ["docker", "inspect"]:
            return _done(argv, out="running|healthy\n")
        if argv[:2] == ["docker", "restart"]:
            return _done(argv)
        raise AssertionError(f"unexpected command {argv}")


def _declare(tmp_path: Path, store: DevLaneBranchInstallStore, path: str) -> None:
    source = tmp_path / "branch" / "handler.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(BRANCH_COPY)
    register(
        store,
        lane="dev",
        install_id="mcp-node-omn26765",
        repo="omnimarket",
        pr=6933,
        branch="omn-26765-mcp-node",
        head=HEAD,
        ticket="OMN-26765",
        registered_by="newfileinst-0055",
        services=[SERVICE],
        sources={path: source},
        expires_at=NOW + timedelta(hours=24),
        restart=False,
        now=NOW,
    )


def test_a_file_in_a_new_node_directory_installs_and_reads_back(
    tmp_path: Path,
) -> None:
    store = DevLaneBranchInstallStore(tmp_path / "state")
    _declare(tmp_path, store, NEW_NODE_FILE)
    lane = _Lane()
    handler = HandlerDevLaneBranchInstall(
        store, PROJECT, run=lane, sleep=lambda _s: None, clock=lambda: NOW
    )
    receipt = handler.apply()
    assert lane.mkdirs == [NEW_NODE_PARENT]
    assert lane.files[NEW_NODE_FILE] == BRANCH_COPY
    assert receipt.ok is True


@pytest.mark.parametrize(
    "path",
    [
        "/etc/onex/new-dir/x.py",
        "/opt/other/x.py",
        "/app/../etc/new-dir/x.py",
        "/app/new-dir/../../etc/x.py",
    ],
)
def test_no_directory_is_created_for_a_path_outside_every_declared_root(
    tmp_path: Path, path: str
) -> None:
    store = DevLaneBranchInstallStore(tmp_path / "state")
    with pytest.raises(DevLaneBranchInstallRefusedError):
        _declare(tmp_path, store, path)
    assert store.load().installs == ()
