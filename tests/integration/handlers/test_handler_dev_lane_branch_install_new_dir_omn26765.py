# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A branch install of a file in a new directory survives a refresh (OMN-26765).

omnimarket#6933 adds a node directory the installed package lacks. The record is
written by one store; the refresh recreates the container, whose package tree again
lacks the directory; a second store over the same state directory makes the parent
directory, copies the file and reads it back at its recorded sha256.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from omnibase_infra.enums.enum_dev_lane_branch_install_state import (
    EnumDevLaneBranchInstallState,
)
from omnibase_infra.handlers.handler_dev_lane_branch_install import (
    DevLaneBranchInstallStore,
    HandlerDevLaneBranchInstall,
    register,
)

pytestmark = pytest.mark.integration

PROJECT = "omnibase-infra"
SERVICE = "omninode-runtime"
PKG = "/app/.venv/lib/python3.12/site-packages/omnimarket/nodes"
TARGET = f"{PKG}/node_mcp_registry_server/handlers/handler_mcp_serve.py"
BRANCH_COPY = b"# the branch's new node handler\n"
NOW = datetime(2026, 11, 28, 1, 0, tzinfo=UTC)


class _Docker:
    """A runtime container whose ``docker cp`` needs the destination's parent directory."""

    def __init__(self) -> None:
        self.generation = 0
        self.recreate()

    def recreate(self) -> None:
        self.generation += 1
        self.container = f"{self.generation:064x}"
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {"/app", PKG}

    def __call__(
        self, argv: list[str], timeout: float = 0.0
    ) -> subprocess.CompletedProcess[str]:
        def done(
            code: int = 0, out: str = "", err: str = ""
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, code, out, err)

        verb = argv[1]
        if verb == "ps":
            return done(out=self.container + "\n")
        if verb == "exec":
            command = argv[2:]
            if command[:2] == ["-u", "0"]:
                command = command[2:]
            command = command[1:]
            if command[:3] == ["mkdir", "-p", "--"]:
                self.dirs.add(command[3])
                return done()
            if command[:2] == ["sha256sum", "--"]:
                data = self.files.get(command[2])
                if data is None:
                    return done(code=1, err="sha256sum: No such file or directory")
                return done(out=f"{hashlib.sha256(data).hexdigest()}  {command[2]}\n")
            if command[:1] == ["stat"]:
                return done(code=1)
            return done()
        if verb == "cp":
            path = argv[3].partition(":")[2]
            if str(Path(path).parent) not in self.dirs:
                return done(code=1)
            self.files[path] = Path(argv[2]).read_bytes()
            return done()
        if verb == "inspect":
            return done(out="running|healthy\n")
        raise AssertionError(f"unexpected command {argv}")


def test_a_new_node_directory_file_is_reapplied_by_a_fresh_store_after_the_refresh(
    tmp_path: Path,
) -> None:
    source = tmp_path / "branch" / "handler.py"
    source.parent.mkdir()
    source.write_bytes(BRANCH_COPY)
    state = tmp_path / "state"
    register(
        DevLaneBranchInstallStore(state),
        lane="dev",
        install_id="new-dir-omn26765",
        repo="omnimarket",
        pr=6933,
        branch="omn-26765-new-node-dir",
        head="8020dbdcc9a4961cdc5684d14c63b3bc024eef30",
        ticket="OMN-26765",
        registered_by="replay-test",
        services=[SERVICE],
        sources={TARGET: source},
        expires_at=NOW + timedelta(hours=48),
        restart=False,
        replace=False,
        now=NOW,
    )

    docker = _Docker()
    handler = HandlerDevLaneBranchInstall(
        DevLaneBranchInstallStore(state),
        PROJECT,
        run=docker,
        clock=lambda: NOW,
        sleep=lambda _s: None,
    )
    assert not handler.check().ok
    receipt = handler.apply()

    assert receipt.ok
    assert docker.files[TARGET] == BRANCH_COPY
    assert receipt.readbacks[0].state is EnumDevLaneBranchInstallState.REAPPLIED
    assert handler.check().ok
