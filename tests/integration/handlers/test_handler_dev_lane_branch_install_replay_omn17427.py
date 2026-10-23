# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev-lane branch install survives a refresh across a store reload (OMN-17427).

The record is written to disk by one store, the refresh recreates the container, and a
second store opened over the same directory (the deploy agent's next process) reads the
record back and puts the file back at its recorded sha256.
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
TARGET = "/app/.venv/lib/python3.12/site-packages/omnimarket/writer.py"
DEV_COPY = b"# dev's copy\n"
BRANCH_COPY = b"# the branch's copy\n"
NOW = datetime(2026, 10, 22, 11, 0, tzinfo=UTC)


class _Docker:
    """One dev-lane runtime container; ``recreate`` replays the refresh."""

    def __init__(self) -> None:
        self.generation = 0
        self.files: dict[str, bytes] = {}
        self.restarts: list[str] = []
        self.recreate()

    def recreate(self) -> None:
        self.generation += 1
        self.container = f"{self.generation:064x}"
        self.files = {TARGET: DEV_COPY}

    def __call__(
        self, argv: list[str], timeout: float = 0.0
    ) -> subprocess.CompletedProcess[str]:
        def done(code: int = 0, out: str = "") -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, code, out, "")

        verb = argv[1]
        if verb == "ps":
            return done(out=self.container + "\n")
        if verb == "exec":
            command = argv[2:]
            if command[:2] == ["-u", "0"]:
                command = command[2:]
            command = command[1:]
            if command[:2] == ["sha256sum", "--"]:
                data = self.files.get(command[2])
                if data is None:
                    return done(code=1)
                return done(out=f"{hashlib.sha256(data).hexdigest()}  {command[2]}\n")
            if command[:1] == ["stat"]:
                return done(out="1000:1000 644\n")
            return done()
        if verb == "cp":
            self.files[argv[3].partition(":")[2]] = Path(argv[2]).read_bytes()
            return done()
        if verb == "restart":
            self.restarts.append(argv[2])
            return done()
        if verb == "inspect":
            return done(out="running|healthy\n")
        raise AssertionError(f"unexpected command {argv}")


def test_a_registered_install_is_reapplied_by_a_fresh_store_after_the_refresh(
    tmp_path: Path,
) -> None:
    source = tmp_path / "branch" / "writer.py"
    source.parent.mkdir()
    source.write_bytes(BRANCH_COPY)
    state = tmp_path / "state"
    register(
        DevLaneBranchInstallStore(state),
        lane="dev",
        install_id="replay-omn17427",
        repo="omnimarket",
        pr=4312,
        branch="jonah/omn-17427-replay",
        head="8020dbdcc9a4961cdc5684d14c63b3bc024eef30",
        ticket="OMN-17427",
        registered_by="replay-test",
        services=[SERVICE],
        sources={TARGET: source},
        expires_at=NOW + timedelta(hours=48),
        restart=True,
        replace=False,
        now=NOW,
    )

    docker = _Docker()
    docker.recreate()
    assert docker.files[TARGET] == DEV_COPY

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
    assert docker.restarts == [docker.container]
    assert handler.check().ok
