# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev branch installer installs a declared overlay root (OMN-25536).

The runtime reads overlays from the roots ``ONEX_SKILL_OVERLAY_ROOTS`` names. Two
of them are read-only mounts (the overlays repository store and the frozen
config root), so a branch install cannot write there. The third root is a
directory in the container's writable layer, declared in
``dev_lane_branch_install_roots.yaml`` and named first in the compose variable.
A destination under it is accepted; any other path outside ``/app/`` is refused.
"""

from __future__ import annotations

import hashlib
import re
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

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT = "omnibase-infra"
SERVICE = "omninode-runtime"
OVERLAY_ROOT = "/etc/onex/skill-overlays-branch"
OVERLAY_FILE = f"{OVERLAY_ROOT}/node_example_compute/overlay.yaml"
HEAD = "8020dbdcc9a4961cdc5684d14c63b3bc024eef30"
NOW = datetime(2026, 11, 18, 1, 0, tzinfo=UTC)
BRANCH_COPY = b"schema_version: 1\n# the branch's overlay\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _done(
    argv: list[str], code: int = 0, out: str = "", err: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, code, out, err)


class _Lane:
    """A fake Docker: one container, a file map, and directories that must exist."""

    def __init__(self) -> None:
        self.container = f"{1:064x}"
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {"/app", "/etc", "/etc/onex"}
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
    source = tmp_path / "branch" / "overlay.yaml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(BRANCH_COPY)
    register(
        store,
        lane="dev",
        install_id="overlay-omn25536",
        repo="overlays",
        pr=1,
        branch="omn-25536-overlay",
        head=HEAD,
        ticket="OMN-25536",
        registered_by="overlayinstall-0105",
        services=[SERVICE],
        sources={path: source},
        expires_at=NOW + timedelta(hours=24),
        restart=False,
        now=NOW,
    )


def test_a_destination_under_the_declared_overlay_root_is_accepted(
    tmp_path: Path,
) -> None:
    store = DevLaneBranchInstallStore(tmp_path / "state")
    _declare(tmp_path, store, OVERLAY_FILE)
    (install,) = store.load().installs
    assert [f.path for f in install.files] == [OVERLAY_FILE]


@pytest.mark.parametrize(
    "path",
    [
        "/etc/onex/skill-overlays-store/node_example_compute/overlay.yaml",
        "/etc/onex/skill-overlays/node_example_compute/overlay.yaml",
        "/etc/passwd",
        "/etc/onex/skill-overlays-branch",
        "/etc/onex/skill-overlays-branch/",
        "/etc/onex/skill-overlays-branch/../skill-overlays-store/x.yaml",
        "/etc/onex/skill-overlays-branch/./x.yaml",
        "/etc/onex/skill-overlays-branch//x.yaml",
        "/etc/onex/skill-overlays-branchx/x.yaml",
        "/etc/onex/skill-overlays-branch/dir/",
        "relative/x.yaml",
    ],
)
def test_any_other_path_outside_the_app_tree_is_still_refused(
    tmp_path: Path, path: str
) -> None:
    store = DevLaneBranchInstallStore(tmp_path / "state")
    with pytest.raises(DevLaneBranchInstallRefusedError):
        _declare(tmp_path, store, path)
    assert store.load().installs == ()


def test_apply_creates_the_root_directory_then_reads_the_file_back(
    tmp_path: Path,
) -> None:
    store = DevLaneBranchInstallStore(tmp_path / "state")
    _declare(tmp_path, store, OVERLAY_FILE)
    lane = _Lane()
    handler = HandlerDevLaneBranchInstall(
        store, PROJECT, run=lane, sleep=lambda _s: None, clock=lambda: NOW
    )
    receipt = handler.apply()
    assert lane.files[OVERLAY_FILE] == BRANCH_COPY
    assert lane.mkdirs == [f"{OVERLAY_ROOT}/node_example_compute"]
    assert receipt.ok is True


def test_the_declared_roots_come_from_data_and_the_runtime_reads_each_overlay_one() -> (
    None
):
    from omnibase_infra.models.model_dev_lane_branch_install_roots import (
        load_dev_lane_branch_install_roots,
    )

    roots = load_dev_lane_branch_install_roots()
    overlay_paths = [r.path for r in roots.roots if r.overlay_root]
    assert overlay_paths == [f"{OVERLAY_ROOT}/"]
    assert "/app/" in [r.path for r in roots.roots]
    for compose in (
        "docker-compose.dev-lane.yml",
        "docker-compose.dev-tenant-fold.yml",
    ):
        text = (REPO_ROOT / "docker" / compose).read_text(encoding="utf-8")
        values = re.findall(r"^\s+ONEX_SKILL_OVERLAY_ROOTS: (\S+)$", text, re.MULTILINE)
        assert len(values) == 1, compose
        assert values[0].split(":")[0] == OVERLAY_ROOT, compose
