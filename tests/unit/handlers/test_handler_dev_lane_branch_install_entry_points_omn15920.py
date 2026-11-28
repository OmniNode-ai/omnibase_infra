# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev branch install writes a declared entry point into the installed dist-info (OMN-15920).

The runtime finds nodes through the ``onex.nodes`` entry points of the installed
distributions (``read_package_inventory``). omnimarket#6933 adds a node and its
``pyproject.toml`` entry point, and a file install copies source only, so the node
is on disk but never discovered. The fake Docker below runs the installer's own
program with a real interpreter against a real dist-info directory.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from importlib.metadata import PathDistribution
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_dev_lane_branch_install import (
    DevLaneBranchInstallStore,
    HandlerDevLaneBranchInstall,
    main,
    register,
)
from omnibase_infra.models.model_dev_lane_branch_install_entry_point import (
    ModelDevLaneBranchInstallEntryPoint,
)

pytestmark = pytest.mark.unit

PROJECT = "omnibase-infra"
SERVICE = "omninode-runtime"
PKG = "/app/.venv/lib/python3.12/site-packages/omnimarket"
FILE = f"{PKG}/nodes/node_mcp_server_effect/__init__.py"
HEAD = "1a449d5fda4f138ccf713e75d5b61612c794a39f"
NOW = datetime(2026, 11, 28, 4, 30, tzinfo=UTC)
EXISTING = "node_projection_mcp_tools = omnimarket.nodes.node_projection_mcp_tools"
EP = ModelDevLaneBranchInstallEntryPoint(
    dist="omnimarket",
    group="onex.nodes",
    name="node_mcp_server_effect",
    value="omnimarket.nodes.node_mcp_server_effect",
)


def _done(
    argv: list[str], code: int = 0, out: str = "", err: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, code, out, err)


def _make_dist(site: Path, text: str) -> Path:
    dist_info = site / "omnimarket-0.4.330.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: omnimarket\nVersion: 0.4.330\n"
    )
    (dist_info / "RECORD").write_text(
        "omnimarket-0.4.330.dist-info/METADATA,,\n"
        "omnimarket-0.4.330.dist-info/RECORD,,\n"
        "omnimarket-0.4.330.dist-info/entry_points.txt,,\n"
    )
    entry_points = dist_info / "entry_points.txt"
    entry_points.write_text(text)
    return entry_points


class _Lane:
    """A fake Docker: one container; its python is this interpreter on a fake site dir."""

    def __init__(self, site: Path) -> None:
        self.container = f"{1:064x}"
        self.site = site
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {"/app", PKG, f"{PKG}/nodes"}
        self.ep_writes = 0

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
            if command[:1] == ["/app/.venv/bin/python"]:
                if "tempfile" in command[2]:
                    self.ep_writes += 1
                real = subprocess.run(
                    [sys.executable, *command[1:]],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "PYTHONPATH": str(self.site)},
                    check=False,
                )
                return _done(argv, real.returncode, real.stdout, real.stderr)
            if command[:3] == ["mkdir", "-p", "--"]:
                self.dirs.add(command[3])
                return _done(argv)
            if command[:2] == ["sha256sum", "--"]:
                path = command[2]
                if path not in self.files:
                    return _done(argv, code=1, err="No such file or directory")
                digest = hashlib.sha256(self.files[path]).hexdigest()
                return _done(argv, out=f"{digest}  {path}\n")
            if command[:1] == ["stat"]:
                return _done(argv, code=1)
            if command[:1] in (["chown"], ["chmod"]):
                return _done(argv)
            raise AssertionError(f"unexpected exec {command[:1]}")
        if argv[:2] == ["docker", "cp"]:
            path = argv[3].partition(":")[2]
            self.files[path] = Path(argv[2]).read_bytes()
            return _done(argv)
        if argv[:2] == ["docker", "inspect"]:
            return _done(argv, out="running|healthy\n")
        if argv[:2] == ["docker", "restart"]:
            return _done(argv)
        raise AssertionError(f"unexpected command {argv}")


def _declare(
    tmp_path: Path,
    store: DevLaneBranchInstallStore,
    entry_points: tuple[ModelDevLaneBranchInstallEntryPoint, ...],
) -> None:
    source = tmp_path / "branch" / "init.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"# the branch's new node\n")
    register(
        store,
        lane="dev",
        install_id="mcp-node-omn15920",
        repo="omnimarket",
        pr=6933,
        branch="omn-15920-mcp-tools-list-node",
        head=HEAD,
        ticket="OMN-15920",
        registered_by="q-mcplive3-052000",
        services=[SERVICE],
        sources={FILE: source},
        expires_at=NOW + timedelta(hours=24),
        restart=False,
        entry_points=entry_points,
        now=NOW,
    )


def _listed(entry_points: Path) -> dict[str, str]:
    dist = PathDistribution(entry_points.parent)
    return {e.name: e.value for e in dist.entry_points if e.group == "onex.nodes"}


def _handler(
    store: DevLaneBranchInstallStore, lane: _Lane
) -> HandlerDevLaneBranchInstall:
    return HandlerDevLaneBranchInstall(
        store, PROJECT, run=lane, sleep=lambda _s: None, clock=lambda: NOW
    )


def test_the_declared_entry_point_is_written_and_read_back(tmp_path: Path) -> None:
    ep_file = _make_dist(
        tmp_path / "site",
        f"[console_scripts]\nonex = a.b:main\n\n[onex.nodes]\n{EXISTING}\n\n[other]\nx = y\n",
    )
    store = DevLaneBranchInstallStore(tmp_path / "state")
    _declare(tmp_path, store, (EP,))
    lane = _Lane(tmp_path / "site")
    receipt = _handler(store, lane).apply()
    assert receipt.ok
    listed = _listed(ep_file)
    assert listed["node_mcp_server_effect"] == "omnimarket.nodes.node_mcp_server_effect"
    assert "node_projection_mcp_tools" in listed
    text = ep_file.read_text()
    assert text.index("node_mcp_server_effect") < text.index("[other]")
    assert "[console_scripts]\nonex = a.b:main" in text


def test_a_second_apply_writes_nothing_and_a_dropped_entry_is_reapplied(
    tmp_path: Path,
) -> None:
    ep_file = _make_dist(tmp_path / "site", f"[onex.nodes]\n{EXISTING}\n")
    store = DevLaneBranchInstallStore(tmp_path / "state")
    _declare(tmp_path, store, (EP,))
    lane = _Lane(tmp_path / "site")
    handler = _handler(store, lane)
    assert handler.apply().ok
    assert lane.ep_writes == 1
    assert handler.apply().ok
    assert lane.ep_writes == 1
    ep_file.write_text(f"[onex.nodes]\n{EXISTING}\n")  # a recreate resets the file
    assert not handler.check().ok
    assert handler.apply().ok
    assert lane.ep_writes == 2
    assert "node_mcp_server_effect" in _listed(ep_file)


def test_a_missing_entry_point_fails_the_install_while_the_files_are_present(
    tmp_path: Path,
) -> None:
    """Known bad: the file-only install reads whole although the node is never discovered."""
    ep_file = _make_dist(tmp_path / "site", f"[onex.nodes]\n{EXISTING}\n")
    store = DevLaneBranchInstallStore(tmp_path / "state")
    _declare(tmp_path, store, (EP,))
    lane = _Lane(tmp_path / "site")
    lane.files[FILE] = b"# the branch's new node\n"
    assert not _handler(store, lane).check().ok
    assert "node_mcp_server_effect" not in _listed(ep_file)


def test_an_install_without_entry_points_is_unchanged(tmp_path: Path) -> None:
    _make_dist(tmp_path / "site", f"[onex.nodes]\n{EXISTING}\n")
    store = DevLaneBranchInstallStore(tmp_path / "state")
    _declare(tmp_path, store, ())
    lane = _Lane(tmp_path / "site")
    assert _handler(store, lane).apply().ok
    assert lane.ep_writes == 0


@pytest.mark.parametrize(
    "value",
    ["a b", "x;rm -rf /", "../x", "mod:attr:more", ""],
)
def test_a_value_that_is_not_a_module_path_is_refused(value: str) -> None:
    with pytest.raises(ValueError):
        ModelDevLaneBranchInstallEntryPoint(
            dist="omnimarket", group="onex.nodes", name="n", value=value
        )


def test_the_register_flag_parses_and_a_malformed_one_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "f.py"
    source.write_text("x = 1\n")
    base = [
        "register", "--state-dir", str(tmp_path / "st"), "--lane", "dev",
        "--id", "mcp-node-omn15920", "--repo", "omnimarket", "--pr", "6933",
        "--branch", "b", "--head", HEAD, "--ticket", "OMN-15920",
        "--registered-by", "q-mcplive3-052000", "--service", SERVICE,
        "--file", f"{FILE}={source}", "--expires-in-hours", "24",
    ]  # fmt: skip
    ok = main(
        [
            *base,
            "--entry-point",
            f"omnimarket:onex.nodes:node_mcp_server_effect={EP.value}",
        ],
        clock=lambda: NOW,
    )
    out = capsys.readouterr().out
    assert ok == 0
    assert '"name":"node_mcp_server_effect"' in out
    bad = main(
        [*base, "--replace", "--entry-point", "no-equals-sign"], clock=lambda: NOW
    )
    assert bad != 0
    assert "must be DIST:GROUP:NAME=VALUE" in capsys.readouterr().out
