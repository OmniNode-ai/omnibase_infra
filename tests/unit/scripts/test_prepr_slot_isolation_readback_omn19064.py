# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A pre-PR slot's run reads its own isolation (OMN-19064).

AC-1: a branch build on a slot touches no other compose project, read by
container id, start time and restart count of every declared lane before and
after the run. AC-2: the slot's project publishes exactly its reserved port
block. The on-host halves are shell functions in prepr_pool_prove.sh, run here
against a fake ``docker``; the judge half is prepr_runtime_pool.py.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PY = REPO_ROOT / "scripts" / "runtime_build" / "prepr_runtime_pool.py"
PROVE_SH = REPO_ROOT / "scripts" / "runtime_build" / "prepr_pool_prove.sh"
POOL_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prepr_runtime_pool.yaml"

pytestmark = pytest.mark.unit

SLOT_1_BLOCK = "28085|28086|28090|23002"

# A docker that answers `ps -aq --filter label=...=<project>` from
# $FAKE/<project>.ids, `inspect -f '<project> {{...}}' <ids>` from $FAKE/ids/<id>
# ("<name> <started> <restarts>"), and `ps --format {{.Ports}}` from $FAKE/ports.
FAKE_DOCKER = r"""#!/bin/bash
if [ "$1" = ps ] && [ "$2" = -aq ]; then
  proj=${4#label=com.docker.compose.project=}
  cat "$FAKE/$proj.ids" 2>/dev/null
  exit 0
fi
if [ "$1" = ps ]; then cat "$FAKE/ports" 2>/dev/null; exit 0; fi
if [ "$1" = inspect ]; then
  proj=${3%% *}; shift 3
  for id in "$@"; do
    read -r name started restarts < "$FAKE/ids/$id"
    echo "$proj /$name $id $started $restarts"
  done
  exit 0
fi
exit 1
"""


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pool = _load("prepr_runtime_pool", POOL_PY)


def _functions(marker: str) -> str:
    text = PROVE_SH.read_text(encoding="utf-8")
    begin, end = f"# --- {marker} begin ---", f"# --- {marker} end ---"
    assert begin in text and end in text, marker
    return text[text.index(begin) : text.index(end)]


def _fake(tmp_path: Path) -> Path:
    fake = tmp_path / "fake"
    (fake / "ids").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER, encoding="utf-8")
    docker.chmod(0o755)
    return fake


def _container(fake: Path, project: str, cid: str, name: str, started: str) -> None:
    with (fake / f"{project}.ids").open("a", encoding="utf-8") as fh:
        fh.write(cid + "\n")
    (fake / "ids" / cid).write_text(f"{name} {started} 0\n", encoding="utf-8")


def _bash(tmp_path: Path, script: str, **env: str) -> str:
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
            "FAKE": str(tmp_path / "fake"),
            **env,
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def _neighbour_run(tmp_path: Path, fake: Path, change: str) -> str:
    """snap-pre, an optional change to the host, snap-post: the diff it prints."""
    funcs = _functions("neighbour-readback")
    env = {"NEIGHBOUR_PROJECTS": "lane-a|lane-b"}
    pre, post = tmp_path / "pre.txt", tmp_path / "post.txt"
    _bash(tmp_path, f"{funcs}\nneighbour_snapshot > {pre}", **env)
    if change:
        _bash(tmp_path, change)
    _bash(tmp_path, f"{funcs}\nneighbour_snapshot > {post}", **env)
    return _bash(tmp_path, f"{funcs}\nneighbour_diff {pre} {post}", **env)


def _neighbours(tmp_path: Path) -> Path:
    fake = _fake(tmp_path)
    _container(fake, "lane-a", "aaa1", "omninode-runtime", "T1")
    _container(fake, "lane-b", "bbb1", "stab-runtime", "T2")
    _container(fake, "slot-x", "ccc1", "slot-runtime", "T3")
    return fake


def test_an_untouched_host_reads_zero_changes_over_a_non_zero_neighbour_count(
    tmp_path: Path,
) -> None:
    fake = _neighbours(tmp_path)
    out = _neighbour_run(tmp_path, fake, "")
    # the count is the positive control: two declared-lane containers were read
    assert out.splitlines()[0] == "neighbour-diff=0 neighbours=2", out


def test_a_restarted_declared_lane_container_is_named_by_id(tmp_path: Path) -> None:
    fake = _neighbours(tmp_path)
    out = _neighbour_run(tmp_path, fake, f"echo 'stab-runtime T9 1' > {fake}/ids/bbb1")
    assert out.splitlines()[0] == "neighbour-diff=2 neighbours=2", out
    assert "neighbour-changed > lane-b /stab-runtime bbb1 T9 1" in out


def test_a_recreated_declared_lane_container_is_named_by_its_new_id(
    tmp_path: Path,
) -> None:
    fake = _neighbours(tmp_path)
    out = _neighbour_run(
        tmp_path,
        fake,
        f"echo aaa2 > {fake}/lane-a.ids; echo 'omninode-runtime T5 0' > {fake}/ids/aaa2",
    )
    assert out.splitlines()[0] == "neighbour-diff=2 neighbours=2", out
    assert "neighbour-changed > lane-a /omninode-runtime aaa2 T5 0" in out


def test_the_slot_s_own_containers_are_not_neighbours(tmp_path: Path) -> None:
    fake = _neighbours(tmp_path)
    out = _neighbour_run(tmp_path, fake, f"echo 'slot-runtime T8 0' > {fake}/ids/ccc1")
    assert out.splitlines()[0] == "neighbour-diff=0 neighbours=2", out


def _ports(tmp_path: Path, published: str) -> str:
    fake = _fake(tmp_path)
    (fake / "ports").write_text(published, encoding="utf-8")
    script = f"{_functions('slot-ports-readback')}\nslot_ports_readback"
    return _bash(tmp_path, script, P="slot-x", PORTS=SLOT_1_BLOCK).strip()


def test_a_slot_publishing_only_its_block_reads_nothing_outside(tmp_path: Path) -> None:
    out = _ports(
        tmp_path,
        "127.0.0.1:28085->8085/tcp\n127.0.0.1:28086->8085/tcp\n"
        "8085/tcp, 127.0.0.1:23002->3002/tcp\n\n",
    )
    assert out == "slot-published-ports=23002,28085,28086 outside-block=none"


def test_a_port_outside_the_block_is_named(tmp_path: Path) -> None:
    # the shape found live 2026-10-19: a project beside the dev lane
    # publishing 28098, which no slot policy reserves
    out = _ports(
        tmp_path,
        "127.0.0.1:28085->8085/tcp\n8085/tcp, 127.0.0.1:28098->8098/tcp\n"
        "0.0.0.0:5436->5432/tcp, [::]:5436->5432/tcp\n",
    )
    assert out == "slot-published-ports=28085,28098,5436 outside-block=28098,5436"


def test_a_port_range_is_reported_outside_as_written(tmp_path: Path) -> None:
    out = _ports(tmp_path, "0.0.0.0:28085-28086->8085-8086/tcp\n")
    assert out == "slot-published-ports=28085-28086 outside-block=28085-28086"


def test_a_slot_publishing_nothing_reads_none(tmp_path: Path) -> None:
    out = _ports(tmp_path, "8085/tcp\n")
    assert out == "slot-published-ports=none outside-block=none"


def _rb() -> Any:
    return pool.Readback(
        checks={"stack_built": True}, notes=[], restored=True, residue=""
    )


@pytest.mark.parametrize(
    ("probe", "ok", "note"),
    [
        (
            "slot-published-ports=23002,28085,28086 outside-block=none",
            True,
            "all in the reserved block",
        ),
        (
            "slot-published-ports=28085,28098 outside-block=28098",
            False,
            "published 28098 outside the reserved block",
        ),
        (
            "slot-published-ports=none outside-block=none",
            False,
            "published no host port",
        ),
        (
            "port 28085 HTTP 200 status healthy healthy True",
            False,
            "no slot-published-ports",
        ),
    ],
)
def test_the_judge_holds_a_slot_to_its_reserved_ports(
    probe: str, ok: bool, note: str
) -> None:
    rb = _rb()
    pool.judge_slot_isolation(rb, {"probe": probe})
    assert rb.checks["slot_ports_in_block"] is ok
    assert any(note in n for n in rb.notes), rb.notes


@pytest.mark.parametrize(
    ("post", "note"),
    [
        (
            "neighbour-diff=0 neighbours=40",
            "40 declared-lane containers, none recreated",
        ),
        ("neighbour-diff=0 neighbours=0", "no declared-lane container was read"),
        ("snapshot-diff=0", "no neighbour readback"),
        (
            "neighbour-diff=1 neighbours=40\nneighbour-changed > lane-a /x id2 T 0",
            "1 changed lines across 40",
        ),
    ],
)
def test_the_judge_reads_the_neighbours_into_the_readback(post: str, note: str) -> None:
    rb = _rb()
    pool.judge_slot_isolation(rb, {"snap-post": post})
    assert any(note in n for n in rb.notes), rb.notes
    assert "neighbours_untouched" not in rb.checks
    if "neighbour-changed" in post:
        assert "neighbour-changed > lane-a /x id2 T 0" in rb.notes


def test_the_driver_passes_every_declared_lane_to_a_slot_and_none_to_an_isolated_member() -> (
    None
):
    cfg = pool.load_pool_config(POOL_FIXTURE)
    slot = next(h for h in cfg.hosts if h.kind == "prepr-slot")
    isolated = next(h for h in cfg.hosts if h.kind != "prepr-slot")
    declared = pool._slot_policy().DECLARED_LANE_PROJECTS
    names = pool.neighbour_projects(slot).split("|")
    # the AC-1 set (dev, stability, judge, collaborator) is the policy's
    # declared-lane table, and no slot is its own neighbour
    assert names and sorted(names) == sorted(declared)
    slots = {sp.compose_project for sp in pool._slot_policy().SLOTS.values()}
    assert not slots & set(names)
    assert pool.neighbour_projects(isolated) == ""
    assert "NEIGHBOUR_PROJECTS" in pool.RESERVED_PARAMS
