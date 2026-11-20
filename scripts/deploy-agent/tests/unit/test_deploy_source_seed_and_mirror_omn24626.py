# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy-source clone is seeded at the served commit; the mirror fallback holds on every fetch (OMN-24626).

Review findings MED 1 and MED 2 of omnibase_infra#6027.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .test_agent_own_clone_omn24562 import _commit, _ensure, _git, _World, dsc

pytestmark = pytest.mark.unit


@pytest.fixture
def world(tmp_path: Path) -> _World:
    return _World(tmp_path)


def test_seed_at_the_served_commit_not_the_canonical_head(world: _World) -> None:
    local = _commit(world.canonical, "local-only")
    assert _ensure(world.settings(served_sha=world.pinned_sha)).created is True
    head = _git(world.agent_clone, "rev-parse", "HEAD")
    assert head == world.pinned_sha
    assert head != local
    assert _git(world.canonical, "rev-parse", "HEAD") == local


def test_seed_without_a_readable_served_commit_is_the_tracked_ref_head_on_origin(
    world: _World,
) -> None:
    _commit(world.canonical, "local-only")
    assert _ensure(world.settings()).created is True
    assert _git(world.agent_clone, "rev-parse", "HEAD") == world.dev_sha
    assert _git(world.agent_clone, "symbolic-ref", "--short", "HEAD") == "dev"


def _mirrored_clone(world: _World) -> tuple[object, Path]:
    mirror = world.root / "mirror.git"
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(world.origin), str(mirror)], check=True
    )
    settings = world.settings(str(mirror))
    assert _ensure(settings).created is True
    assert dsc.read_mirror_fallback_count(world.agent_clone) == 0
    return settings, mirror


def test_mirror_that_stops_answering_on_a_later_fetch_falls_back_to_origin_and_counts(
    world: _World, caplog: pytest.LogCaptureFixture
) -> None:
    settings, mirror = _mirrored_clone(world)
    subprocess.run(["rm", "-rf", str(mirror)], check=True)
    advanced = _commit(world.canonical, "after-mirror")
    _git(world.canonical, "push", "-q", "origin", "dev")
    with caplog.at_level("WARNING"):
        result = dsc.fetch(settings, run=dsc.local_runner(world.agent_clone, 60))
    assert result.returncode == 0, result.stderr
    assert dsc.read_mirror_fallback_count(world.agent_clone) == 1
    assert "fallback to origin" in caplog.text
    assert _git(world.agent_clone, "rev-parse", "refs/remotes/origin/dev") == advanced


def test_an_answering_mirror_serves_the_fetch_and_counts_nothing(
    world: _World,
) -> None:
    settings, _ = _mirrored_clone(world)
    assert (
        dsc.fetch(settings, run=dsc.local_runner(world.agent_clone, 60)).returncode == 0
    )
    assert dsc.read_mirror_fallback_count(world.agent_clone) == 0


def test_a_dead_mirror_does_not_fail_the_remote_reachability_probe(
    world: _World,
) -> None:
    settings, mirror = _mirrored_clone(world)
    subprocess.run(["rm", "-rf", str(mirror)], check=True)
    assert (
        dsc.remote_reachable(settings, run=dsc.local_runner(world.agent_clone, 30))
        is True
    )
    assert dsc.read_mirror_fallback_count(world.agent_clone) == 1
