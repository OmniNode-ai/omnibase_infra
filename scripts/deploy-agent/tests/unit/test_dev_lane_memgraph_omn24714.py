# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24714 -- a dev deploy through the agent must start omnibase-infra-memgraph.

``docker/docker-compose.infra.yml`` declares the service under the compose
profiles ``omnimemory`` and ``full``. The agent starts the runtime family by
name under ``--profile runtime`` and the dependency set by ``--profile core``
with no service names, which compose resolves to every service that carries NO
profile. A service in neither reached no deploy, so the architecture graph on
dev had no Memgraph to answer from.

The dev-lane overlay empties the service's profile list, which makes it a
dependency of that lane and starts it in the core phase. These tests pin the
whole path: the base declaration is the gap (positive control), the dev lane's
merged declaration carries no profile, no other lane's does, the agent's core
``up`` names no service and so reaches every un-profiled one, and the second dev
instance scales it to zero. Every other compose file that layers the dev-lane
overlay (found by reading the files, not from the routing table, which omits a
lane) must fence it too. One test runs the real compose binary over a stub,
so the ``!override []`` semantics are compose's own and not this file's.

Scope of the start: the service is started only by a deploy whose argv runs the
``--profile core`` leg with no service names, that is a ``scope=core`` command
and the core leg of a ``scope=full`` deploy. A ``scope=runtime`` deploy names the
runtime family under ``--profile runtime`` and NEVER starts it. The orchestrator
emits ``full`` (``executor.py`` ``observe_deps_convergence``), so a full deploy
is the route that starts it in production.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from deploy_agent.events import EnumRuntimeLane, Scope
from deploy_agent.executor import (
    DeployExecutor,
    dev_instance_lane_configs,
    lane_config_for,
    select_dev_instance,
)

from .test_dev_202_executor_lane_omn19522 import _capture_compose_up
from .test_full_rebuild_converges_deps_omn18640 import (
    _capture_full_rebuild,
    _deps_leg,
)

pytestmark = pytest.mark.unit

MEMGRAPH = "omnibase-infra-memgraph"
_DEFAULT_INSTANCE = "dev-201"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOCKER = _REPO_ROOT / "docker"


class _OverrideLoader(yaml.SafeLoader):
    """Safe YAML that reads compose's ``!override`` tag as the value it wraps."""


def _override(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    raise yaml.YAMLError(f"unsupported !override node {node!r}")


_OverrideLoader.add_constructor("!override", _override)


def _services(path: Path) -> dict[str, dict[str, Any]]:
    loaded = _OverrideLoader(path.read_text()).get_single_data()
    return {name: spec or {} for name, spec in loaded["services"].items()}


def _tree_file(file: str) -> Path:
    """The lane config names the registry checkout; read this tree's copy of it."""
    return _DOCKER / Path(file).name


def _merged_profiles(files: Sequence[str]) -> list[str]:
    """The service's profiles after the lane's compose files are merged in order."""
    profiles: list[str] = []
    for file in files:
        spec = _services(_tree_file(file)).get(MEMGRAPH)
        if spec is not None and "profiles" in spec:
            profiles = list(spec["profiles"])
    return profiles


def _merged_scale(files: Sequence[str]) -> int | None:
    scale: int | None = None
    for file in files:
        spec = _services(_tree_file(file)).get(MEMGRAPH)
        if spec is not None and "scale" in spec:
            scale = int(spec["scale"])
    return scale


#: Profiles the dev deploy agent activates: ``--profile core`` and ``--profile runtime``.
_STARTED_PROFILES = frozenset({"core", "runtime"})

#: A compose file layers the dev-lane overlay when its header gives the command
#: ``-f docker/docker-compose.dev-lane.yml`` or says it is layered over it. Other
#: files only mention the overlay in a comment.
_LAYERS_DEV_LANE = re.compile(
    r"-f\s+docker/docker-compose\.dev-lane\.yml"
    r"|Layered[^\n]*over docker-compose\.infra\.yml and docker-compose\.dev-lane\.yml"
)


def _layering_overlays() -> list[str]:
    """Every compose file in ``docker/`` that layers the dev-lane overlay."""
    return [
        path.name
        for path in sorted(_DOCKER.glob("docker-compose*.yml"))
        if path.name != "docker-compose.dev-lane.yml"
        and _LAYERS_DEV_LANE.search(path.read_text(encoding="utf-8"))
    ]


def _is_fenced(files: Sequence[str]) -> bool:
    """True when no dev deploy's ``core`` or ``runtime`` leg starts the service."""
    if _merged_scale(files) == 0:
        return True
    profiles = _merged_profiles(files)
    return bool(profiles) and not set(profiles) & _STARTED_PROFILES


def test_memgraph_base_declaration_is_outside_every_deploy_profile() -> None:
    """Positive control: the gap is real, the base file alone starts it nowhere."""
    base = _services(_DOCKER / "docker-compose.infra.yml")[MEMGRAPH]
    assert set(base["profiles"]) == {"omnimemory", "full"}


def test_memgraph_dev_lane_merge_has_no_profile() -> None:
    config = lane_config_for(EnumRuntimeLane.DEV)
    assert _merged_profiles(config.compose_files) == []


@pytest.mark.parametrize("lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST])
def test_memgraph_is_started_by_no_other_lane(lane: EnumRuntimeLane) -> None:
    config = lane_config_for(lane)
    assert _merged_profiles(config.compose_files) == ["omnimemory", "full"]


def test_memgraph_dev_core_up_names_no_service_so_it_reaches_the_dependency_set() -> (
    None
):
    select_dev_instance(_DEFAULT_INSTANCE)
    up = _capture_compose_up(DeployExecutor(), Scope.CORE)[0]
    assert up[up.index("--profile") + 1] == "core"
    assert "--no-deps" not in up
    assert MEMGRAPH not in up, "the core phase starts it as a dependency, never by name"


@pytest.mark.parametrize(
    "instance", sorted(set(dev_instance_lane_configs()) - {_DEFAULT_INSTANCE})
)
def test_memgraph_every_other_dev_instance_scales_it_to_zero(instance: str) -> None:
    select_dev_instance(instance)
    try:
        files = lane_config_for(EnumRuntimeLane.DEV).compose_files
        assert _merged_profiles(files) == []
        assert _merged_scale(files) == 0
    finally:
        select_dev_instance(_DEFAULT_INSTANCE)
    default_files = dev_instance_lane_configs()[_DEFAULT_INSTANCE].compose_files
    assert _merged_scale(default_files) is None


def test_memgraph_layering_overlays_are_found_by_reading_the_files() -> None:
    """Positive control: the enumeration finds the known lanes, so it cannot go
    vacuous, and omits the dev-lane file itself and the primary dev instance's own
    overlay, which is the lane that starts the service."""
    found = _layering_overlays()
    assert "docker-compose.prepr.yml" in found
    assert len(found) >= 3
    assert "docker-compose.dev-tenant-fold.yml" not in found
    assert "docker-compose.dev-lane.yml" not in found


@pytest.mark.parametrize("overlay", _layering_overlays())
def test_memgraph_every_overlay_that_layers_the_dev_lane_fences_it(
    overlay: str,
) -> None:
    """The dev-lane file empties the profile list for every composition that
    layers it, so each layering overlay must fence the service itself (scale 0 or
    an unreachable profile). One such lane is not in the routing table, so a test over
    ``dev_instance_lane_configs()`` alone never saw it start a 4 GiB graph store on
    a host whose overlay promises loopback-only ports."""
    files = ["docker-compose.infra.yml", "docker-compose.dev-lane.yml", overlay]
    assert _is_fenced(files), (
        f"{overlay} layers docker-compose.dev-lane.yml and does not fence {MEMGRAPH}: "
        f"merged profiles {_merged_profiles(files)}, scale {_merged_scale(files)}"
    )


def _require_docker_binary() -> None:
    """``compose config --services`` needs the binary, not a daemon."""
    if shutil.which("docker") is None:
        pytest.skip("requires the docker binary on PATH")


def test_memgraph_real_compose_test_skips_when_there_is_no_docker_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a host with no docker binary the real-compose test skips; it does not
    raise ``FileNotFoundError`` and turn the unit suite red."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert shutil.which("docker") is None
    with pytest.raises(pytest.skip.Exception):
        test_memgraph_override_semantics_under_real_compose(tmp_path)


def test_memgraph_full_scope_deploy_starts_it_through_the_converge_core_leg() -> None:
    """The orchestrator emits ``full``. Its core leg is ``--profile core`` with no
    service names and converges (no force recreate), so the service is started as
    a dependency and left alone while it runs. A ``scope=runtime`` deploy names the
    runtime family under ``--profile runtime`` and never starts it."""
    select_dev_instance(_DEFAULT_INSTANCE)
    leg = _deps_leg(_capture_full_rebuild(EnumRuntimeLane.DEV))
    assert leg[leg.index("--profile") + 1] == "core"
    assert "--force-recreate" not in leg
    assert "--no-deps" not in leg
    assert MEMGRAPH not in leg, "the core leg starts it as a dependency, never by name"
    runtime = _capture_compose_up(DeployExecutor(), Scope.RUNTIME)[0]
    assert runtime[runtime.index("--profile") + 1] == "runtime"
    assert MEMGRAPH not in runtime


def test_memgraph_override_semantics_under_real_compose(tmp_path: Path) -> None:
    """compose itself: base profiles hide the service, the overlay's empty list shows it."""
    _require_docker_binary()
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "postgres": {"image": "busybox:latest"},
                    MEMGRAPH: {
                        "image": "busybox:latest",
                        "profiles": ["omnimemory", "full"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(f"services:\n  {MEMGRAPH}:\n    profiles: !override []\n")

    def services(*files: Path) -> set[str]:
        argv = ["docker", "compose"]
        for file in files:
            argv += ["-f", str(file)]
        argv += ["--profile", "core", "config", "--services"]
        done = subprocess.run(
            argv, capture_output=True, text=True, check=True, timeout=60
        )
        return set(done.stdout.split())

    assert services(base) == {"postgres"}
    assert services(base, overlay) == {"postgres", MEMGRAPH}
