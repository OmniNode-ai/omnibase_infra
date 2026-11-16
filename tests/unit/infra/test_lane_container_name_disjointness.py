# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane container_name disjointness ratchet (OMN-13815).

Root cause guarded here: the base ``docker-compose.infra.yml`` hardcodes bare
(non-lane-prefixed) ``container_name`` values for runtime-tier services. The dev
lane (base file only) intentionally keeps those bare names. Every *non-dev* lane
must rename each runtime-reachable service with a lane-scoped prefix, otherwise
two lanes fight over the same host container name and ``docker compose up``
aborts the whole batch with a ``Conflict. The container name ... is already in
use`` error.

The stability-test and prod overlays disable most of these services via a
``*-disabled`` profile, but that is NOT sufficient protection: ``onex-runtime-deploy``
starts several of them *explicitly by name* (``up -d --no-deps --force-recreate
<service>``) which bypasses the profile gate. Explicitly-started services fall
back to the inherited bare ``container_name`` unless the overlay overrides it.
This test therefore ratchets on the *effective* container_name (overlay override
if present, else the base bare name) for every runtime-reachable service.

Static YAML parse only — no docker daemon required, so it runs in unit CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from deploy_agent import executor as agent_executor
from deploy_agent.events import EnumRuntimeLane, Scope
from deploy_agent.executor import (
    _LANE_CONFIGS,
    ModelLaneConfig,
    _requested_services_for_up,
    lane_config_for,
)


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Passthrough constructor for Docker Compose merge/override tags.

    Compose files carry tags (``!override``, ``!reset``, ``!!merge``) that plain
    ``yaml.safe_load`` cannot resolve. This unwraps them to the underlying value
    so static structural inspection works without a docker daemon.
    """
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """Test-local YAML loader with Docker Compose tag support."""


_ComposeLoader.add_constructor("!override", _construct_compose_value)
_ComposeLoader.add_constructor("!reset", _construct_compose_value)
# Catch-all for any remaining compose-only tag (e.g. explicit ``!!merge``).
# Standard scalar/map/seq tags keep their exact SafeLoader constructors; only
# tags with no exact constructor fall through to this prefix handler.
_ComposeLoader.add_multi_constructor("", _construct_compose_value)


DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"
DEPLOY_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "handlers"
    / "handler_runtime_deploy.sh"
)

BASE_FILE = DOCKER_DIR / "docker-compose.infra.yml"
STABILITY_FILE = DOCKER_DIR / "docker-compose.stability-test.yml"
PROD_FILE = DOCKER_DIR / "docker-compose.prod.yml"
JUDGE_FILE = DOCKER_DIR / "docker-compose.judge.yml"

# Overlay lanes the deploy agent drives. The agent hands its runtime up list to
# ``docker compose up -d --no-deps`` by name for every service its lane config does not
# list in ``disabled_services`` (``_requested_services_for_up`` in
# deploy_agent/executor.py), and an explicit name bypasses the overlay's profile-disable.
# A service in that list that the lane neither disables nor renames is created under the
# dev lane's bare name. The map is derived from the agent's own ``_LANE_CONFIGS``, so a
# lane added there is covered or has to be named below.
#
# The prod overlay is the one named exclusion: it still carries both runtime-list-only
# services on their bare names, and prod is outside this lane's reach. Taking it out of
# the exclusion list is the follow-up that adds the two names to prod's
# ``disabled_services``.
AGENT_RUNTIME_EXCLUSIONS = {
    EnumRuntimeLane.PROD: "prod's disabled_services is a prod config change, not made here",
}
AGENT_RUNTIME_LANES = {
    DOCKER_DIR / Path(config.compose_files[1]).name: lane
    for lane, config in _LANE_CONFIGS.items()
    if lane is not EnumRuntimeLane.DEV and lane not in AGENT_RUNTIME_EXCLUSIONS
}

# Layered overlays merge on top of the base infra file and inherit un-overridden
# fields (including container_name). The judge overlay is standalone (see its
# header) and self-defines every service, so it is validated separately.
LAYERED_OVERLAYS = {
    "stability-test": STABILITY_FILE,
    "prod": PROD_FILE,
}


def _load_services(path: Path) -> dict[str, dict]:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader)  # noqa: S506
    assert isinstance(data, dict)
    return data.get("services", {}) or {}


def _container_names(services: dict[str, dict]) -> dict[str, str]:
    return {
        name: cfg["container_name"]
        for name, cfg in services.items()
        if isinstance(cfg, dict) and cfg.get("container_name")
    }


def _runtime_services_from_deploy_script() -> set[str]:
    """Parse the explicit RUNTIME_SERVICES array from onex-runtime-deploy.

    These services are started by name, bypassing compose profile gating, so any
    bare container_name they inherit becomes a live host-name collision.
    """
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"RUNTIME_SERVICES=\((.*?)\)", text, re.DOTALL)
    assert match, "RUNTIME_SERVICES array not found in onex-runtime-deploy"
    body = match.group(1)
    return {
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


@pytest.fixture(autouse=True)
def _committed_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this checkout's runtime policy, not the deploy host's clone path."""
    monkeypatch.setattr(
        "deploy_agent.executor.RUNTIME_POLICY_ENV_FILE",
        DOCKER_DIR / "runtime-policy.env",
    )


def _agent_runtime_services(overlay_file: Path) -> set[str]:
    """What the deploy agent names for this overlay's lane, from its own list."""
    lane = AGENT_RUNTIME_LANES.get(overlay_file)
    if lane is None:
        return set()
    return set(_requested_services_for_up(Scope.RUNTIME, [], lane=lane))


def _instantiated_services(overlay_file: Path) -> set[str]:
    """Base services a *layered* lane actually brings up via the deploy path.

    A service is instantiated in the lane if either:
      * it is in onex-runtime-deploy's RUNTIME_SERVICES (started explicitly by name,
        which bypasses profile gating), or, for a lane the deploy agent drives
        (``AGENT_RUNTIME_LANES``), in the up list the agent builds for that lane
        (``_requested_services_for_up``), or
      * its *effective* profile (overlay override if present, else base) is empty
        (always-on) or contains ``runtime`` (the deploy ``--profile runtime`` set).

    A service the overlay disables (``profiles: !override [<lane>-disabled]``) and
    that is absent from RUNTIME_SERVICES is NOT instantiated: it never occupies a
    host container name in this lane, so it needs no lane-scoped rename and must
    NOT be declared in the desired-state census (that would be phantom drift).
    Omnimemory/full-only services (e.g. memgraph) are likewise never on the
    runtime deploy path and are excluded.
    """
    base_services = _load_services(BASE_FILE)
    overlay_services = _load_services(overlay_file)
    runtime_explicit = _runtime_services_from_deploy_script()
    runtime_explicit |= _agent_runtime_services(overlay_file)
    instantiated: set[str] = set()
    for name, base_cfg in base_services.items():
        if not isinstance(base_cfg, dict):
            continue
        overlay_cfg = overlay_services.get(name)
        if isinstance(overlay_cfg, dict) and "profiles" in overlay_cfg:
            profiles = overlay_cfg.get("profiles")
        else:
            profiles = base_cfg.get("profiles")
        if name in runtime_explicit or not profiles or "runtime" in profiles:
            instantiated.add(name)
    return instantiated


def _lane_names(overlay_file: Path) -> set[str]:
    """Effective container_names a layered lane can put on the host.

    Every instantiated base service (overlay override if present, else the bare
    base name) plus any overlay-only named service (e.g. runtime-worker, which the
    base leaves unnamed for compose to auto-name per project).
    """
    base_names = _container_names(_load_services(BASE_FILE))
    overlay_names = _container_names(_load_services(overlay_file))
    instantiated = _instantiated_services(overlay_file)
    names: set[str] = set()
    for service in instantiated:
        names.add(overlay_names.get(service, base_names.get(service, "")))
    names.discard("")
    # overlay-only named services (base has no explicit container_name)
    for service, name in overlay_names.items():
        if service not in base_names:
            names.add(name)
    return names


@pytest.mark.unit
def test_deploy_runtime_services_are_named_and_reachable() -> None:
    """Every explicitly-started runtime service resolves to a real base service."""
    runtime_explicit = _runtime_services_from_deploy_script()
    base_services = _load_services(BASE_FILE)
    assert runtime_explicit, "RUNTIME_SERVICES parsed empty"
    for service in runtime_explicit:
        assert service in base_services, (
            f"onex-runtime-deploy starts unknown service {service!r}"
        )


def _bare_survivors(overlay_file: Path) -> list[str]:
    """Instantiated services the overlay leaves on the dev lane's bare name."""
    instantiated = _instantiated_services(overlay_file)
    base_names = _container_names(_load_services(BASE_FILE))
    overlay_names = _container_names(_load_services(overlay_file))

    bare_survivors: list[str] = []
    for service in sorted(instantiated):
        if service not in base_names:
            # Base has no explicit container_name (e.g. runtime-worker) -> compose
            # auto-names it per compose-project, already lane-isolated.
            continue
        override = overlay_names.get(service)
        if override is None or override == base_names[service]:
            bare_survivors.append(service)
    return bare_survivors


@pytest.mark.unit
@pytest.mark.parametrize("lane", sorted(LAYERED_OVERLAYS))
def test_layered_overlay_renames_every_instantiated_service(lane: str) -> None:
    """No service a non-dev lane instantiates may keep its bare (dev/base) name."""
    bare_survivors = _bare_survivors(LAYERED_OVERLAYS[lane])

    assert not bare_survivors, (
        f"{lane} overlay instantiates {bare_survivors} but leaves them on the bare "
        f"(dev/base) container_name; onex-runtime-deploy starts these explicitly and "
        f"they collide with the dev lane. Add a lane-prefixed container_name override."
    )


@pytest.mark.unit
def test_instantiated_container_names_are_pairwise_disjoint() -> None:
    """base/dev, stability-test, prod, judge must not share any container_name."""
    base_names = _container_names(_load_services(BASE_FILE))
    lane_sets: dict[str, set[str]] = {
        # dev == base file only; it runs the full set on the bare names.
        "base/dev": set(base_names.values()),
    }
    for lane, overlay_file in LAYERED_OVERLAYS.items():
        lane_sets[lane] = _lane_names(overlay_file)

    # Judge is a standalone stack: every one of its container_names is defined in
    # the judge file itself and must be judge-prefixed + disjoint from all lanes.
    lane_sets["judge"] = set(_container_names(_load_services(JUDGE_FILE)).values())

    lanes = sorted(lane_sets)
    for i, left in enumerate(lanes):
        for right in lanes[i + 1 :]:
            overlap = lane_sets[left] & lane_sets[right]
            assert not overlap, (
                f"container_name collision between {left} and {right} lanes: {overlap}"
            )


@pytest.mark.unit
def test_judge_container_names_are_lane_scoped() -> None:
    """Every judge container_name carries a judge-scoped token (standalone stack)."""
    judge_names = _container_names(_load_services(JUDGE_FILE))
    assert judge_names, "judge overlay defines no container_name services"
    for service, name in judge_names.items():
        assert "judge" in name, (
            f"judge service {service!r} container_name {name!r} is not judge-scoped"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lane", "service", "expected"),
    [
        # OMN-13815: the four RUNTIME_SERVICES consumers onex-runtime-deploy starts
        # explicitly in each non-dev layered lane (stability-test).
        (
            "stability-test",
            "agent-actions-consumer",
            "omninode-stability-test-agent-actions-consumer",
        ),
        (
            "stability-test",
            "skill-lifecycle-consumer",
            "omninode-stability-test-skill-lifecycle-consumer",
        ),
        (
            "stability-test",
            "omninode-contract-resolver",
            "omninode-stability-test-contract-resolver",
        ),
        # ... and prod.
        (
            "prod",
            "agent-actions-consumer",
            "omninode-prod-agent-actions-consumer",
        ),
        (
            "prod",
            "skill-lifecycle-consumer",
            "omninode-prod-skill-lifecycle-consumer",
        ),
        (
            "prod",
            "omninode-contract-resolver",
            "omninode-prod-contract-resolver",
        ),
    ],
)
def test_expected_lane_container_name_overrides(
    lane: str, service: str, expected: str
) -> None:
    """Regression anchor for the specific OMN-13815 overrides."""
    overlay_names = _container_names(_load_services(LAYERED_OVERLAYS[lane]))
    assert overlay_names.get(service) == expected, (
        f"{lane}: expected {service!r} -> {expected!r}, got {overlay_names.get(service)!r}"
    )


RUNTIME_LIST_ONLY_SERVICES = {"autoheal", "context-audit-consumer"}


@pytest.mark.unit
def test_agent_runtime_list_feeds_the_instantiated_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ratchet sees a runtime-scope service that only the agent's list names.

    With the lane's ``disabled_services`` as committed, the agent's list holds nothing
    the bash ``RUNTIME_SERVICES`` array lacks, so the union line cannot be exercised
    by the real config. The control empties the lane's ``disabled_services``: the two
    runtime-list-only services then reach the agent's up list, the instantiated set
    and the bare-name ratchet. Deleting ``runtime_explicit |= ...`` from
    ``_instantiated_services`` fails the control.
    """
    bash_list = _runtime_services_from_deploy_script()
    assert not RUNTIME_LIST_ONLY_SERVICES & bash_list, (
        "autoheal / context-audit-consumer joined RUNTIME_SERVICES; the control below "
        "no longer isolates the agent list"
    )

    # Committed config: both are disabled, so neither is instantiated or bare.
    assert not RUNTIME_LIST_ONLY_SERVICES & _instantiated_services(STABILITY_FILE)
    assert not RUNTIME_LIST_ONLY_SERVICES & set(_bare_survivors(STABILITY_FILE))

    # Control: the lane's config disables nothing.
    real_lane_config_for = lane_config_for

    def lane_config_without_disabled(lane: EnumRuntimeLane) -> ModelLaneConfig:
        config = real_lane_config_for(lane)
        if lane is EnumRuntimeLane.STABILITY_TEST:
            return config.model_copy(update={"disabled_services": frozenset()})
        return config

    monkeypatch.setattr(agent_executor, "lane_config_for", lane_config_without_disabled)
    assert RUNTIME_LIST_ONLY_SERVICES.issubset(_agent_runtime_services(STABILITY_FILE))
    assert RUNTIME_LIST_ONLY_SERVICES.issubset(_instantiated_services(STABILITY_FILE))
    assert RUNTIME_LIST_ONLY_SERVICES.issubset(_bare_survivors(STABILITY_FILE))


@pytest.mark.unit
def test_agent_runtime_lanes_cover_every_overlay_lane_of_the_agent() -> None:
    """Every overlay lane of the agent is in the ratchet's map or a named exclusion."""
    overlay_lanes = set(_LANE_CONFIGS) - {EnumRuntimeLane.DEV}
    covered = set(AGENT_RUNTIME_LANES.values())
    excluded = set(AGENT_RUNTIME_EXCLUSIONS)
    assert covered | excluded == overlay_lanes
    assert not covered & excluded
    assert all(AGENT_RUNTIME_EXCLUSIONS.values()), "name a reason per exclusion"
    layered = set(LAYERED_OVERLAYS.values())
    for overlay_file, lane in AGENT_RUNTIME_LANES.items():
        assert overlay_file.exists(), f"{lane} overlay {overlay_file} is missing"
        assert overlay_file in layered, (
            f"{lane} is driven by the agent but its overlay is not a layered overlay "
            f"the bare-name ratchet checks"
        )


@pytest.mark.unit
@pytest.mark.parametrize("lane", sorted(LAYERED_OVERLAYS))
def test_disabled_uninstantiated_services_are_not_renamed(lane: str) -> None:
    """Services the lane does NOT instantiate must not carry a phantom override.

    Renaming a profile-disabled, non-RUNTIME_SERVICES container would force a
    desired-state entry into the lane-census manifest for something that never
    runs, re-creating the false-drift surface OMN-13011 exists to prevent.
    """
    overlay_file = LAYERED_OVERLAYS[lane]
    instantiated = _instantiated_services(overlay_file)
    overlay_names = _container_names(_load_services(overlay_file))
    base_services = _load_services(BASE_FILE)

    phantom = [
        service
        for service in overlay_names
        if service in base_services and service not in instantiated
    ]
    assert not phantom, (
        f"{lane}: overlay names non-instantiated base services {phantom}; either "
        f"drop the container_name or make the service genuinely part of the lane."
    )
