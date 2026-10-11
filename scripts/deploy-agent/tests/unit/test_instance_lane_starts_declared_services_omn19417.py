# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19417: every service a dev instance's census declares running, its deploy starts.

The compose-dev lab-pass receipt carries a ``lane_sync`` check (OMN-19417): the
census planner runs on the lane's host and any finding fails the receipt. The
census reads the lane manifest (``deploy/lane-census/lane-manifest.yaml``), and
the dev-202 and dev-200 entries declare ``omnibase-infra-dev-<n>-phoenix`` as a
``service`` with one replica, because those entries are parity-checked against
their compose overlays.

Nothing in the deploy agent ever started that container. A FULL deploy runs two
legs: the deps leg is ``docker compose --profile core up -d`` with no service
list, so it starts every service with no profile or with ``core``; the runtime
leg is ``--profile runtime up -d --no-deps`` over ``services_for_scope``'s
explicit list. Phoenix is profiled ``["runtime", "full"]`` in the base file and
is in no scope list, so neither leg reaches it. The .201 dev lane has a Phoenix
container only because one was brought up by hand on 2026-07-31, and its
manifest entry leaves Phoenix loose. dev-202 was built by the agent alone, so
its Phoenix never existed, and every compose-dev-202 receipt since the
``lane_sync`` check merged read FAIL on ``container_absent``
``omnibase-infra-dev-202-phoenix`` and nothing else (receipts for omnimarket dev
9e369b046 through a169bca9b, 2026-10-09T05:47Z to 09:40Z).

This test pins the property the census needs: for each dev instance, every
container its manifest entry declares as a running ``service`` is the
container of a compose service that one of the deploy's two legs starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from deploy_agent.build_budget import _load_compose_documents
from deploy_agent.compose_budget import _merge_service_maps
from deploy_agent.events import (
    DEV_LANE_GATEWAY_SERVICES,
    EnumRuntimeLane,
    Scope,
    services_for_scope,
)
from deploy_agent.executor import DEV_INSTANCE_LANE_CONFIGS, ModelLaneConfig

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOCKER = _REPO_ROOT / "docker"
_MANIFEST = _REPO_ROOT / "deploy" / "lane-census" / "lane-manifest.yaml"

#: The instances whose manifest entry is parity-checked against its overlay and
#: therefore declares every service the composition runs.
_OVERLAY_INSTANCES = ("dev-202", "dev-200")


def _merged_services(config: ModelLaneConfig) -> dict[str, dict[str, Any]]:
    """The lane's composition, read from this checkout rather than the agent clone."""
    paths = [str(_DOCKER / Path(f).name) for f in config.compose_files]
    return _merge_service_maps(_load_compose_documents(paths))


def _deps_leg(
    services: dict[str, dict[str, Any]], disabled: frozenset[str]
) -> set[str]:
    """``--profile core up -d`` with no service list: unprofiled or ``core``."""
    return {
        name
        for name, spec in services.items()
        if name not in disabled
        and (not spec.get("profiles") or "core" in spec["profiles"])
    }


def _runtime_leg(disabled: frozenset[str]) -> set[str]:
    """``--profile runtime up -d --no-deps`` over the DEV lane's explicit list."""
    listed = set(services_for_scope(Scope.RUNTIME, lane=EnumRuntimeLane.DEV))
    return listed - set(DEV_LANE_GATEWAY_SERVICES) - disabled


def _declared_running(compose_project: str) -> set[str]:
    manifest = yaml.safe_load(_MANIFEST.read_text())
    lanes = [
        spec
        for spec in manifest["lanes"].values()
        if spec.get("compose_project") == compose_project
    ]
    assert len(lanes) == 1, f"one manifest lane for {compose_project}, got {len(lanes)}"
    return {
        s["name"]
        for s in lanes[0]["services"]
        if s.get("kind") == "service" and int(s.get("replicas", 1)) >= 1
    }


@pytest.mark.parametrize("instance", _OVERLAY_INSTANCES)
def test_instance_lane_deploy_starts_every_declared_running_service(
    instance: str,
) -> None:
    config = DEV_INSTANCE_LANE_CONFIGS[instance]
    services = _merged_services(config)
    by_container = {
        spec.get("container_name", name): name for name, spec in services.items()
    }
    started = _deps_leg(services, config.disabled_services) | _runtime_leg(
        config.disabled_services
    )

    declared = _declared_running(config.compose_project)
    # Positive controls: the read found the lane, and both legs are non-empty and
    # carry the services they always have.
    assert declared, f"{instance}: the manifest declares no running service"
    assert "postgres" in _deps_leg(services, config.disabled_services)
    assert "omninode-runtime" in _runtime_leg(config.disabled_services)

    never_started = sorted(
        container
        for container in declared
        if by_container.get(container) not in started
    )
    assert never_started == [], (
        f"{instance}: the census declares these containers running, and no leg "
        f"of the agent's deploy starts their compose service: {never_started}"
    )


def test_instance_lane_phoenix_stays_out_of_the_201_deps_leg() -> None:
    """Control: the .201 composition is unchanged; its Phoenix stays runtime-only."""
    config = DEV_INSTANCE_LANE_CONFIGS["dev-201"]
    services = _merged_services(config)
    assert "phoenix" in services
    assert "phoenix" not in _deps_leg(services, config.disabled_services)
