# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Each deploy-agent instance's dev-lane composition, declared as data (OMN-19543).

WHY THIS EXISTS
---------------
OMN-19522 gave the second dev instance (``dev-202``) its own composition as a
Python literal in ``executor.DEV_INSTANCE_LANE_CONFIGS``. The operator's
2026-09-25 ruling puts a deploy slot on every lab host (omni_home ledger RULING
2026-09-25T10:22:22Z, "well for parllelism anyway"), so a literal per host is a
Python edit per host. This module reads the composition from the same table
that already names the instances, ``config/deploy_lane_routing.yaml``: an
instance other than the default carries a ``lane:`` block, and the executor
builds its ``ModelLaneConfig`` from that block. A new instance is a table row
and an overlay file, never a code change.

WHAT A ``lane:`` BLOCK SAYS
---------------------------
Everything that identified dev-202 in the literal, and nothing else:

* ``compose_overlay`` -- the instance's overlay, relative to the deploy-source
  clone, layered on the .201 dev lane's pair (infra plus dev-lane).
* ``compose_project`` -- the project it builds and runs under (both, as the
  literal did: compose names a built image ``<project>-<service>``).
* ``postgres_container`` and ``runtime_container`` -- its renamed containers.
* ``health_ports`` -- ``main`` and ``effects``, the host ports its overlay
  publishes the two runtime services on. The health targets keep the SERVICE
  names, as on the .201 dev lane (the verify recreate looks them up by label).
* ``disabled_phases`` -- the .201-only phases it never runs. Disabling
  ``gateway-deploy`` also disables the gateway project's services, which have
  no lane on a host without that phase.
* ``disabled_services`` -- the services its overlay disables by profile.
* ``receipt_lane`` and ``proves`` -- the lab-pass receipt name it emits and the
  repositories whose merges that receipt may prove. The executor does not read
  them; the receipt readers in ``scripts/ci`` do, from this same table.

THE DEFAULT INSTANCE HAS NO BLOCK. ``dev-201`` is the historical dev lane and
keeps the composition the executor has always had; the table may not redeclare
it, so there is one source for it.

PARSING IS STRICT
-----------------
An unknown key, an unknown phase, a missing field or a non-default instance
without a block refuses. The agent reads its own clone's table at start, so a
malformed table stops the agent there, as a malformed routing table already
does. ``routing.parse_routing_table`` ignores the ``lane:`` key, so an agent
reading a newer table at a command's ref still routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from deploy_agent.routing import ROUTING_TABLE_RELPATH, RoutingTableError

#: The key an instance's composition sits under in the routing table.
LANE_KEY: Final = "lane"


class ModelHealthPorts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    main: int
    effects: int


class ModelInstanceLane(BaseModel):
    """One instance's ``lane:`` block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    compose_overlay: str
    compose_project: str
    postgres_container: str
    runtime_container: str
    health_ports: ModelHealthPorts
    disabled_phases: tuple[str, ...] = ()
    disabled_services: tuple[str, ...] = ()
    receipt_lane: str
    proves: tuple[str, ...]


def parse_instance_lanes(text: str) -> dict[str, ModelInstanceLane]:
    """Every non-default instance's block, keyed by instance name.

    Refuses when a non-default instance has no block, when the default
    instance has one, or when a block does not validate.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RoutingTableError(f"routing table is not YAML: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("instances"), dict):
        raise RoutingTableError("routing table declares no instances")
    default = str(raw.get("default_instance") or "").strip()
    lanes: dict[str, ModelInstanceLane] = {}
    for name, spec in raw["instances"].items():
        block = spec.get(LANE_KEY) if isinstance(spec, dict) else None
        if str(name) == default:
            if block is not None:
                raise RoutingTableError(
                    f"default instance {name!r} declares a {LANE_KEY}: block; it "
                    "keeps the dev lane's historical composition, which has one "
                    "source, the executor"
                )
            continue
        if not isinstance(block, dict):
            raise RoutingTableError(
                f"instance {name!r} declares no {LANE_KEY}: block, so it has no "
                "dev-lane composition to deploy with"
            )
        try:
            lanes[str(name)] = ModelInstanceLane.model_validate(block)
        except ValidationError as exc:
            raise RoutingTableError(
                f"instance {name!r} {LANE_KEY}: block is invalid: {exc}"
            ) from exc
    receipt_lanes = [lane.receipt_lane for lane in lanes.values()]
    if len(set(receipt_lanes)) != len(receipt_lanes):
        raise RoutingTableError(
            f"two instances declare the same receipt_lane: {sorted(receipt_lanes)}"
        )
    return lanes


def load_instance_lanes(repo_root: str | Path) -> dict[str, ModelInstanceLane]:
    """The blocks in ``repo_root``'s committed table."""
    path = Path(repo_root) / ROUTING_TABLE_RELPATH
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RoutingTableError(f"routing table unreadable at {path}: {exc}") from exc
    return parse_instance_lanes(text)
