# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which deploy-agent instance lanes may prove a repository's merge (OMN-19543).

The operator put one deploy slot on every lab host (omni_home ledger RULING
2026-09-25T10:22:22Z, amended by the CORRECTION 2026-09-25T10:22:36Z). Each
slot is an instance in ``config/deploy_lane_routing.yaml`` whose ``lane:`` block
names the lab-pass receipt lane it emits (``receipt_lane``) and the repositories
that receipt may prove (``proves``). This module is the one reader of those two
fields for the receipt consumers:

* the release train's ``compose-dev-or-instance-lanes`` premise
  (``scripts/ci/release_train.py``), which reads ``compose-dev`` first and then
  these lanes, in table order;
* the staging delivery's sibling read (``deliver-dev-candidate-to-staging.yml``),
  which admits a pinned sibling revision on ``compose-dev`` or any of these.

Before this, both hardcoded dev-202: a literal ``compose-dev-202`` in the
workflow and a ``compose-dev-or-compose-dev-202`` evidence value. A new instance
now reaches both by its table row. The receipt lane name itself is still a
member of the closed ``EnumLabLane`` (``scripts/ci/lab_pass_receipt.py``),
because a receipt is a typed artifact; the release train refuses a table that
names a lane the enum does not know.

The delivery workflow reaches it through ``lab_pass_receipt.py gate
--instance-lanes-for <repo>``, the gate it already runs, so there is one reader
of receipts and this module is its input, not a second gate. An unreadable or
malformed table raises, and the gate refuses rather than reading fewer lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import yaml

ROUTING_TABLE: Final = (
    Path(__file__).resolve().parents[2] / "config" / "deploy_lane_routing.yaml"
)


class InstanceReceiptLanesError(Exception):
    """The routing table is missing, unreadable or malformed."""


def _instances(text: str) -> dict[str, dict[str, object]]:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InstanceReceiptLanesError(f"routing table is not YAML: {exc}") from exc
    instances = raw.get("instances") if isinstance(raw, dict) else None
    if not isinstance(instances, dict):
        raise InstanceReceiptLanesError("routing table declares no instances")
    return {str(name): spec for name, spec in instances.items()}


def receipt_lanes_from_text(text: str, repo: str) -> tuple[str, ...]:
    """The receipt lane of every instance whose ``proves`` names ``repo``."""
    lanes: list[str] = []
    for name, spec in _instances(text).items():
        block = spec.get("lane") if isinstance(spec, dict) else None
        if block is None:
            continue
        if not isinstance(block, dict):
            raise InstanceReceiptLanesError(f"instance {name!r} lane: is not a mapping")
        receipt_lane = str(block.get("receipt_lane") or "").strip()
        proves = block.get("proves")
        if not receipt_lane or not isinstance(proves, list):
            raise InstanceReceiptLanesError(
                f"instance {name!r} lane: must declare receipt_lane and a proves list"
            )
        if repo in proves:
            lanes.append(receipt_lane)
    return tuple(lanes)


def receipt_lanes_for(repo: str, table: Path | None = None) -> tuple[str, ...]:
    """``receipt_lanes_from_text`` over the committed table (or ``table``)."""
    table = table if table is not None else ROUTING_TABLE
    try:
        text = table.read_text(encoding="utf-8")
    except OSError as exc:
        raise InstanceReceiptLanesError(
            f"routing table unreadable at {table}: {exc}"
        ) from exc
    return receipt_lanes_from_text(text, repo)


def _spec(table: dict[str, object], name: str) -> dict[str, object]:
    instances = table.get("instances")
    spec = instances.get(name) if isinstance(instances, dict) else None
    if not isinstance(spec, dict):
        raise InstanceReceiptLanesError(f"instance {name!r} is not declared")
    return spec


def _instance_receipt_lane(table: dict[str, object], name: str) -> str:
    """The receipt lane ``name`` emits: its ``verify:`` lane, else its ``lane:`` one."""
    spec = _spec(table, name)
    for key in ("verify", "lane"):
        block = spec.get(key)
        lane = block.get("receipt_lane") if isinstance(block, dict) else None
        if lane:
            return str(lane)
    raise InstanceReceiptLanesError(f"instance {name!r} declares no receipt_lane")


def _load(text: str) -> dict[str, object]:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InstanceReceiptLanesError(f"routing table is not YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise InstanceReceiptLanesError("routing table is not a mapping")
    return raw


def _read(table: Path | None) -> str:
    table = table if table is not None else ROUTING_TABLE
    try:
        return table.read_text(encoding="utf-8")
    except OSError as exc:
        raise InstanceReceiptLanesError(
            f"routing table unreadable at {table}: {exc}"
        ) from exc


@dataclass(frozen=True)
class FrozenLane:
    """A receipt lane whose instance is frozen, and the lane that stands in."""

    instance: str
    until: datetime
    substitute_instance: str
    substitute_receipt_lane: str


def _freeze(spec: dict[str, object]) -> dict[str, object] | None:
    flags = spec.get("flags")
    freeze = flags.get("freeze") if isinstance(flags, dict) else None
    return freeze if isinstance(freeze, dict) else None


def _until(freeze: dict[str, object]) -> datetime:
    until = freeze.get("until")
    if isinstance(until, str):
        until = datetime.fromisoformat(until.replace("Z", "+00:00"))
    if not isinstance(until, datetime):
        raise InstanceReceiptLanesError("a freeze declares no `until` timestamp")
    return until if until.tzinfo is not None else until.replace(tzinfo=UTC)


def frozen_receipt_lanes_from_text(text: str, now: datetime) -> dict[str, FrozenLane]:
    """Receipt lane -> its frozen instance's declared substitute, while ``now < until``.

    Only a freeze that names a ``substitute_instance`` appears: a freeze with no
    substitute declares no lane that stands in, so it is not an explained absence.
    """
    table = _load(text)
    instances = table.get("instances")
    if not isinstance(instances, dict):
        raise InstanceReceiptLanesError("routing table declares no instances")
    frozen: dict[str, FrozenLane] = {}
    for name, spec in instances.items():
        freeze = _freeze(spec) if isinstance(spec, dict) else None
        substitute = str(freeze.get("substitute_instance") or "") if freeze else ""
        if freeze is None or not substitute:
            continue
        until = _until(freeze)
        if now >= until:
            continue
        frozen[_instance_receipt_lane(table, str(name))] = FrozenLane(
            instance=str(name),
            until=until,
            substitute_instance=substitute,
            substitute_receipt_lane=_instance_receipt_lane(table, substitute),
        )
    return frozen


def frozen_receipt_lanes(
    now: datetime, table: Path | None = None
) -> dict[str, FrozenLane]:
    """``frozen_receipt_lanes_from_text`` over the committed table (or ``table``)."""
    return frozen_receipt_lanes_from_text(_read(table), now)


def check_substitute_routes_from_text(text: str) -> None:
    """Refuse a table whose freeze substitute and ``while_frozen`` routes disagree.

    Both halves of the declaration must exist together: a freeze naming a
    substitute needs at least one route standing in for it, and a route
    marked ``while_frozen`` needs that instance's freeze to name the route's
    instance and the instance to prove the repository. Dropping the freeze
    without the route would leave a lane routing to a substitute for nothing;
    dropping the route alone would leave the frozen lane unproven.
    """
    table = _load(text)
    instances = table.get("instances")
    if not isinstance(instances, dict):
        raise InstanceReceiptLanesError("routing table declares no instances")
    routes = [r for r in table.get("routes") or () if isinstance(r, dict)]
    for name, spec in instances.items():
        freeze = _freeze(spec) if isinstance(spec, dict) else None
        substitute = str(freeze.get("substitute_instance") or "") if freeze else ""
        if not substitute:
            continue
        _spec(table, substitute)
        if not any(
            r.get("while_frozen") == name and r.get("instance") == substitute
            for r in routes
        ):
            raise InstanceReceiptLanesError(
                f"{name} freezes with substitute_instance {substitute}, but no "
                f"route is while_frozen: {name} to {substitute}, so nothing "
                "deploys there in its place"
            )
    for route in routes:
        frozen = str(route.get("while_frozen") or "")
        if not frozen:
            continue
        repo = str(route.get("requester_repository"))
        target = str(route.get("instance"))
        freeze = _freeze(_spec(table, frozen))
        if freeze is None or str(freeze.get("substitute_instance") or "") != target:
            raise InstanceReceiptLanesError(
                f"the route for {repo} to {target} is while_frozen: {frozen}, but "
                f"{frozen} declares no freeze naming {target} as its "
                "substitute_instance; remove the route with the freeze"
            )
        if repo not in (_proves(_spec(table, target)) or ()):
            raise InstanceReceiptLanesError(
                f"the route for {repo} to {target} is while_frozen: {frozen}, but "
                f"{target} lane.proves does not name {repo}"
            )


def _proves(spec: dict[str, object]) -> list[object] | None:
    block = spec.get("lane")
    proves = block.get("proves") if isinstance(block, dict) else None
    return proves if isinstance(proves, list) else None


def routed_receipt_lane(
    repo: str, table: Path | None = None, runtime_lane: str = "dev"
) -> str:
    """The receipt lane of the instance a ``runtime_lane`` rebuild from ``repo`` runs on.

    The same rule as ``deploy_agent.routing.ModelRoutingTable.route`` (and
    ``deploy_lane_verify_route.route``): the first matching row, else
    ``default_instance``. The delivery gate requires this lane for the delivered
    sha, since it is the one lane that actually ran it.
    """
    parsed = _load(_read(table))
    default = str(parsed.get("default_instance") or "").strip()
    if not default:
        raise InstanceReceiptLanesError("routing table declares no default_instance")
    instance = default
    for route in parsed.get("routes") or ():
        if (
            isinstance(route, dict)
            and str(route.get("runtime_lane")) == runtime_lane
            and str(route.get("requester_repository")) == repo
        ):
            instance = str(route.get("instance"))
            break
    return _instance_receipt_lane(parsed, instance)
