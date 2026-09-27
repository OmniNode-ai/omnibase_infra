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
