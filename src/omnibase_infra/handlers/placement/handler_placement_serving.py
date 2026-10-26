# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-node serving counts from runtime membership, and their health verdict (OMN-21421).

The h201 outage was invisible because no read reported a node served by one
runtime. A node loaded by fewer than ``min_serving`` live runtimes is a FAIL.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from omnibase_infra.handlers.placement.enum_placement_health_verdict import (
    EnumPlacementHealthVerdict,
)
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
)
from omnibase_infra.handlers.placement.handler_placement_session import (
    MEMBERS_PREFIX,
)
from omnibase_infra.handlers.placement.model_placement_serving_count import (
    ModelPlacementServingCount,
)
from omnibase_infra.handlers.placement.model_placement_serving_report import (
    ModelPlacementServingReport,
)


def evaluate_serving_health(
    counts: Sequence[ModelPlacementServingCount], *, min_serving: int = 2
) -> ModelPlacementServingReport:
    """FAIL when any node is served by fewer than ``min_serving`` runtimes."""
    under = tuple(sorted(c.node_key for c in counts if c.serving < min_serving))
    return ModelPlacementServingReport(
        counts=tuple(sorted(counts, key=lambda c: c.node_key)),
        min_serving=min_serving,
        verdict=EnumPlacementHealthVerdict.FAIL
        if under
        else EnumPlacementHealthVerdict.OK,
        under_served=under,
    )


async def read_serving_report(
    adapter: HandlerPlacementEtcd,
    *,
    expected_nodes: Sequence[str],
    min_serving: int = 2,
) -> ModelPlacementServingReport:
    """Count, per expected node, the live runtimes whose membership lists it."""
    members = await adapter.get_prefix(MEMBERS_PREFIX)
    serving: dict[str, list[str]] = {name: [] for name in expected_nodes}
    for raw in members.values():
        record = json.loads(raw)
        runtime_id = str(record.get("runtime_id", ""))
        for node_name in record.get("nodes", []):
            serving.setdefault(str(node_name), []).append(runtime_id)
    counts = [
        ModelPlacementServingCount(
            node_key=name, serving=len(ids), runtime_ids=tuple(sorted(ids))
        )
        for name, ids in serving.items()
    ]
    return evaluate_serving_health(counts, min_serving=min_serving)


__all__ = ["evaluate_serving_health", "read_serving_report"]
