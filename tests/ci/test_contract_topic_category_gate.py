# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression for the mixed-category routing gate (OMN-14605).

Proves three things against the REAL detector + REAL production helpers:

1. RED: a handler_routing entry whose assigned topics span >1 real category is
   flagged (the node_swarm_fanout_orchestrator / OMN-14606 shape).
2. GREEN: a single-category entry is NOT flagged.
3. AGREEMENT (OMN-12549 seam, OMN-14208 discipline): the EXACT synthetic shape
   the gate flags is ALSO hard-rejected by ``MixinNodeDispatch`` at route/
   dispatcher registration time, and the shape the gate passes is accepted.
   This is the tested assertion that keeps the CI gate and the core runtime
   invariant from silently diverging — if either side stops agreeing, this test
   goes red.

Plus a live-repo assertion that the seeded baseline is green day-1 (the
WARN-on-baseline / hard-fail-on-growth ratchet).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.runtime.mixin_node_dispatch import MixinNodeDispatch
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.runtime.auto_wiring import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.validators.contract_topic_category import (
    REASON_MIXED,
    category_findings,
    main,
)

pytestmark = pytest.mark.unit

_CMD_TOPIC = "onex.cmd.omnimarket.swarm-fanout.v1"
_EVT_TOPIC = "onex.evt.omnimarket.delegation-call-completed.v1"
_EVT_TOPIC_2 = "onex.evt.omnimarket.delegation-all-tiers-failed.v1"
_HANDLER = ModelHandlerRef(name="HandlerSwarmFanout", module="omnimarket.fake")


def _contract(
    *, name: str, subscribe_topics: tuple[str, ...]
) -> ModelDiscoveredContract:
    """A sole-handler operation_match contract with NO per-entry event_model, so
    ``_topics_for_handler_entry`` assigns the entry EVERY subscribe topic (the exact
    shape that produced the OMN-14606 mixed-category NO_DISPATCHER)."""
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/tmp/omn-14605/contract.yaml"),  # noqa: S108
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe_topics, publish_topics=()
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(ModelHandlerRoutingEntry(operation="fanout", handler=_HANDLER),),
        ),
    )


def test_gate_flags_mixed_category_entry() -> None:
    """RED: an entry assigned command + event topics is flagged."""
    findings = category_findings(
        [_contract(name="node_x", subscribe_topics=(_CMD_TOPIC, _EVT_TOPIC))]
    )
    assert len(findings) == 1, findings
    f = findings[0]
    assert f.contract == "node_x"
    assert f.reason == REASON_MIXED
    assert "command+event" in f.detail


def test_gate_passes_single_category_entry() -> None:
    """GREEN: an entry assigned only event topics is NOT flagged."""
    findings = category_findings(
        [_contract(name="node_ok", subscribe_topics=(_EVT_TOPIC, _EVT_TOPIC_2))]
    )
    assert findings == []


def _register_entry_shape_in_mixin(topics: tuple[str, ...]) -> None:
    """Register ONE dispatcher (single derived category) + one route per topic
    into a real ``MixinNodeDispatch`` — the exact registration shape
    ``_prepare_handler_wiring`` produces for a single handler_routing entry.

    Category is derived from ``topics[0]`` (mirrors handler_wiring's single-
    category derivation). Raises ``ModelOnexError`` iff the routes disagree in
    category with the one dispatcher category — i.e. iff the entry is mixed.
    """
    mixin = MixinNodeDispatch()

    async def _cb(_envelope: object) -> None:
        return None

    entry_category = EnumMessageCategory.from_topic(topics[0])
    assert entry_category is not None
    mixin.register_dispatcher("d1", _cb, category=entry_category)
    for i, topic in enumerate(topics):
        route_category = EnumMessageCategory.from_topic(topic)
        assert route_category is not None
        mixin.register_route(
            ModelDispatchRoute(
                route_id=f"r{i}",
                topic_pattern=topic,
                message_category=route_category,
                handler_id="d1",
            )
        )


def test_mixin_node_dispatch_rejects_exactly_the_flagged_shape() -> None:
    """AGREEMENT: the mixed shape the gate flags is hard-rejected by the core
    runtime invariant (MixinNodeDispatch), and the clean shape is accepted.

    This binds the CI gate to OMN-12549: both surfaces must reject the same
    thing. If they diverge, one of these two assertions fails."""
    # Clean (single-category) — gate passes AND MixinNodeDispatch accepts.
    assert (
        category_findings(
            [_contract(name="clean", subscribe_topics=(_EVT_TOPIC, _EVT_TOPIC_2))]
        )
        == []
    )
    _register_entry_shape_in_mixin((_EVT_TOPIC, _EVT_TOPIC_2))  # no raise

    # Mixed — gate flags AND MixinNodeDispatch rejects the identical shape.
    assert (
        len(
            category_findings(
                [_contract(name="mixed", subscribe_topics=(_CMD_TOPIC, _EVT_TOPIC))]
            )
        )
        == 1
    )
    with pytest.raises(ModelOnexError, match="category"):
        _register_entry_shape_in_mixin((_CMD_TOPIC, _EVT_TOPIC))


def test_live_repo_scan_is_green_at_zero_with_no_baseline() -> None:
    """The live repo scan exits 0 with ZERO findings and no baseline anywhere.

    OMN-18013 replaced the OMN-14605 WARN-on-baseline ratchet: the baseline was
    burned to zero and the file DELETED, so a non-zero exit here means a real
    new offender and there is nothing to freeze it into. The gate also refuses
    to run without an explicit --scope, because defaulting the scope is how a
    gate silently scans nothing.
    """
    assert main(["--scope", "omnibase_infra"]) == 0
    assert main([]) == 1, "the gate must refuse to run with no scope"
    repo_root = Path(__file__).resolve().parents[2]
    assert not (
        repo_root / "config" / "validation" / "mixed_category_routing_baseline.yaml"
    ).exists()
