# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Boot-time single-owner-per-topic invariants for the S6 core runtime (§c.3).

S4's ``RuntimeDispatch.__init__`` already asserts the I4 injective-at-boot guard over
every route's ``published_events``. It does NOT assert single-owner ACROSS the new
routing map and the legacy kernel. These three assertions are S6's job — any violation
RAISES at boot (fail closed), never a silent double-subscribe / mis-route.

1. **RuntimeDispatch ⟂ legacy** — no allowlisted topic is ALSO subscribed by the legacy
   push path (the legacy subscribe path skips allowlisted topics; this asserts the
   realized legacy-subscribed set is disjoint from ``core_runtime_topics``).
2. **RuntimeDispatch internal single-owner** — every allowlist topic resolves to exactly
   one route (the map build already raises on a duplicate key; this re-asserts the set
   equality so a partial build cannot slip through).
3. **Fan-out safety** — no allowlist topic is a subscribe topic of a DIFFERENT discovered
   contract that is NOT being moved. This keeps legitimately multi-consumer ``evt`` topics
   (``inference-response.v1``, ``delegation-completed.v1``, …) OUT of the single-route
   runtime (§a.4 made mechanical).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.runtime.runtime_dispatch import DispatchRoute
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)

__all__ = ["assert_single_owner_split"]


def assert_single_owner_split(
    *,
    core_runtime_topics: frozenset[str],
    routing_map: Mapping[str, DispatchRoute],
    legacy_subscribed_topics: frozenset[str],
    contracts: Sequence[ModelDiscoveredContract],
) -> None:
    """Assert the three single-owner invariants for the allowlist split (§c.3)."""
    # (1) RuntimeDispatch ⟂ legacy.
    overlap = core_runtime_topics & legacy_subscribed_topics
    if overlap:
        raise ModelOnexError(
            message=(
                f"S6 single-owner: allowlisted topics {sorted(overlap)} are ALSO "
                "subscribed by the legacy kernel push path. The legacy subscribe path "
                "must skip every core-runtime topic (double-consume / double-commit "
                "hazard). Fail closed."
            ),
            error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
        )

    # (2) RuntimeDispatch internal single-owner (exact coverage, one route per topic).
    mapped = set(routing_map)
    if mapped != set(core_runtime_topics):
        raise ModelOnexError(
            message=(
                "S6 single-owner: routing_map topics "
                f"{sorted(mapped)} do not exactly match the allowlist "
                f"{sorted(core_runtime_topics)}. Every allowlist topic must resolve to "
                "exactly one route and no route may exist outside the allowlist."
            ),
            error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
        )

    # (3) Fan-out safety — an allowlist topic subscribed by >1 discovered contract is a
    #     legitimate multi-consumer fan-out that cannot move to the single-route runtime.
    owners_by_topic: dict[str, list[str]] = {}
    for contract in contracts:
        if contract.event_bus is None:
            continue
        for topic in contract.event_bus.subscribe_topics:
            if topic in core_runtime_topics:
                owners_by_topic.setdefault(topic, []).append(contract.name)
    multi = {t: names for t, names in owners_by_topic.items() if len(names) > 1}
    if multi:
        raise ModelOnexError(
            message=(
                "S6 single-owner: allowlisted topics have multiple subscribing "
                f"contracts {multi}. A multi-consumer fan-out topic cannot move to the "
                "single-route RuntimeDispatch — keep it on the legacy kernel (its "
                "fan-out is independent Kafka consumer groups). Fail closed."
            ),
            error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
        )
