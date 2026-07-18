# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Boot-time single-owner-per-topic invariants for the core runtime (§c.3, S8 §D1=4b).

S4's ``RuntimeDispatch.__init__`` already asserts the I4 injective-at-boot guard over
every route's ``published_events``. It does NOT assert single-owner ACROSS the new
routing map and the legacy kernel. These assertions are the composition root's job — any
violation RAISES at boot (fail closed), never a silent double-subscribe / mis-route.

1. **RuntimeDispatch ⟂ legacy** — no *single-owner* allowlist topic is ALSO subscribed by
   the legacy push path (the legacy subscribe path skips the owner's subscription for an
   allowlisted topic; this asserts the realized legacy-subscribed set is disjoint from
   the single-owner core-runtime topics). A genuine fan-out topic (>1 subscriber) is
   EXEMPT here — its non-owner consumers legitimately remain legacy (see §4b below).
2. **RuntimeDispatch internal single-owner** — every allowlist topic resolves to exactly
   one route (the map build already raises on a duplicate key; this re-asserts the set
   equality so a partial build cannot slip through).
3. **Fan-out safety (S8 §D1=4b)** — an allowlist topic subscribed by MORE than one
   discovered contract is a genuine multi-consumer fan-out. It is permitted onto the
   single-route runtime ONLY when exactly ONE subscriber is the designated core-runtime
   OWNER and every OTHER subscriber stays on the legacy kernel with a DISTINCT consumer
   group (no group collision, and none colliding with the ONE core-runtime group). A
   fan-out topic with no designated owner fails closed — allowlisting it would silently
   no-op its non-designated consumers. This replaces the pre-S8 behavior that refused
   every multi-consumer allowlist topic outright (option 4a). Kafka fan-out across
   distinct consumer groups is valid; the "single-route" constraint is only *within* the
   one RuntimeDispatch map.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.runtime.runtime_dispatch import DispatchRoute
from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.utils import compute_consumer_group_id

__all__ = ["CORE_RUNTIME_GROUP", "assert_single_owner_split"]

# The ONE core-runtime consumer group. Single source of truth (kernel_glue imports it for
# the live KafkaTransport group). Distinct from every per-node legacy group so offsets are
# independent (rollback safety, R-4) and so the §4b fan-out check can assert no legacy
# consumer collides with it.
CORE_RUNTIME_GROUP = "onex.core-runtime.delegation"

# A resolver maps a discovered contract to the Kafka consumer group id the legacy push
# path would derive for it. Injected so the §4b group-distinctness check is faithful to
# the real derivation (``_subscribe_contract_topics`` uses the same identity model), and
# so unit tests can supply a trivial resolver.
ConsumerGroupResolver = Callable[[ModelDiscoveredContract], str]


def _default_consumer_group_resolver(contract: ModelDiscoveredContract) -> str:
    """Derive a contract's legacy consumer group id (mirrors the legacy subscribe path).

    Uses the SAME ``compute_consumer_group_id`` derivation ``_subscribe_contract_topics``
    applies. The ``env`` component is held fixed here on purpose: the §4b check only needs
    group DISTINCTNESS between the topic's legacy consumers (and non-collision with the
    literal core-runtime group), both of which are ``env``-invariant — every contract in
    one runtime shares one ``env``, so it cancels out of a distinctness comparison.
    """
    identity = ModelNodeIdentity(
        env="runtime",
        service=contract.package_name,
        node_name=contract.name,
        version=str(contract.contract_version),
    )
    return compute_consumer_group_id(identity, EnumConsumerGroupPurpose.CONSUME)


def assert_single_owner_split(
    *,
    core_runtime_topics: frozenset[str],
    routing_map: Mapping[str, DispatchRoute],
    legacy_subscribed_topics: frozenset[str],
    contracts: Sequence[ModelDiscoveredContract],
    owners: Mapping[str, str] | None = None,
    consumer_group_resolver: ConsumerGroupResolver | None = None,
    core_runtime_group: str = CORE_RUNTIME_GROUP,
) -> None:
    """Assert the three single-owner invariants for the allowlist split (§c.3, §D1=4b).

    ``owners`` maps an allowlist topic to its resolved core-runtime owner contract name
    (see ``composition.resolve_core_runtime_owners``). It is required for any genuine
    fan-out (>1-subscriber) allowlist topic; single-subscriber topics need no entry.
    """
    owner_by_topic = dict(owners or {})
    resolver = consumer_group_resolver or _default_consumer_group_resolver

    # Subscriber contracts per allowlist topic (one pass; drives invariants 1 and 3).
    subscribers_by_topic: dict[str, list[ModelDiscoveredContract]] = {}
    for contract in contracts:
        if contract.event_bus is None:
            continue
        if contract.event_bus.plugin_managed and (
            core_runtime_topics & set(contract.event_bus.subscribe_topics)
        ):
            raise ModelOnexError(
                message=(
                    "S8 single-owner (4b): plugin-managed contract "
                    f"{contract.name!r} subscribes to allowlist topic(s) "
                    f"{sorted(core_runtime_topics & set(contract.event_bus.subscribe_topics))}. "
                    "Plugin-managed subscription paths do not yet expose ownership-aware "
                    "consumer group validation; fail closed."
                ),
                error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
            )
        for topic in contract.event_bus.subscribe_topics:
            if topic in core_runtime_topics:
                subscribers_by_topic.setdefault(topic, []).append(contract)

    def _is_fanout(topic: str) -> bool:
        return len(subscribers_by_topic.get(topic, [])) > 1

    # (1) RuntimeDispatch ⟂ legacy — SINGLE-OWNER topics only. A fan-out topic legitimately
    #     stays legacy-subscribed by its non-owner consumers (§4b), so exempt it here; its
    #     owner's disjointness is enforced by the legacy skip + invariant (3).
    bad_overlap = {
        topic
        for topic in (core_runtime_topics & legacy_subscribed_topics)
        if not _is_fanout(topic)
    }
    if bad_overlap:
        raise ModelOnexError(
            message=(
                f"S6 single-owner: single-owner allowlist topics {sorted(bad_overlap)} "
                "are ALSO subscribed by the legacy kernel push path. The legacy subscribe "
                "path must skip the owner's subscription for every single-owner "
                "core-runtime topic (double-consume / double-commit hazard). Fail closed."
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

    # (3) Fan-out safety / §D1=4b enablement — a multi-consumer allowlist topic may move
    #     ONLY when exactly one subscriber is the designated owner and every non-owner
    #     subscriber keeps a distinct legacy consumer group (no collision, none equal to
    #     the core-runtime group).
    for topic, subscribers in subscribers_by_topic.items():
        if len(subscribers) <= 1:
            continue
        subscriber_names = sorted(c.name for c in subscribers)
        owner = owner_by_topic.get(topic)
        if owner is None:
            raise ModelOnexError(
                message=(
                    f"S8 single-owner (4b): allowlist topic {topic!r} is subscribed by "
                    f"multiple contracts {subscriber_names} (fan-out) but no core-runtime "
                    "owner is designated. Exactly one subscriber may move to the single "
                    "runtime; the others must stay legacy on distinct consumer groups. "
                    "Fail closed."
                ),
                error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
            )
        if owner not in {c.name for c in subscribers}:
            raise ModelOnexError(
                message=(
                    f"S8 single-owner (4b): designated owner {owner!r} for fan-out topic "
                    f"{topic!r} is not among its subscribers {subscriber_names}."
                ),
                error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
            )
        # Non-owner subscribers stay legacy — assert their consumer groups are distinct
        # from each other and from the ONE core-runtime group (no partition-stealing).
        seen_groups: dict[str, str] = {}
        for contract in subscribers:
            if contract.name == owner:
                continue
            group = resolver(contract)
            if group == core_runtime_group:
                raise ModelOnexError(
                    message=(
                        f"S8 single-owner (4b): legacy fan-out consumer {contract.name!r} "
                        f"on topic {topic!r} resolves to the core-runtime consumer group "
                        f"{core_runtime_group!r}. A legacy consumer must keep a DISTINCT "
                        "group so the moved owner and the legacy consumer do not steal "
                        "each other's partitions. Fail closed."
                    ),
                    error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
                )
            if group in seen_groups:
                raise ModelOnexError(
                    message=(
                        f"S8 single-owner (4b): legacy fan-out consumers "
                        f"{seen_groups[group]!r} and {contract.name!r} on topic {topic!r} "
                        f"share consumer group {group!r} (group collision — they would "
                        "compete for partitions instead of each receiving the full event "
                        "stream). Fail closed."
                    ),
                    error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
                )
            seen_groups[group] = contract.name
