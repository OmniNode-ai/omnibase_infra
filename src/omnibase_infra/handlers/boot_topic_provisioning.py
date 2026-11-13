# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot topic provisioning for contracts, wired or not (OMN-23951).

The per-contract boot interleave (OMN-13237) provisions a contract's topics
only when the contract is WIRED and passes the eligibility filters of
``subscribe_wired_contract_topics``. A node that publishes but declares no
``subscribe_topics`` and no ``handler_routing`` is SKIPPED by auto-wiring, and
so are the contracts those filters drop (plugin-managed, no dispatchers,
demand-deferred), so the declared ``publish_topics`` of every one of them were
created by no one when the universe warm is off
(``ONEX_BOOT_UNIVERSE_PROVISION=0``, the standing dev setting).

``provision_unattached_publish_topics`` creates those topics through the same
``ProtocolTopicProvisioner`` the interleave uses. ``ensure_topics_exist`` is
the one per-topic create-and-tolerate step both paths share.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.protocols.protocol_topic_provisioner import (
    ProtocolTopicProvisioner,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.auto_wiring.models import (
        ModelAutoWiringManifest,
        ModelDiscoveredContract,
    )

logger = logging.getLogger(__name__)


def contract_publish_topics(contract: ModelDiscoveredContract) -> tuple[str, ...]:
    """Return the de-duplicated ``event_bus.publish_topics`` the contract declares.

    Names come from the contract's own declaration only, never a Python literal.
    """
    if contract.event_bus is None:
        return ()
    publish_topics = contract.event_bus.publish_topics
    return tuple(dict.fromkeys(t for t in publish_topics if t and t.strip()))


async def ensure_topics_exist(
    provisioner: ProtocolTopicProvisioner,
    topics: Sequence[str],
    *,
    contract_name: str,
) -> tuple[str, ...]:
    """Idempotently create ``topics``; return those the provisioner confirmed.

    ``TopicReplicationPolicyError`` re-raises (fail closed, OMN-15395): a
    durability-policy violation must never be silently skipped. Any other
    failure, and a topic the provisioner could not confirm (``False``), is
    logged with the contract and topic and left out of the result; none is fatal
    to the boot.
    """
    outcomes = await asyncio.gather(
        *(provisioner.ensure_topic_exists(topic_name=t) for t in topics),
        return_exceptions=True,
    )
    ensured: list[str] = []
    for topic, outcome in zip(topics, outcomes, strict=True):
        if isinstance(outcome, TopicReplicationPolicyError) or not isinstance(
            outcome, (bool, Exception)
        ):
            raise outcome
        if outcome is True:
            ensured.append(topic)
            continue
        logger.warning(
            "Topic provisioning did not confirm contract '%s' topic '%s' (non-fatal)",
            contract_name,
            topic,
            exc_info=outcome if isinstance(outcome, Exception) else None,
        )
    return tuple(ensured)


async def provision_unattached_publish_topics(
    manifest: ModelAutoWiringManifest,
    interleaved_contract_names: Collection[str],
    provisioner: ProtocolTopicProvisioner | None,
    *,
    max_concurrent: int,
) -> dict[str, tuple[str, ...]]:
    """Create the publish topics of every contract the interleave never reached.

    Each contract in *manifest* whose name is not in *interleaved_contract_names*
    and that declares publish topics has them ensured, with at most
    *max_concurrent* contracts in flight. Returns ``{contract: topics ensured}``;
    with no *provisioner* nothing is created and ``{}`` is returned.

    *interleaved_contract_names* is every contract the provision -> ready ->
    attach interleave owns, attached at boot or deferred to its first request
    (OMN-18087, OMN-23953): a deferred contract's whole topic set, publish topics
    included, is created when its first request starts it, so creating the topics
    here would create them for a contract that may never load. Only a contract
    nothing will ever interleave (publish-only, plugin-managed, no live
    dispatcher) is provisioned by this pass.
    """
    if provisioner is None:
        return {}
    unattached = [
        contract
        for contract in manifest.contracts
        if contract.name not in interleaved_contract_names
        and contract_publish_topics(contract)
    ]
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _provision(contract: ModelDiscoveredContract) -> tuple[str, ...]:
        async with semaphore:
            return await ensure_topics_exist(
                provisioner,
                contract_publish_topics(contract),
                contract_name=contract.name,
            )

    results = await asyncio.gather(*(_provision(c) for c in unattached))
    ensured = {
        contract.name: topics
        for contract, topics in zip(unattached, results, strict=True)
        if topics
    }
    logger.info(
        "Provisioned publish topics for %d unattached contract(s): %d topic(s) "
        "(OMN-23951)",
        len(ensured),
        sum(len(topics) for topics in ensured.values()),
    )
    return ensured


__all__: list[str] = [
    "contract_publish_topics",
    "ensure_topics_exist",
    "provision_unattached_publish_topics",
]
