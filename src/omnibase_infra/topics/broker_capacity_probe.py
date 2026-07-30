# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Measure a live broker's node count and bind it to the provisioning policy.

Why this exists (OMN-15395)
---------------------------
:class:`~omnibase_infra.topics.model_topic_provisioning_policy.ModelTopicProvisioningPolicy`
may reduce a contract-declared replication factor down to what the target
broker can physically host. The first revision of that policy derived the
ceiling from the SASL mechanism — anything that was not ``AWS_MSK_IAM`` was
assumed to be a single node and had every declared RF clamped to 1. That
assumption is false for any multi-broker cluster reached over PLAIN or SCRAM,
and it silently recreated the exact ``AWS_KAFKA_HIGH_RISK_CONFIG_RF_EQUALS_ONE``
condition the ticket exists to eliminate.

This module replaces the assumption with a measurement: one ``describe_cluster``
metadata request against the same admin client that is about to issue the
``CreateTopics``, taken once per provisioning pass. It is a plain ``Metadata``
API call — the same class of request the provisioner already makes for
``describe_topics`` — not the ``DescribeTopicDynamicConfiguration`` call MSK IAM
denies.

Fail-open is not an option here, but neither is guessing: when the count cannot
be read the policy stays **unmeasured**, which means no ceiling and therefore no
reduction. A declared RF that the broker cannot host then fails loudly at
``CreateTopics`` instead of being quietly downgraded.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from omnibase_infra.topics.model_topic_provisioning_policy import (
    ModelTopicProvisioningPolicy,
)

logger = logging.getLogger(__name__)


async def probe_broker_count(admin: object) -> int | None:
    """Return the live broker count, or ``None`` when it cannot be measured.

    Args:
        admin: A started ``AIOKafkaAdminClient`` (or any object exposing an
            awaitable ``describe_cluster()`` returning a mapping with a
            ``brokers`` sequence).

    Returns:
        The number of brokers the cluster reports, or ``None`` when the client
        does not expose ``describe_cluster``, the call fails, or the response
        carries no usable broker list.
    """
    describe_cluster = getattr(admin, "describe_cluster", None)
    if describe_cluster is None:
        logger.warning(
            "Admin client %s exposes no describe_cluster(); topic replication "
            "will be resolved WITHOUT a capacity ceiling — a declared "
            "replication factor above the broker's node count will fail at "
            "CreateTopics rather than being silently reduced (OMN-15395)",
            type(admin).__name__,
        )
        return None

    try:
        described = await describe_cluster()
    except Exception as exc:  # noqa: BLE001 — boundary: unmeasured beats guessed
        logger.warning(
            "describe_cluster() failed (%s); topic replication will be "
            "resolved WITHOUT a capacity ceiling rather than assuming a node "
            "count (OMN-15395)",
            type(exc).__name__,
        )
        return None

    if not isinstance(described, Mapping):
        logger.warning(
            "describe_cluster() returned %s, expected a mapping; leaving the "
            "replication capacity ceiling unmeasured (OMN-15395)",
            type(described).__name__,
        )
        return None

    brokers = described.get("brokers")
    if not isinstance(brokers, Sequence) or isinstance(brokers, (str, bytes)):
        logger.warning(
            "describe_cluster() carried no broker sequence; leaving the "
            "replication capacity ceiling unmeasured (OMN-15395)",
        )
        return None

    count = len(brokers)
    if count < 1:
        logger.warning(
            "describe_cluster() reported zero brokers; leaving the replication "
            "capacity ceiling unmeasured (OMN-15395)",
        )
        return None
    return count


async def bind_policy_to_broker_capacity(
    admin: object,
    policy: ModelTopicProvisioningPolicy,
) -> ModelTopicProvisioningPolicy:
    """Return ``policy`` bound to the live broker count, or unchanged.

    The durability floor and the profile are never altered by the measurement —
    see
    :meth:`~omnibase_infra.topics.model_topic_provisioning_policy.ModelTopicProvisioningPolicy.with_broker_capacity`.

    Args:
        admin: A started admin client.
        policy: The configuration-derived (unmeasured) policy.

    Returns:
        The measured policy, or ``policy`` itself when no count could be read.
    """
    broker_count = await probe_broker_count(admin)
    if broker_count is None:
        return policy
    measured = policy.with_broker_capacity(broker_count)
    logger.info(
        "Broker capacity measured: %d node(s) on the %s cluster; replication "
        "ceiling=%s, undeclared default=%s (OMN-15395)",
        broker_count,
        measured.profile.value,
        measured.capacity_replication_factor,
        measured.default_replication_factor,
    )
    return measured


__all__: list[str] = [
    "bind_policy_to_broker_capacity",
    "probe_broker_count",
]
