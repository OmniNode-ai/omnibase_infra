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
``CreateTopics``, taken **once per provisioner instance** — the attempt is
memoized, not just its success, so an unmeasurable cluster is probed once and
not re-probed on every entrypoint (OMN-15395 D4). It is a plain ``Metadata``
API call — the same class of request the provisioner already makes for
``describe_topics`` — not the ``DescribeTopicDynamicConfiguration`` call MSK IAM
denies.

Fail-open is not an option here, but neither is guessing: when the count cannot
be read the policy stays **unmeasured**, which means no ceiling and therefore no
reduction. A declared RF that the broker cannot host then fails loudly at
``CreateTopics``: the provisioner classifies the broker's
``INVALID_REPLICATION_FACTOR`` rejection and raises
:class:`~omnibase_infra.errors.TopicReplicationPolicyError` out of the
best-effort boundary rather than appending the topic to ``failed`` behind a
``logger.warning`` (OMN-15395 D5). Quietly downgrading is not an option and
quietly *not creating* is not one either — an unmeasurable probe must not leave
a topic silently absent.

Two client shapes are supported, because there are two live ``CreateTopics``
paths in this repository and both must resolve capacity the same way:

* the async ``AIOKafkaAdminClient`` used by the runtime provisioner
  (:func:`probe_broker_count` / :func:`bind_policy_to_broker_capacity`); and
* the synchronous ``confluent_kafka.admin.AdminClient`` used by the operator
  CLI ``scripts/create_kafka_topics.py``, whose ``list_topics()`` already
  returns a ``ClusterMetadata`` carrying the broker list
  (:func:`broker_count_from_cluster_metadata`).

Both funnel into :func:`bind_policy_to_broker_count`, so a ceiling installed by
either path obeys the identical invariants.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from omnibase_infra.topics.model_topic_provisioning_policy import (
    ModelTopicProvisioningPolicy,
)

logger = logging.getLogger(__name__)


# Kafka error code 38. Matched by wire name rather than by importing
# ``aiokafka.errors.InvalidReplicationFactorError``: the classification must
# survive a driver that wraps or re-raises the broker error, and it must not
# make this module's fail-closed behaviour depend on an optional import that is
# absent in the ``aiokafka not available`` degradation path.
INVALID_REPLICATION_FACTOR_ERRNO = 38
_INVALID_REPLICATION_FACTOR_MARKERS = (
    "INVALID_REPLICATION_FACTOR",
    "InvalidReplicationFactor",
)


def is_invalid_replication_factor_error(exc: BaseException) -> bool:
    """True when the broker rejected a ``CreateTopics`` for its replica count.

    OMN-15395 (D5): the capacity ceiling is a MEASUREMENT, so an unmeasurable
    cluster gets no ceiling and a contract-declared RF the broker cannot host
    reaches ``CreateTopics`` unreduced — deliberately, because guessing a
    ceiling is a silent durability downgrade. That design is only honest if the
    broker's refusal is then LOUD. Landing it in the generic
    ``except Exception`` boundary made it a ``logger.warning`` plus a name in
    ``failed``, indistinguishable from a transient connection blip, with the
    pass returning ``status="partial"`` and the topic silently absent.
    """
    if type(exc).__name__ == "InvalidReplicationFactorError":
        return True
    if getattr(exc, "errno", None) == INVALID_REPLICATION_FACTOR_ERRNO:
        return True
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in _INVALID_REPLICATION_FACTOR_MARKERS)


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


def broker_count_from_cluster_metadata(cluster_metadata: object) -> int | None:
    """Return the live broker count from a synchronous ``ClusterMetadata``.

    ``confluent_kafka.admin.AdminClient.list_topics()`` already carries the
    cluster's broker map, so the operator CLI measures capacity from the SAME
    metadata request it makes to diff topics — zero extra round trips, and no
    second, differently-shaped notion of "how many brokers are there".

    Args:
        cluster_metadata: A ``confluent_kafka.admin.ClusterMetadata`` (or any
            object exposing a sized ``brokers`` attribute).

    Returns:
        The number of brokers, or ``None`` when the count cannot be read.
    """
    brokers = getattr(cluster_metadata, "brokers", None)
    if brokers is None:
        logger.warning(
            "Cluster metadata %s carries no broker map; topic replication will "
            "be resolved WITHOUT a capacity ceiling rather than assuming a node "
            "count (OMN-15395)",
            type(cluster_metadata).__name__,
        )
        return None
    try:
        count = len(brokers)
    except TypeError:
        logger.warning(
            "Cluster metadata broker map is not sized (%s); leaving the "
            "replication capacity ceiling unmeasured (OMN-15395)",
            type(brokers).__name__,
        )
        return None
    if count < 1:
        logger.warning(
            "Cluster metadata reported zero brokers; leaving the replication "
            "capacity ceiling unmeasured (OMN-15395)",
        )
        return None
    return count


def bind_policy_to_broker_count(
    policy: ModelTopicProvisioningPolicy,
    broker_count: int | None,
) -> ModelTopicProvisioningPolicy:
    """Return ``policy`` bound to ``broker_count``, or unchanged when ``None``.

    The single binding seam shared by the async runtime provisioner and the
    synchronous operator CLI. The durability floor and the profile are never
    altered by the measurement — see
    :meth:`~omnibase_infra.topics.model_topic_provisioning_policy.ModelTopicProvisioningPolicy.with_broker_capacity`.

    Args:
        policy: The configuration-derived (unmeasured) policy.
        broker_count: A measured live node count, or ``None`` when unmeasurable.

    Returns:
        The measured policy, or ``policy`` itself when no count was measured.
    """
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


async def bind_policy_to_broker_capacity(
    admin: object,
    policy: ModelTopicProvisioningPolicy,
) -> ModelTopicProvisioningPolicy:
    """Return ``policy`` bound to the live broker count, or unchanged.

    Args:
        admin: A started admin client.
        policy: The configuration-derived (unmeasured) policy.

    Returns:
        The measured policy, or ``policy`` itself when no count could be read.
    """
    return bind_policy_to_broker_count(policy, await probe_broker_count(admin))


__all__: list[str] = [
    "INVALID_REPLICATION_FACTOR_ERRNO",
    "bind_policy_to_broker_capacity",
    "bind_policy_to_broker_count",
    "broker_count_from_cluster_metadata",
    "is_invalid_replication_factor_error",
    "probe_broker_count",
]
