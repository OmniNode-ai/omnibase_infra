# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Subscription verification probe.

Verifies that a contract's declared subscribe_topics are actually subscribed
by the consumer group. Consumer group IDs are derived via
compute_consumer_group_id() -- never hardcoded.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from omnibase_infra.models import ModelNodeIdentity

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
)
from omnibase_infra.verification.models.model_contract_check_result import (
    ModelContractCheckResult,
)

logger = logging.getLogger(__name__)

_OMNIBASE_ENV = Path.home() / ".omnibase" / ".env"


def _rpk_env() -> dict[str, str]:
    """Return os.environ merged with ~/.omnibase/.env for rpk subprocess calls."""
    env = dict(os.environ)
    if _OMNIBASE_ENV.exists():
        with open(_OMNIBASE_ENV) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env.setdefault(key.strip(), value.strip())
    return env


# Type alias: given a consumer group ID, return the set of topics it subscribes to.
# Raises on infrastructure failure.
KafkaAdminFn = Callable[[str], set[str]]
GroundingTag = Literal["EXACT", "DISCOVERED", "FABRICATED"]


def _parse_topics_from_rpk_describe_json(stdout: str) -> set[str]:
    """Extract subscribed topic names from ``rpk group describe --format json``."""
    import json

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"rpk returned invalid JSON: {stdout[:200]}") from exc

    topics: set[str] = set()
    members = data.get("members", [])
    for member in members:
        assignments = member.get("assignments", [])
        for assignment in assignments:
            topic = assignment.get("topic", "")
            if topic:
                topics.add(topic)
    return topics


# Per-topic split-group infix. The Kafka event bus scopes each consumer group
# per topic as "...__t.{topic}" (see util_consumer_group.apply_instance_discriminator
# and event_bus_kafka._scope_group_id_per_topic). The optional ".__i.{instance}"
# infix is appended *before* the topic suffix in multi-container runtimes.
_TOPIC_SUFFIX_INFIX = ".__t."


def _rpk_group_list() -> list[dict[str, object]]:
    """Return the raw ``rpk group list --format json`` payload as a list."""
    try:
        result = subprocess.run(
            ["rpk", "group", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_rpk_env(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("rpk not found on PATH") from exc

    if result.returncode != 0:
        raise RuntimeError(f"rpk group list failed: {result.stderr.strip()}")

    import json

    try:
        groups = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"rpk returned invalid JSON: {result.stdout[:200]}") from exc

    return groups if isinstance(groups, list) else []


def _topic_from_split_group_name(group_name: str) -> str | None:
    """Extract the subscribed topic from a per-topic split-group name.

    The runtime scopes each consumer group per topic with a ``.__t.{topic}``
    suffix, e.g.::

        local.omnibase_infra.node_registration_orchestrator.consume.1.1.1.__i.runtime-main.__t.onex.evt.platform.node-heartbeat.v1

    Returns the topic (everything after the *last* ``.__t.`` infix) or None if
    the group is not a per-topic split group.
    """
    marker = group_name.rfind(_TOPIC_SUFFIX_INFIX)
    if marker == -1:
        return None
    topic = group_name[marker + len(_TOPIC_SUFFIX_INFIX) :]
    return topic or None


def _list_topic_scoped_groups(base_group_id: str) -> list[str]:
    """List topic-scoped consumer groups derived from a base group ID."""
    direct_prefix = f"{base_group_id}{_TOPIC_SUFFIX_INFIX}"
    instance_prefix = f"{base_group_id}.__i."
    scoped_groups: list[str] = []
    for group in _rpk_group_list():
        group_name = group.get("name", "") if isinstance(group, dict) else ""
        if isinstance(group_name, str) and group_name.startswith(
            (direct_prefix, instance_prefix)
        ):
            scoped_groups.append(group_name)
    return scoped_groups


def _discover_split_group_topics(contract_name: str) -> set[str]:
    """Discover subscribed topics from live per-topic split-groups by contract name.

    The legacy group-id derivation can compute a base group name that no longer
    matches the live runtime (e.g. when the service/version/node-name naming
    convention has drifted). Rather than reconstruct the exact base group, this
    discovers the runtime's actual per-topic split-groups directly: any live
    consumer group whose name contains the contract's node name AND carries a
    ``.__t.{topic}`` suffix contributes its topic. The topic is parsed straight
    from the group name, so this stays correct across instance/version drift.

    Args:
        contract_name: The node name (e.g. ``node_registration_orchestrator``).

    Returns:
        The set of topics subscribed by live split-groups for this node. Empty
        when no matching split-group exists on the broker.
    """
    topics: set[str] = set()
    for group in _rpk_group_list():
        group_name = group.get("name", "") if isinstance(group, dict) else ""
        if not isinstance(group_name, str) or contract_name not in group_name:
            continue
        topic = _topic_from_split_group_name(group_name)
        if topic is not None:
            topics.add(topic)
    return topics


def _rpk_fallback(group_id: str, *, contract_name: str | None = None) -> set[str]:
    """Fallback: query subscribed topics via rpk CLI.

    Resolution order (each step only runs if the previous returned nothing):

    1. Describe ``group_id`` directly (the derived base group).
    2. Aggregate per-topic split-groups anchored on ``group_id``
       (``{group_id}.__t.{topic}`` / ``{group_id}.__i.{instance}.__t.{topic}``).
    3. Discover live split-groups by ``contract_name`` and parse the topic
       directly from the group name. This handles the case where the derived
       base group name no longer matches the live naming convention (the root
       cause of the recurring false 7/7-missing FAIL, OMN-13554).

    Args:
        group_id: Consumer group ID to describe.
        contract_name: The node name used to discover live split-groups when
            the derived base group does not exist on the broker.

    Returns:
        Set of topic names the group is subscribed to.

    Raises:
        RuntimeError: If rpk is not available or returns an error.
    """
    try:
        result = subprocess.run(
            ["rpk", "group", "describe", group_id, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_rpk_env(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("rpk not found on PATH") from exc

    if result.returncode != 0:
        raise RuntimeError(f"rpk group describe failed: {result.stderr.strip()}")

    topics = _parse_topics_from_rpk_describe_json(result.stdout)
    if topics:
        return topics

    # The Kafka event bus scopes consumer groups per topic as
    # "{base_group_id}.__t.{topic}". Aggregate those groups when the base
    # group is empty or dead so verification matches the live runtime shape.
    scoped_topics: set[str] = set()
    for scoped_group_id in _list_topic_scoped_groups(group_id):
        scoped_result = subprocess.run(
            ["rpk", "group", "describe", scoped_group_id, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_rpk_env(),
        )
        if scoped_result.returncode != 0:
            continue
        scoped_topics.update(_parse_topics_from_rpk_describe_json(scoped_result.stdout))
    if scoped_topics:
        return scoped_topics

    # The derived base group did not match the live runtime naming convention.
    # Discover the actual per-topic split-groups by contract name and read the
    # topic straight from each group name (OMN-13554).
    if contract_name:
        return _discover_split_group_topics(contract_name)
    return scoped_topics


def _derive_consumer_group_id(
    contract_name: str,
    *,
    identity: ModelNodeIdentity | None = None,
) -> tuple[str, GroundingTag]:
    """Derive the consumer group ID from a contract name.

    Uses compute_consumer_group_id with a ModelNodeIdentity. When identity is
    provided, uses it directly. Otherwise attempts rpk discovery; if rpk is
    unavailable, falls back to unknown markers (fabricated identity = always
    QUARANTINE-grade).

    Args:
        contract_name: Contract name (e.g., "node_registration_orchestrator").
        identity: Optional ModelNodeIdentity. If None, attempts rpk discovery
            then falls back to unknown markers.

    Returns:
        Tuple of canonical consumer group ID and how it was grounded.
    """
    from omnibase_infra.enums import EnumConsumerGroupPurpose
    from omnibase_infra.models import ModelNodeIdentity
    from omnibase_infra.utils.util_consumer_group import compute_consumer_group_id

    if identity is not None:
        return (
            compute_consumer_group_id(identity, EnumConsumerGroupPurpose.CONSUME),
            "EXACT",
        )

    # Attempt rpk discovery for runtime identity
    discovered_identity = _discover_identity_via_rpk(contract_name)
    if discovered_identity is not None:
        return (
            compute_consumer_group_id(
                discovered_identity, EnumConsumerGroupPurpose.CONSUME
            ),
            "DISCOVERED",
        )

    # Fallback: fabricated identity with unknown markers
    fallback_identity = ModelNodeIdentity(
        env="unknown",
        service="unknown",
        node_name=contract_name,
        version="v0",
    )
    return (
        compute_consumer_group_id(fallback_identity, EnumConsumerGroupPurpose.CONSUME),
        "FABRICATED",
    )


def _subscription_verdict_for_grounding(
    grounding: GroundingTag,
) -> EnumValidationVerdict:
    """Return the verdict class appropriate for the consumer-group grounding."""
    if grounding == "FABRICATED":
        return EnumValidationVerdict.QUARANTINE
    return EnumValidationVerdict.FAIL


def _discover_identity_via_rpk(contract_name: str) -> ModelNodeIdentity | None:
    """Attempt to discover node identity via rpk group list.

    Returns:
        A ModelNodeIdentity if discovery succeeds, None otherwise.
    """
    try:
        result = subprocess.run(
            ["rpk", "group", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None

        import json

        groups = json.loads(result.stdout)
        # Look for a consumer group matching the contract name
        for group in groups if isinstance(groups, list) else []:
            group_name = group.get("name", "") if isinstance(group, dict) else ""
            if contract_name in group_name:
                # Parse identity from group name segments
                parts = group_name.split(".")
                if len(parts) >= 5:
                    from omnibase_infra.models import ModelNodeIdentity

                    return ModelNodeIdentity(
                        env=parts[0],
                        service=parts[1],
                        node_name=parts[2],
                        version=parts[4],
                    )
    # ONEX_EXCLUDE: blind_except - rpk discovery is best-effort; any failure returns None
    except Exception:  # noqa: BLE001
        pass
    return None


def check_subscriptions(
    parsed_contract: ModelParsedContractForVerification,
    kafka_admin_fn: KafkaAdminFn | None = None,
    *,
    identity: ModelNodeIdentity | None = None,
) -> list[ModelContractCheckResult]:
    """Verify a contract's declared subscribe_topics are actually subscribed.

    Args:
        parsed_contract: Parsed contract with subscribe_topics.
        kafka_admin_fn: Injectable callable(group_id) -> set[str] of subscribed
            topics. If None, uses rpk fallback.
        identity: Optional exact runtime identity to use for group derivation.

    Returns:
        One ModelContractCheckResult per subscribe_topic.
    """
    if not parsed_contract.subscribe_topics:
        return [
            ModelContractCheckResult(
                check_type=EnumContractCheckType.SUBSCRIPTION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.PASS,
                evidence="No subscribe_topics declared; nothing to verify.",
                contract_name=parsed_contract.name,
                message="Subscription check passed: no topics declared.",
            )
        ]

    group_id, grounding = _derive_consumer_group_id(
        parsed_contract.name,
        identity=identity,
    )

    # Resolve the admin function. The default rpk fallback is bound to the
    # contract name so it can discover live per-topic split-groups when the
    # derived base group does not match the live runtime convention (OMN-13554).
    admin_fn: KafkaAdminFn = kafka_admin_fn or (
        lambda gid: _rpk_fallback(gid, contract_name=parsed_contract.name)
    )

    # Query subscribed topics
    try:
        subscribed_topics = admin_fn(group_id)
    # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kafka admin unavailable for group %s: %s", group_id, exc)
        return [
            ModelContractCheckResult(
                check_type=EnumContractCheckType.SUBSCRIPTION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.QUARANTINE,
                evidence=(
                    f"Kafka admin API unavailable for group '{group_id}' "
                    f"(grounding={grounding}): {exc}"
                ),
                contract_name=parsed_contract.name,
                message="Subscription check quarantined: admin API unavailable.",
            )
            for _ in parsed_contract.subscribe_topics
        ]

    verdict = _subscription_verdict_for_grounding(grounding)
    message = "failed"
    if verdict == EnumValidationVerdict.QUARANTINE:
        message = "quarantined"

    # If the admin call succeeded but returned no topics at all, the consumer
    # group either doesn't exist or has zero members.
    if not subscribed_topics:
        return [
            ModelContractCheckResult(
                check_type=EnumContractCheckType.SUBSCRIPTION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=verdict,
                evidence=(
                    f"Consumer group '{group_id}' has no subscribed topics. "
                    f"Group may not exist or has zero active members. "
                    f"grounding={grounding}."
                ),
                contract_name=parsed_contract.name,
                message=(
                    f"Subscription check {message} for '{topic}': "
                    f"consumer group has no subscriptions."
                ),
            )
            for topic in parsed_contract.subscribe_topics
        ]

    # Check each declared topic individually
    results: list[ModelContractCheckResult] = []
    for topic in parsed_contract.subscribe_topics:
        if topic in subscribed_topics:
            results.append(
                ModelContractCheckResult(
                    check_type=EnumContractCheckType.SUBSCRIPTION,
                    severity=EnumCheckSeverity.REQUIRED,
                    verdict=EnumValidationVerdict.PASS,
                    evidence=(
                        f"Topic '{topic}' is subscribed by group '{group_id}' "
                        f"(grounding={grounding})."
                    ),
                    contract_name=parsed_contract.name,
                    message=f"Subscription check passed for '{topic}'.",
                )
            )
        else:
            results.append(
                ModelContractCheckResult(
                    check_type=EnumContractCheckType.SUBSCRIPTION,
                    severity=EnumCheckSeverity.REQUIRED,
                    verdict=verdict,
                    evidence=(
                        f"Topic '{topic}' is NOT subscribed by group '{group_id}'. "
                        f"Subscribed topics: {sorted(subscribed_topics)}. "
                        f"grounding={grounding}."
                    ),
                    contract_name=parsed_contract.name,
                    message=f"Subscription check {message} for '{topic}'.",
                )
            )

    return results


__all__: list[str] = [
    "GroundingTag",
    "KafkaAdminFn",
    "check_subscriptions",
]
