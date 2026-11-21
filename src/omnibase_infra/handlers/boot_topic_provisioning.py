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
import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

import yaml

from omnibase_infra.contracts.contract_yaml import read_contract_yaml
from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.models.event_bus.model_contract_topic_entry import (
    TENANT_SLUG_PATTERN,
)
from omnibase_infra.protocols.protocol_topic_provisioner import (
    ProtocolTopicProvisioner,
)

if TYPE_CHECKING:
    from omnibase_infra.event_bus.model_topic_validation_result import (
        ModelTopicValidationResult,
    )
    from omnibase_infra.runtime.auto_wiring.models import (
        ModelAutoWiringManifest,
        ModelDiscoveredContract,
    )

logger = logging.getLogger(__name__)

# The topic grammar a declared name must match before boot creates it (OMN-25650):
# a canonical topic, or one tenant's ``tenant-<slug>.`` copy of it (OMN-25548).
_DECLARED_TOPIC = re.compile(
    rf"^(?:tenant-{TENANT_SLUG_PATTERN}\.)?"
    r"onex\.(?:evt|cmd|snapshot)\.[a-z0-9-]+\.[a-z0-9._-]+\.v\d+$"
)


def _topic_names(value: object) -> list[str]:
    """Return the topic-shaped strings in a declared value (a string, list or mapping)."""
    if isinstance(value, str):
        items: Sequence[object] = (value,)
    elif isinstance(value, Mapping):
        items = tuple(value.values())
    elif isinstance(value, (list, tuple)):
        items = tuple(value)
    else:
        return []
    return [i for i in items if isinstance(i, str) and _DECLARED_TOPIC.match(i)]


def declared_topics_outside_event_bus(raw: Mapping[str, object]) -> tuple[str, ...]:
    """Return the topics a contract declares outside ``event_bus`` (OMN-25650).

    A node served by its own host process declares ``runtime_dispatch`` and no
    ``event_bus.subscribe_topics``, so the runtime never wires it, yet the process
    that serves it reads ``runtime_dispatch.command_topic`` and answers on
    ``runtime_dispatch.terminal_events`` / ``terminal_event``, and its readers are
    named in ``externally_consumed_topics`` and ``published_events``. Only
    topic-shaped names are returned, de-duplicated in declaration order.
    """
    found: list[str] = []
    runtime_dispatch = raw.get("runtime_dispatch")
    if isinstance(runtime_dispatch, Mapping):
        found += _topic_names(runtime_dispatch.get("command_topic"))
        found += _topic_names(runtime_dispatch.get("terminal_events"))
    found += _topic_names(raw.get("terminal_event"))
    found += _topic_names(raw.get("externally_consumed_topics"))
    published = raw.get("published_events")
    if isinstance(published, (list, tuple)):
        for event in published:
            if isinstance(event, Mapping):
                found += _topic_names(event.get("topic"))
    return tuple(dict.fromkeys(found))


def contract_publish_topics(contract: ModelDiscoveredContract) -> tuple[str, ...]:
    """Return the de-duplicated ``event_bus.publish_topics`` the contract declares.

    Names come from the contract's own declaration only, never a Python literal.
    """
    if contract.event_bus is None:
        return ()
    publish_topics = contract.event_bus.publish_topics
    return tuple(dict.fromkeys(t for t in publish_topics if t and t.strip()))


def _read_declared_topics(contract_path: Path, contract_name: str) -> tuple[str, ...]:
    """Read the topics the contract file declares outside ``event_bus``.

    An unreadable file leaves the contract with its ``event_bus.publish_topics``
    alone and is logged with the contract and path; it is never fatal to the boot.
    """
    try:
        raw = read_contract_yaml(contract_path)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "Boot could not read the declared topics of contract '%s' at %s (%s); "
            "creating its event_bus publish topics only (non-fatal)",
            contract_name,
            contract_path,
            exc,
        )
        return ()
    return declared_topics_outside_event_bus(raw) if isinstance(raw, Mapping) else ()


def contract_declared_topics(contract: ModelDiscoveredContract) -> tuple[str, ...]:
    """Return every topic boot creates for an unwired contract (OMN-25650).

    The ``event_bus.publish_topics`` (OMN-23951) and the topics the contract file
    declares outside ``event_bus`` (:func:`declared_topics_outside_event_bus`).
    Reads the file, so callers on the event loop run it in a thread.
    """
    return tuple(
        dict.fromkeys(
            (
                *contract_publish_topics(contract),
                *_read_declared_topics(contract.contract_path, contract.name),
            )
        )
    )


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
    """Create the declared topics of every contract the interleave never reached.

    Each contract in *manifest* whose name is not in *interleaved_contract_names*
    and that declares publish topics, or topics outside ``event_bus``
    (:func:`contract_declared_topics`, OMN-25650), has them ensured, with at most
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
    ]
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _provision(contract: ModelDiscoveredContract) -> tuple[str, ...]:
        async with semaphore:
            topics = await asyncio.to_thread(contract_declared_topics, contract)
            if not topics:
                return ()
            return await ensure_topics_exist(
                provisioner, topics, contract_name=contract.name
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
    "contract_declared_topics",
    "contract_publish_topics",
    "declared_topics_outside_event_bus",
    "ensure_topics_exist",
    "provision_unattached_publish_topics",
]


class ProtocolProvisionedTopicsEnsurer(Protocol):
    """The kernel's one ``TopicProvisioner``, as the auto-create pass uses it."""

    async def ensure_provisioned_topics_exist(
        self, correlation_id: UUID | None = None
    ) -> Mapping[str, list[str] | str]: ...


class ProtocolTopicValidator(Protocol):
    """The kernel's ``TopicStartupValidator``, as the auto-create pass uses it."""

    async def validate(
        self, correlation_id: UUID | None = None, *, log_missing: bool = True
    ) -> ModelTopicValidationResult: ...


async def auto_create_missing_topics(
    provisioner: ProtocolProvisionedTopicsEnsurer | None,
    validator: ProtocolTopicValidator,
    validation_result: ModelTopicValidationResult,
    *,
    correlation_id: UUID | None,
    strict_validation: bool,
) -> ModelTopicValidationResult:
    """Create the topics a failed startup validation found missing (OMN-7810).

    Handles topics added to the provisioning registry but not yet created on the
    broker (first startup after adding new topic suffixes), then validates again
    and returns the new result. The pass reuses the kernel's one
    ``TopicProvisioner`` (OMN-26359) rather than building a second one and
    scanning every contract again; with no provisioner (the kernel's
    construction failed) the original result stands. A durability-policy
    violation (``TopicReplicationPolicyError``) re-raises (OMN-15395); any other
    failure is logged and leaves the original result in place.
    """
    if provisioner is None:
        logger.warning(
            "Auto-create missing topics skipped: the kernel built no topic "
            "provisioner (correlation_id=%s)",
            correlation_id,
        )
        return validation_result
    try:
        auto_result = await provisioner.ensure_provisioned_topics_exist(
            correlation_id=correlation_id,
        )
        if auto_result["created"]:
            logger.info(
                "Auto-created %d missing topics after validation: %s "
                "(correlation_id=%s)",
                len(auto_result["created"]),
                auto_result["created"],
                correlation_id,
            )
        validation_result = await validator.validate(
            correlation_id=correlation_id,
            log_missing=strict_validation,
        )
        if validation_result.status == "success":
            logger.info(
                "Topic validation recovered after auto-create (correlation_id=%s)",
                correlation_id,
            )
    except TopicReplicationPolicyError:
        raise
    except Exception:
        logger.exception(
            "Auto-create missing topics failed (best-effort) (correlation_id=%s)",
            correlation_id,
        )
    return validation_result
