# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The real ``dlq_topic_resolver`` for the S6 core runtime (epic OMN-14717, §b).

``RuntimeDispatch`` ships a placeholder resolver that returns ``f"{topic}.dlq"`` — a
6-segment non-ONEX topic no provisioner creates, so a dead-letter ``send`` would target
an unprovisioned topic (latent DLQ-loss bug, verifier's flag). The composition root MUST
inject a real resolver. This module builds it:

1. **Contract-declared first.** If the owning contract's ``event_bus.dlq_topics`` names a
   DLQ topic, the source topic maps to it verbatim.
2. **Derive otherwise.** For a source topic ``onex.{kind}.{producer}.{event-name}.v{n}``
   the DLQ is the ONEX-canonical name produced by
   ``omnibase_infra.event_bus.topic_constants.build_dlq_topic`` (fixed ``onex.dlq.`` prefix,
   ``omnibase-infra`` scope). We REUSE ``build_dlq_topic`` rather than re-derive the
   string so the DLQ name matches exactly what the provisioner creates.
3. **Fail closed** on an unparseable source topic (not the ONEX shape) — a
   ``ModelOnexError`` at resolve time, never a ``.dlq``-suffixed non-topic.

``build_delegation_dlq_resolver`` also returns the full resolved DLQ set for the
allowlisted source topics so the boot provisioning pass can create them BEFORE the first
dead-letter (R-6).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import yaml

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.event_bus.topic_constants import build_dlq_topic
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DlqTopicResolver",
    "build_delegation_dlq_resolver",
    "derive_canonical_dlq_topic",
    "load_contract_dlq_topics",
]

DlqTopicResolver = Callable[[str], str]


def load_contract_dlq_topics(contract_path: Path) -> tuple[str, ...]:
    """Read ``event_bus.dlq_topics`` from a contract YAML (empty tuple when absent).

    The infra ``ModelEventBusWiring`` discovery model does not capture ``dlq_topics``,
    so it is read directly from the contract source (same approach as
    ``load_published_events_map``). Malformed / non-string entries are skipped.
    """
    if not contract_path.exists():
        return ()
    try:
        with contract_path.open() as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        logger.warning(
            "Failed to parse contract YAML %s for dlq_topics: %s", contract_path, exc
        )
        return ()
    if not isinstance(data, dict):
        return ()
    event_bus = data.get("event_bus")
    if not isinstance(event_bus, dict):
        return ()
    declared = event_bus.get("dlq_topics")
    if not isinstance(declared, list):
        return ()
    return tuple(t for t in declared if isinstance(t, str) and t.strip())


def derive_canonical_dlq_topic(source_topic: str) -> str:
    """Derive the ONEX-canonical DLQ topic for an ``onex.*`` source topic (§b.2).

    ``onex.{kind}.{producer}.{event-name}.v{n}`` → ``build_dlq_topic(event-name, v{n})``
    = ``onex.dlq.omnibase-infra.{event-name}.v{n}``. Fail-closed on a topic that does not
    follow the ONEX 5+-segment convention with a trailing ``v<digits>`` version.
    """
    parts = source_topic.split(".")
    version = parts[-1] if parts else ""
    if (
        len(parts) < 5
        or parts[0] != "onex"
        or not (version.startswith("v") and version[1:].isdigit())
    ):
        raise ModelOnexError(
            message=(
                f"S6 DLQ resolver: source topic {source_topic!r} does not follow the "
                "ONEX 'onex.{kind}.{producer}.{event-name}.v{n}' convention, so no "
                "canonical DLQ topic can be derived. Refusing to emit a non-ONEX "
                "'<topic>.dlq' name the provisioner never creates (fail closed)."
            ),
            error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
        )
    event_name = parts[3]
    # Reuse build_dlq_topic so the derived name is byte-identical to the provisioned one.
    return build_dlq_topic(event_name, version=version)


def build_delegation_dlq_resolver(
    contracts: Sequence[ModelDiscoveredContract],
    allowlist: frozenset[str],
    *,
    contract_dlq_loader: Callable[[Path], Sequence[str]] = load_contract_dlq_topics,
) -> tuple[DlqTopicResolver, frozenset[str]]:
    """Build ``(resolver, provision_set)`` for the allowlisted source topics (§b).

    ``resolver(topic)`` returns the contract-declared DLQ when the owning contract names
    one for an allowlisted topic, else the ONEX-canonical derived DLQ (works for ANY
    well-formed ONEX topic, including an unmapped topic that fails closed to DLQ). The
    ``provision_set`` is every DLQ topic the allowlisted source topics resolve to, for
    the boot provisioning pass.
    """
    declared_map: dict[str, str] = {}
    for contract in contracts:
        if contract.event_bus is None:
            continue
        allowlisted = [t for t in contract.event_bus.subscribe_topics if t in allowlist]
        if not allowlisted:
            continue
        declared = tuple(contract_dlq_loader(contract.contract_path))
        if not declared:
            continue
        if len(declared) > 1:
            # Ambiguous: cannot map each source topic to a single declared DLQ.
            raise ModelOnexError(
                message=(
                    f"S6 DLQ resolver: contract {contract.name!r} declares "
                    f"{len(declared)} dlq_topics {list(declared)} for allowlisted "
                    f"topics {allowlisted}; cannot deterministically map source -> DLQ. "
                    "Declare exactly one dlq_topic or none (derive)."
                ),
                error_code=EnumCoreErrorCode.CONTRACT_VALIDATION_ERROR,
            )
        for topic in allowlisted:
            declared_map[topic] = declared[0]

    # Precompute the provision set for the allowlisted source topics (declared or derived).
    provision: set[str] = set()
    for topic in allowlist:
        provision.add(declared_map.get(topic) or derive_canonical_dlq_topic(topic))

    def _resolver(topic: str) -> str:
        declared_dlq = declared_map.get(topic)
        if declared_dlq is not None:
            return declared_dlq
        return derive_canonical_dlq_topic(topic)

    return _resolver, frozenset(provision)
