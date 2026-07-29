# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Ratchet: every declared replication_factor is >= the managed floor, and some exist.

OMN-15395 removed the module-level RF1 default, but eleven contracts in this
repo *declared* ``topic_config.replication_factor: 1`` outright — the provisioner
would have honoured them and minted RF1 topics on MSK even with the default
gone. Those eleven were raised to the managed durability floor (2); this test is
the mechanism that keeps them there, because a rule nobody can execute is not
enforcement.

Two assertions, and the second one matters as much as the first:

1. **No declaration below the managed floor.** An RF1 declaration mints an
   unrecoverable topic on MSK.
2. **The declarations still exist.** An earlier revision of this ratchet checked
   only (1), so *deleting every declaration* satisfied it — which is what
   happened: the tree went to zero declared replication factors, and against a
   refuse-on-undeclared managed policy that made topic provisioning a 100%
   no-op. A coverage floor is what makes "contract-driven" a property rather
   than a slogan.

Declaring RF2 (rather than leaving it undeclared) is safe for single-broker
self-hosted brokers because the provisioner measures the target cluster's node
count (``describe_cluster``) before resolving anything and reduces the declared
value to that measurement at the creation site. A declared value is never
raised, only reduced to what the broker demonstrably can host — and only when
the broker has actually been counted. On a multi-broker cluster the declared
RF2 reaches ``CreateTopics`` as RF2, whatever its auth mechanism.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.topics.model_topic_provisioning_policy import (
    MANAGED_MINIMUM_REPLICATION_FACTOR,
)

pytestmark = [pytest.mark.unit]

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra"

#: Coverage floor. Set to the number of producer-owned ``topic_config`` blocks
#: that declare a replication factor today. Raising it as contracts migrate is
#: the intended direction; lowering it means declarations were deleted, which is
#: the regression this guard exists to catch.
MINIMUM_DECLARED_REPLICATION_FACTORS = 11


def _walk_topic_configs(node: Any, source: Path) -> Iterator[tuple[Path, Any, Any]]:
    """Yield ``(source, topic, topic_config)`` for every declared topic config."""
    if isinstance(node, dict):
        topic_config = node.get("topic_config")
        if isinstance(topic_config, dict):
            yield source, node.get("topic"), topic_config
        for value in node.values():
            yield from _walk_topic_configs(value, source)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_topic_configs(item, source)


def _yaml_sources() -> list[Path]:
    sources = sorted(SRC_ROOT.glob("nodes/*/contract.yaml"))
    sources += sorted(SRC_ROOT.rglob("topics.yaml"))
    return sources


def _declared_replication_factors() -> list[tuple[str, Any, int]]:
    """Every ``(source, topic, replication_factor)`` declared across the tree."""
    declared: list[tuple[str, Any, int]] = []
    for source in _yaml_sources():
        try:
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # pragma: no cover - malformed YAML is a bug
            pytest.fail(f"{source} is not parseable YAML: {exc}")
        for path, topic, topic_config in _walk_topic_configs(document, source):
            replication_factor = topic_config.get("replication_factor")
            if isinstance(replication_factor, int):
                declared.append(
                    (
                        str(path.relative_to(SRC_ROOT.parent.parent)),
                        topic,
                        replication_factor,
                    )
                )
    return declared


def test_no_contract_declares_replication_factor_below_managed_floor() -> None:
    """A contract-declared RF below the managed floor is an MSK durability defect."""
    offenders = [
        f"{source}: {topic} (replication_factor={replication_factor})"
        for source, topic, replication_factor in _declared_replication_factors()
        if replication_factor < MANAGED_MINIMUM_REPLICATION_FACTOR
    ]

    assert not offenders, (
        f"Contracts declaring replication_factor < "
        f"{MANAGED_MINIMUM_REPLICATION_FACTOR} will mint under-replicated topics "
        "on the managed cluster; RF1 is unrecoverable data loss on a single "
        "broker failure (AWS_KAFKA_HIGH_RISK_CONFIG_RF_EQUALS_ONE). Raise the "
        "declaration to the managed floor — the self-hosted capacity ceiling "
        "reduces it to 1 for single-broker brokers automatically. Offenders: "
        f"{offenders}"
    )


def test_replication_factor_declarations_have_not_been_deleted() -> None:
    """Deleting declarations must not be a way to satisfy the RF1 ratchet.

    This is the guard the previous revision lacked. With a floor-only check, the
    cheapest way to make the ratchet green was to strip every
    ``replication_factor`` line from the tree — leaving zero contract-declared
    replication factors, which is the opposite of "explicit and contract-driven
    for every provisioned topic".
    """
    declared = _declared_replication_factors()

    assert len(declared) >= MINIMUM_DECLARED_REPLICATION_FACTORS, (
        f"only {len(declared)} contract-declared replication factor(s) found, "
        f"expected at least {MINIMUM_DECLARED_REPLICATION_FACTORS}. Declarations "
        "were deleted rather than raised to the managed floor. If a topic was "
        "genuinely retired, lower MINIMUM_DECLARED_REPLICATION_FACTORS in the "
        f"same commit and say why. Found: {declared}"
    )


def test_contract_sources_were_actually_scanned() -> None:
    """Guard the guard: an empty scan would make the ratchet vacuously green."""
    sources = _yaml_sources()
    assert len(sources) > 50, f"expected the full contract tree, scanned {len(sources)}"
    with_configs = [
        topic
        for source in sources
        for _, topic, _ in _walk_topic_configs(
            yaml.safe_load(source.read_text(encoding="utf-8")), source
        )
    ]
    assert with_configs, "no topic_config blocks found — the walker is broken"
