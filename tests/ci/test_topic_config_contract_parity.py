# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""W4 (OMN-13238): contract<->registry per-topic config parity.

Every config-bearing topic that was migrated out of
``ALL_PROVISIONED_TOPIC_SPECS`` into its owning omnibase_infra contract
(``published_events[].topic_config``) must resolve, from contract YAML, the
SAME partitions / replication_factor / kafka_config the legacy registry
declares. This guards against contract drift and proves the migration is
lossless.

The remaining (un-migrated) registry entries are owned by contracts in other
repos or are emitted directly by the runtime kernel; they stay in the
transitional registry (see the banner in platform_topic_suffixes.py) and are
NOT asserted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor
from omnibase_infra.topics import platform_topic_suffixes as pts

# The config-bearing topics whose owning contract lives in omnibase_infra AND
# which already had a ``published_events`` entry (so attaching ``topic_config``
# does not synthesize a new published event — preserving orchestrator/effect
# event-purity invariants). Each was migrated into the owning contract's
# existing ``published_events[].topic_config`` block in this PR.
#
# Config-bearing infra-owned topics that did NOT have a pre-existing
# published_events entry (or are cmd/intent topics that must not appear in
# published_events) stay in the transitional registry — migrating them requires
# either a new published event (a purity change out of scope here) or the
# omnibase_core event_bus schema extension tracked as a follow-up.
MIGRATED_TOPICS: frozenset[str] = frozenset(
    {
        "onex.evt.omnibase-infra.runner-health-snapshot.v1",
        "onex.evt.omnibase-infra.network-pool-status.v1",
        "onex.evt.omnibase-infra.row-count-diagnostic.v1",
        "onex.evt.omnibase-infra.llm-call-completed.v1",
        "onex.evt.omnibase-infra.agent-task-lifecycle.v1",
        "onex.evt.omnibase-infra.inference-response.v1",
        "onex.evt.platform.topic-catalog-response.v1",
        "onex.evt.platform.topic-catalog-changed.v1",
        "onex.evt.omniintelligence.llm-call-completed.v1",
    }
)


def _registry_spec_by_suffix() -> dict[str, pts.ModelTopicSpec]:
    """All registry specs across every group (independent of runtime gating)."""
    all_specs = (
        pts.ALL_PLATFORM_TOPIC_SPECS
        + pts.ALL_INTELLIGENCE_TOPIC_SPECS
        + pts.ALL_OMNIMEMORY_TOPIC_SPECS
        + pts.ALL_OMNIBASE_INFRA_TOPIC_SPECS
        + pts.ALL_VALIDATION_TOPIC_SPECS
        + pts.ALL_OMNINODE_ROUTING_TOPIC_SPECS
        + pts.ALL_OMNICLAUDE_TOPIC_SPECS
    )
    return {spec.suffix: spec for spec in all_specs}


def _contract_entries_by_topic() -> dict[str, object]:
    nodes_root = (
        Path(__file__).resolve().parents[2] / "src" / "omnibase_infra" / "nodes"
    )
    entries = ContractTopicExtractor().extract(nodes_root)
    return {e.topic: e for e in entries}


@pytest.mark.unit
def test_migrated_topics_resolve_same_config_from_contract() -> None:
    registry = _registry_spec_by_suffix()
    contracts = _contract_entries_by_topic()

    mismatches: list[str] = []
    for topic in sorted(MIGRATED_TOPICS):
        spec = registry.get(topic)
        assert spec is not None, f"{topic} missing from registry"
        entry = contracts.get(topic)
        assert entry is not None, (
            f"{topic} not resolvable from any omnibase_infra contract — "
            f"migration incomplete"
        )

        exp_partitions = spec.partitions
        exp_rf = spec.replication_factor
        exp_cfg = dict(spec.kafka_config) if spec.kafka_config else None

        got_partitions = entry.partitions  # type: ignore[attr-defined]
        got_rf = entry.replication_factor  # type: ignore[attr-defined]
        got_cfg = (
            dict(entry.kafka_config)  # type: ignore[attr-defined]
            if entry.kafka_config  # type: ignore[attr-defined]
            else None
        )

        if (got_partitions, got_rf, got_cfg) != (exp_partitions, exp_rf, exp_cfg):
            mismatches.append(
                f"  {topic}: contract={(got_partitions, got_rf, got_cfg)} "
                f"registry={(exp_partitions, exp_rf, exp_cfg)}"
            )

    assert not mismatches, "Per-topic config drift between contract and registry:\n" + (
        "\n".join(mismatches)
    )


@pytest.mark.unit
def test_migrated_topics_carry_config_in_contract() -> None:
    """Each migrated topic must declare config in its contract (not just exist)."""
    contracts = _contract_entries_by_topic()
    missing_config: list[str] = []
    for topic in sorted(MIGRATED_TOPICS):
        entry = contracts.get(topic)
        if entry is None:
            missing_config.append(f"{topic} (not in any contract)")
            continue
        if entry.partitions is None and entry.replication_factor is None:  # type: ignore[attr-defined]
            missing_config.append(f"{topic} (no topic_config block)")
    assert not missing_config, "Migrated topics missing contract topic_config:\n" + (
        "\n".join(missing_config)
    )


@pytest.mark.unit
def test_registry_is_transitional_not_load_bearing_for_runtime() -> None:
    """The registry docstring must mark it transitional (AC-8 no shadow registry).

    A regression that re-promotes the registry to a standing authority (deleting
    the transitional marker) fails this test.
    """
    doc = pts.__dict__.get("ALL_PROVISIONED_TOPIC_SPECS")
    assert doc is not None
    module_doc = pts.__doc__ or ""
    # The symbol docstring is attached to the module via the literal docstring;
    # assert the transitional banner constant lives in the source.
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "omnibase_infra"
        / "topics"
        / "platform_topic_suffixes.py"
    ).read_text()
    assert "TRANSITIONAL" in source
    assert "OMN-13238" in source
    assert "DO NOT add new entries here" in source
    # The misleading "single source of truth for topic creation" claim is gone.
    assert "single source of truth for topic creation" not in source
    assert module_doc is not None
