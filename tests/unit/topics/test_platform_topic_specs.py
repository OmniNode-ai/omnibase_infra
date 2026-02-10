# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""CI drift guard: contract YAML topics vs ALL_PLATFORM_TOPIC_SPECS registry.

Ensures that every platform topic declared in the orchestrator contract YAML
appears in ALL_PLATFORM_TOPIC_SPECS (or is a known consumer-only topic that
does not need provisioning by TopicProvisioner).

If this test fails, it means a new platform topic was added to the contract
but not registered in platform_topic_suffixes.py, or vice versa. This catches
contract / registry drift before it reaches production.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.topics import ALL_PLATFORM_SUFFIXES, ALL_PLATFORM_TOPIC_SPECS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONTRACT_PATH: Path = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_registration_orchestrator"
    / "contract.yaml"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Topics that are consumed by the orchestrator but NOT provisioned by
# TopicProvisioner. They are published by other services or by the node
# itself, so they do not need to appear in ALL_PLATFORM_TOPIC_SPECS.
KNOWN_CONSUMER_ONLY_TOPICS: set[str] = {
    "onex.evt.platform.registry-request-introspection.v1",
    "onex.cmd.platform.node-registration-acked.v1",
}


def _is_platform_topic(topic: str) -> bool:
    """Return True if *topic* is a platform-scoped ONEX topic."""
    return topic.startswith("onex.") and ".platform." in topic


def _load_contract() -> dict:
    """Load and return the orchestrator contract YAML."""
    assert CONTRACT_PATH.exists(), f"Contract YAML not found at {CONTRACT_PATH}"
    with CONTRACT_PATH.open() as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlatformTopicSpecs:
    """CI drift guard tests for platform topic spec registry."""

    def test_all_platform_suffixes_derived_from_specs(self) -> None:
        """ALL_PLATFORM_SUFFIXES must equal tuple(spec.suffix for spec in ALL_PLATFORM_TOPIC_SPECS)."""
        expected = tuple(spec.suffix for spec in ALL_PLATFORM_TOPIC_SPECS)
        assert expected == ALL_PLATFORM_SUFFIXES, (
            f"ALL_PLATFORM_SUFFIXES is out of sync with ALL_PLATFORM_TOPIC_SPECS.\n"
            f"Expected: {expected}\n"
            f"Got:      {ALL_PLATFORM_SUFFIXES}"
        )

    def test_contract_subscribe_topics_in_specs(self) -> None:
        """Every platform subscribe_topic in the contract YAML must be in ALL_PLATFORM_SUFFIXES.

        Consumer-only topics that are published by other services (and therefore
        not provisioned by TopicProvisioner) are exempt; they are listed in
        KNOWN_CONSUMER_ONLY_TOPICS.
        """
        contract = _load_contract()
        subscribe_topics: list[str] = contract["event_bus"]["subscribe_topics"]

        platform_subscribe = {t for t in subscribe_topics if _is_platform_topic(t)}
        suffix_set = set(ALL_PLATFORM_SUFFIXES)

        missing = platform_subscribe - suffix_set - KNOWN_CONSUMER_ONLY_TOPICS
        assert not missing, (
            f"Contract subscribe_topics contain platform topics not in "
            f"ALL_PLATFORM_TOPIC_SPECS and not in KNOWN_CONSUMER_ONLY_TOPICS:\n"
            f"  {sorted(missing)}\n\n"
            f"Either add them to ALL_PLATFORM_TOPIC_SPECS in "
            f"platform_topic_suffixes.py, or add them to "
            f"KNOWN_CONSUMER_ONLY_TOPICS in this test."
        )

    def test_contract_publish_topics_are_valid_onex_format(self) -> None:
        """All publish_topics in the contract YAML must follow the ONEX 5-segment format."""
        contract = _load_contract()
        publish_topics: list[str] = contract["event_bus"]["publish_topics"]

        for topic in publish_topics:
            assert topic.startswith("onex."), (
                f"Publish topic must start with 'onex.': {topic}"
            )
            parts = topic.split(".")
            assert len(parts) == 5, (
                f"Publish topic must have exactly 5 dot-separated segments: {topic} "
                f"(got {len(parts)})"
            )

    def test_specs_have_valid_partitions(self) -> None:
        """Every spec in ALL_PLATFORM_TOPIC_SPECS must have partitions >= 1."""
        for spec in ALL_PLATFORM_TOPIC_SPECS:
            assert spec.partitions >= 1, (
                f"Spec '{spec.suffix}' has invalid partition count: {spec.partitions}"
            )

    def test_no_duplicate_suffixes(self) -> None:
        """No duplicate suffixes in ALL_PLATFORM_TOPIC_SPECS."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for spec in ALL_PLATFORM_TOPIC_SPECS:
            if spec.suffix in seen:
                duplicates.append(spec.suffix)
            seen.add(spec.suffix)
        assert not duplicates, (
            f"Duplicate suffixes found in ALL_PLATFORM_TOPIC_SPECS: {duplicates}"
        )

    def test_snapshot_topic_has_compact_config(self) -> None:
        """The registration-snapshots spec must have kafka_config with cleanup.policy=compact."""
        snapshot_specs = [
            spec
            for spec in ALL_PLATFORM_TOPIC_SPECS
            if "registration-snapshots" in spec.suffix
        ]
        assert len(snapshot_specs) == 1, (
            f"Expected exactly 1 registration-snapshots spec, "
            f"found {len(snapshot_specs)}"
        )

        spec = snapshot_specs[0]
        assert spec.kafka_config is not None, (
            f"Snapshot spec '{spec.suffix}' must have kafka_config set"
        )
        assert spec.kafka_config.get("cleanup.policy") == "compact", (
            f"Snapshot spec '{spec.suffix}' must have cleanup.policy=compact, "
            f"got: {spec.kafka_config}"
        )
