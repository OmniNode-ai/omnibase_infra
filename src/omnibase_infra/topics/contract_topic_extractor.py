# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-driven topic discovery for ONEX infrastructure.

Direction-aware extraction: tracks subscribe vs publish per-topic,
needed by ``check_topic_drift.py`` and tooling that cares about
topic directionality.

Canonical topic extraction and validation lives in
:mod:`omnibase_infra.tools.contract_topic_extractor`.  The dataclass
models here (``NodeTopics``, ``TopicManifest``, ``ExtractedTopic``)
additionally track subscribe/publish direction and Schema B structured
entries (``success_topic``, ``failure_topic``).

Supports two schema variants:

Schema A (flat lists)::

    event_bus:
      subscribe_topics:
        - "onex.evt.platform.node-registration.v1"
      publish_topics:
        - "onex.evt.platform.node-registered.v1"

Schema B (structured dicts)::

    event_bus:
      subscribe_topics:
        - topic: "onex.evt.github.pr-webhook.v1"
          operation: "ingest"
      publish_topics:
        - topic: "onex.evt.artifact.change-detected.v1"
          success_topic: "onex.evt.artifact.change-detected-success.v1"
          failure_topic: "onex.evt.artifact.change-detected-failed.v1"

Additionally extracts topics from ``input_subscriptions`` blocks.

Usage::

    extractor = ContractTopicExtractor()
    manifest = extractor.scan(Path("src/omnibase_infra/nodes"))
    for topic in manifest.all_unique_topics:
        print(topic)

.. versionadded:: 0.22.0
    OMN-5247: Initial implementation.

.. versionchanged:: 0.24.0
    OMN-5132: Consolidated with tools/ extractor; added installed-package
    discovery via delegation to
    :class:`~omnibase_infra.tools.contract_topic_extractor.ContractTopicExtractor`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedTopic:
    """A single topic extracted from a contract YAML.

    Attributes:
        topic: Full ONEX topic string (e.g., "onex.evt.platform.node-registration.v1").
        direction: Either "subscribe" or "publish".
        node_name: Name of the node that declares this topic.
        contract_path: Path to the contract.yaml file.
    """

    topic: str
    direction: str
    node_name: str
    contract_path: Path


@dataclass
class NodeTopics:
    """All topics declared by a single node contract.

    Attributes:
        node_name: Name of the node from the contract.
        contract_path: Path to the contract.yaml file.
        subscribe_topics: Deduplicated list of topics the node subscribes to.
        publish_topics: Deduplicated list of topics the node publishes to.
    """

    node_name: str
    contract_path: Path
    subscribe_topics: list[str] = field(default_factory=list)
    publish_topics: list[str] = field(default_factory=list)


@dataclass
class TopicManifest:
    """Aggregate manifest of all topics across all scanned contracts.

    Attributes:
        nodes: Mapping of node_name to NodeTopics.
    """

    nodes: dict[str, NodeTopics] = field(default_factory=dict)

    @property
    def all_topics(self) -> list[ExtractedTopic]:
        """Return a flat list of all ExtractedTopic entries across all nodes."""
        result: list[ExtractedTopic] = []
        for nt in self.nodes.values():
            for topic in nt.subscribe_topics:
                result.append(
                    ExtractedTopic(
                        topic=topic,
                        direction="subscribe",
                        node_name=nt.node_name,
                        contract_path=nt.contract_path,
                    )
                )
            for topic in nt.publish_topics:
                result.append(
                    ExtractedTopic(
                        topic=topic,
                        direction="publish",
                        node_name=nt.node_name,
                        contract_path=nt.contract_path,
                    )
                )
        return result

    @property
    def all_unique_topics(self) -> set[str]:
        """Return a deduplicated set of all topic strings across all nodes."""
        topics: set[str] = set()
        for nt in self.nodes.values():
            topics.update(nt.subscribe_topics)
            topics.update(nt.publish_topics)
        return topics

    def get_node_topics(self, node_name: str) -> NodeTopics | None:
        """Look up NodeTopics by node name.

        Args:
            node_name: The node name to look up.

        Returns:
            NodeTopics if found, None otherwise.
        """
        return self.nodes.get(node_name)


def _extract_topics_from_list(
    items: list[object],
) -> list[str]:
    """Extract topic strings from a mixed list of strings and dicts.

    Handles both Schema A (flat strings) and Schema B (dicts with topic,
    success_topic, failure_topic keys). Deduplicates while preserving order.

    Args:
        items: List of topic entries (strings or dicts).

    Returns:
        Deduplicated list of topic strings in insertion order.
    """
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        if isinstance(item, str):
            if item not in seen:
                seen.add(item)
                result.append(item)
        elif isinstance(item, dict):
            # Extract all topic-like keys from the dict
            for key in ("topic", "success_topic", "failure_topic"):
                val = item.get(key)
                if isinstance(val, str) and val not in seen:
                    seen.add(val)
                    result.append(val)

    return result


class ContractTopicExtractor:
    """Extract topic declarations from ONEX contract YAML files.

    Tracks subscribe vs publish direction per topic and handles Schema B
    entries (success_topic, failure_topic).  For validated, direction-agnostic
    topic extraction, use
    :class:`~omnibase_infra.tools.contract_topic_extractor.ContractTopicExtractor`.

    .. versionchanged:: 0.24.0
        OMN-5132: Added ``scan_installed_packages()`` delegation to the
        canonical tools/ extractor for installed-package discovery.
    """

    def extract_from_file(self, path: Path) -> NodeTopics:
        """Extract topic declarations from a single contract YAML file.

        Args:
            path: Path to a contract.yaml file.

        Returns:
            NodeTopics with all subscribe and publish topics found.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return NodeTopics(
                node_name="unknown",
                contract_path=path,
            )

        node_name = str(data.get("name", "unknown"))
        subscribe_topics: list[str] = []
        publish_topics: list[str] = []

        # Extract from event_bus block
        event_bus = data.get("event_bus")
        if isinstance(event_bus, dict):
            sub_raw = event_bus.get("subscribe_topics")
            if isinstance(sub_raw, list):
                subscribe_topics = _extract_topics_from_list(sub_raw)

            pub_raw = event_bus.get("publish_topics")
            if isinstance(pub_raw, list):
                publish_topics = _extract_topics_from_list(pub_raw)

        # Extract from input_subscriptions block (additional subscribe topics)
        input_subs = data.get("input_subscriptions")
        if isinstance(input_subs, list):
            additional = _extract_topics_from_list(input_subs)
            # Merge with dedup
            existing = set(subscribe_topics)
            for topic in additional:
                if topic not in existing:
                    existing.add(topic)
                    subscribe_topics.append(topic)

        return NodeTopics(
            node_name=node_name,
            contract_path=path,
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics,
        )

    def scan(self, root: Path) -> TopicManifest:
        """Recursively scan a directory for contract.yaml files and extract topics.

        Args:
            root: Root directory to scan (e.g., src/omnibase_infra/nodes).

        Returns:
            TopicManifest with all discovered node topics.
        """
        manifest = TopicManifest()

        for contract_path in sorted(root.rglob("contract.yaml")):
            try:
                node_topics = self.extract_from_file(contract_path)
                manifest.nodes[node_topics.node_name] = node_topics
            except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError):
                logger.warning(
                    "Failed to parse contract: %s",
                    contract_path,
                    exc_info=True,
                )

        return manifest

    def scan_installed_packages(self) -> TopicManifest:
        """Discover contracts from approved installed packages.

        Delegates to the canonical
        :class:`~omnibase_infra.tools.contract_topic_extractor.ContractTopicExtractor`
        for package discovery, then re-parses the discovered contract files
        through this class's direction-aware ``extract_from_file()`` method.

        Returns:
            TopicManifest with direction-aware topics from all installed
            approved packages.

        Raises:
            ValueError: If duplicate package discovery is detected.

        .. versionadded:: 0.24.0
            OMN-5132
        """
        from omnibase_infra.tools.contract_topic_extractor import (
            _APPROVED_PACKAGES,
        )
        from omnibase_infra.tools.contract_topic_extractor import (
            ContractTopicExtractor as _CanonicalExtractor,
        )

        manifest = TopicManifest()
        canonical = _CanonicalExtractor()

        for pkg_name in _APPROVED_PACKAGES:
            contract_paths = canonical._discover_package_contracts(pkg_name)
            if contract_paths is None:
                continue

            for contract_path in contract_paths:
                try:
                    node_topics = self.extract_from_file(contract_path)
                    manifest.nodes[node_topics.node_name] = node_topics
                except (
                    OSError,
                    yaml.YAMLError,
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    logger.warning(
                        "Failed to parse contract from package %s: %s",
                        pkg_name,
                        contract_path,
                        exc_info=True,
                    )

        return manifest


__all__: list[str] = [
    "ContractTopicExtractor",
    "ExtractedTopic",
    "NodeTopics",
    "TopicManifest",
]
