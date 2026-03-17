# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ContractTopicExtractor — contract-driven topic discovery."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from omnibase_infra.topics.contract_topic_extractor import (
    ContractTopicExtractor,
    NodeTopics,
    TopicManifest,
)


@pytest.mark.unit
class TestExtractFromFile:
    """Tests for ContractTopicExtractor.extract_from_file()."""

    def test_schema_a_flat_lists(self, tmp_path: Path) -> None:
        """Schema A: flat subscribe_topics and publish_topics lists."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            textwrap.dedent("""\
                name: "node_example_effect"
                event_bus:
                  subscribe_topics:
                    - "onex.evt.platform.node-registration.v1"
                    - "onex.cmd.platform.request-introspection.v1"
                  publish_topics:
                    - "onex.evt.platform.node-registered.v1"
            """)
        )
        extractor = ContractTopicExtractor()
        result = extractor.extract_from_file(contract)

        assert result.node_name == "node_example_effect"
        assert result.contract_path == contract
        assert set(result.subscribe_topics) == {
            "onex.evt.platform.node-registration.v1",
            "onex.cmd.platform.request-introspection.v1",
        }
        assert result.publish_topics == ["onex.evt.platform.node-registered.v1"]

    def test_no_event_bus_returns_empty(self, tmp_path: Path) -> None:
        """Contract with no event_bus block returns empty NodeTopics."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            textwrap.dedent("""\
                name: "node_no_bus"
                node_type: "COMPUTE"
            """)
        )
        extractor = ContractTopicExtractor()
        result = extractor.extract_from_file(contract)

        assert result.node_name == "node_no_bus"
        assert result.subscribe_topics == []
        assert result.publish_topics == []

    def test_subscribe_only(self, tmp_path: Path) -> None:
        """Contract with subscribe_topics but no publish_topics."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            textwrap.dedent("""\
                name: "node_consumer"
                event_bus:
                  subscribe_topics:
                    - "onex.evt.omnibase-infra.db-error.v1"
                  publish_topics: []
            """)
        )
        extractor = ContractTopicExtractor()
        result = extractor.extract_from_file(contract)

        assert result.subscribe_topics == ["onex.evt.omnibase-infra.db-error.v1"]
        assert result.publish_topics == []

    def test_publish_only(self, tmp_path: Path) -> None:
        """Contract with publish_topics but no subscribe_topics."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            textwrap.dedent("""\
                name: "node_producer"
                event_bus:
                  publish_topics:
                    - "onex.evt.omnibase-infra.baselines-computed.v1"
            """)
        )
        extractor = ContractTopicExtractor()
        result = extractor.extract_from_file(contract)

        assert result.subscribe_topics == []
        assert result.publish_topics == [
            "onex.evt.omnibase-infra.baselines-computed.v1"
        ]

    def test_schema_b_structured_dicts(self, tmp_path: Path) -> None:
        """Schema B: structured subscribe/publish with topic dicts."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            textwrap.dedent("""\
                name: "node_structured"
                event_bus:
                  subscribe_topics:
                    - topic: "onex.evt.github.pr-webhook.v1"
                      operation: "ingest"
                    - topic: "onex.cmd.artifact.reconcile.v1"
                      operation: "manual"
                  publish_topics:
                    - topic: "onex.evt.artifact.change-detected.v1"
                      success_topic: "onex.evt.artifact.change-detected-success.v1"
                      failure_topic: "onex.evt.artifact.change-detected-failed.v1"
            """)
        )
        extractor = ContractTopicExtractor()
        result = extractor.extract_from_file(contract)

        assert set(result.subscribe_topics) == {
            "onex.evt.github.pr-webhook.v1",
            "onex.cmd.artifact.reconcile.v1",
        }
        # All topic strings extracted from structured dicts
        assert "onex.evt.artifact.change-detected.v1" in result.publish_topics
        assert "onex.evt.artifact.change-detected-success.v1" in result.publish_topics
        assert "onex.evt.artifact.change-detected-failed.v1" in result.publish_topics

    def test_mixed_schema_deduplicates(self, tmp_path: Path) -> None:
        """Mixed flat and structured entries are deduplicated."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            textwrap.dedent("""\
                name: "node_mixed"
                event_bus:
                  subscribe_topics:
                    - "onex.evt.platform.node-registration.v1"
                    - topic: "onex.evt.platform.node-registration.v1"
                      operation: "register"
            """)
        )
        extractor = ContractTopicExtractor()
        result = extractor.extract_from_file(contract)

        assert result.subscribe_topics == ["onex.evt.platform.node-registration.v1"]

    def test_input_subscriptions_extracted(self, tmp_path: Path) -> None:
        """Topics from input_subscriptions are also extracted as subscribe topics."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            textwrap.dedent("""\
                name: "node_with_input_subs"
                input_subscriptions:
                  - topic: "onex.evt.github.pr-webhook.v1"
                    operation: "ingest"
                  - topic: "onex.cmd.artifact.reconcile.v1"
                    operation: "manual"
            """)
        )
        extractor = ContractTopicExtractor()
        result = extractor.extract_from_file(contract)

        assert set(result.subscribe_topics) == {
            "onex.evt.github.pr-webhook.v1",
            "onex.cmd.artifact.reconcile.v1",
        }


@pytest.mark.unit
class TestScan:
    """Tests for ContractTopicExtractor.scan() recursive discovery."""

    def test_scan_finds_all_contracts(self, tmp_path: Path) -> None:
        """scan() recursively finds all contract.yaml files."""
        # Create two node directories with contracts
        node_a = tmp_path / "node_alpha"
        node_a.mkdir()
        (node_a / "contract.yaml").write_text(
            textwrap.dedent("""\
                name: "node_alpha"
                event_bus:
                  subscribe_topics:
                    - "onex.evt.platform.node-registration.v1"
                  publish_topics:
                    - "onex.evt.platform.node-registered.v1"
            """)
        )
        node_b = tmp_path / "node_beta"
        node_b.mkdir()
        (node_b / "contract.yaml").write_text(
            textwrap.dedent("""\
                name: "node_beta"
                event_bus:
                  subscribe_topics:
                    - "onex.cmd.platform.request-introspection.v1"
            """)
        )
        # Non-contract yaml should be ignored
        (tmp_path / "not_a_contract.yaml").write_text("name: ignored")

        extractor = ContractTopicExtractor()
        manifest = extractor.scan(tmp_path)

        assert len(manifest.nodes) == 2
        assert "node_alpha" in manifest.nodes
        assert "node_beta" in manifest.nodes

    def test_scan_nested_contracts(self, tmp_path: Path) -> None:
        """scan() finds contracts in nested directories."""
        nested = tmp_path / "deep" / "node_nested"
        nested.mkdir(parents=True)
        (nested / "contract.yaml").write_text(
            textwrap.dedent("""\
                name: "node_nested"
                event_bus:
                  publish_topics:
                    - "onex.evt.test.nested-topic.v1"
            """)
        )

        extractor = ContractTopicExtractor()
        manifest = extractor.scan(tmp_path)

        assert "node_nested" in manifest.nodes


@pytest.mark.unit
class TestTopicManifest:
    """Tests for TopicManifest aggregate properties."""

    def test_all_unique_topics(self) -> None:
        """all_unique_topics returns deduplicated set of all topic strings."""
        manifest = TopicManifest(
            nodes={
                "node_a": NodeTopics(
                    node_name="node_a",
                    contract_path=Path("/a/contract.yaml"),
                    subscribe_topics=["onex.evt.platform.node-registration.v1"],
                    publish_topics=["onex.evt.platform.node-registered.v1"],
                ),
                "node_b": NodeTopics(
                    node_name="node_b",
                    contract_path=Path("/b/contract.yaml"),
                    subscribe_topics=[
                        "onex.evt.platform.node-registration.v1",
                        "onex.cmd.platform.request-introspection.v1",
                    ],
                    publish_topics=[],
                ),
            }
        )

        unique = manifest.all_unique_topics
        assert unique == {
            "onex.evt.platform.node-registration.v1",
            "onex.evt.platform.node-registered.v1",
            "onex.cmd.platform.request-introspection.v1",
        }

    def test_get_node_topics(self) -> None:
        """get_node_topics returns NodeTopics for a known node."""
        nt = NodeTopics(
            node_name="node_x",
            contract_path=Path("/x/contract.yaml"),
            subscribe_topics=["onex.evt.test.x.v1"],
            publish_topics=[],
        )
        manifest = TopicManifest(nodes={"node_x": nt})

        assert manifest.get_node_topics("node_x") is nt
        assert manifest.get_node_topics("node_missing") is None

    def test_all_topics_flat_list(self) -> None:
        """all_topics returns a flat list of all ExtractedTopic entries."""
        manifest = TopicManifest(
            nodes={
                "node_a": NodeTopics(
                    node_name="node_a",
                    contract_path=Path("/a/contract.yaml"),
                    subscribe_topics=["onex.evt.test.a.v1"],
                    publish_topics=["onex.evt.test.b.v1"],
                ),
            }
        )

        topics = manifest.all_topics
        assert len(topics) == 2
        topic_strings = {t.topic for t in topics}
        assert topic_strings == {"onex.evt.test.a.v1", "onex.evt.test.b.v1"}
