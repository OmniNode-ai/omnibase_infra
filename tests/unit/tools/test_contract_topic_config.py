# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""W3 (OMN-13238): contract-declared per-topic config schema + extractor carry.

Verifies that a contract declaring a ``topic_config`` block produces a
``ModelContractTopicEntry`` (and, through the provisioner, a ``ModelTopicSpec`` /
``NewTopic``) carrying those values, while a contract with no ``topic_config``
falls back to the canonical ``ModelTopicSpec`` defaults.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor
from omnibase_infra.topics.model_topic_spec import (
    DEFAULT_EVENT_TOPIC_PARTITIONS,
    DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR,
)


def _write_contract(tmp_path: Path, body: str) -> Path:
    root = tmp_path / "node_x"
    root.mkdir()
    (root / "contract.yaml").write_text(textwrap.dedent(body))
    return tmp_path


@pytest.mark.unit
def test_topic_config_in_published_events_is_carried(tmp_path: Path) -> None:
    """A ``published_events[].topic_config`` block flows onto the entry."""
    root = _write_contract(
        tmp_path,
        """
        name: node_x
        published_events:
          - topic: "onex.evt.platform.foo.v1"
            event_type: "Foo"
            topic_config:
              partitions: 1
              replication_factor: 1
              kafka_config:
                retention.ms: "604800000"
                cleanup.policy: "delete"
        """,
    )
    entries = {e.topic: e for e in ContractTopicExtractor().extract(root)}
    foo = entries["onex.evt.platform.foo.v1"]
    assert foo.partitions == 1
    assert foo.replication_factor == 1
    assert foo.kafka_config is not None
    assert dict(foo.kafka_config) == {
        "retention.ms": "604800000",
        "cleanup.policy": "delete",
    }


@pytest.mark.unit
def test_topic_config_in_event_bus_publish_dict_is_carried(tmp_path: Path) -> None:
    """A structured ``event_bus.publish_topics`` item may carry topic_config."""
    root = _write_contract(
        tmp_path,
        """
        name: node_x
        event_bus:
          publish_topics:
            - topic: "onex.evt.platform.bar.v1"
              topic_config:
                partitions: 12
                replication_factor: 3
        """,
    )
    entries = {e.topic: e for e in ContractTopicExtractor().extract(root)}
    bar = entries["onex.evt.platform.bar.v1"]
    assert bar.partitions == 12
    assert bar.replication_factor == 3
    assert bar.kafka_config is None


@pytest.mark.unit
def test_no_topic_config_leaves_fields_none(tmp_path: Path) -> None:
    """Absent ``topic_config`` leaves config fields None (defaults apply later)."""
    root = _write_contract(
        tmp_path,
        """
        name: node_x
        published_events:
          - topic: "onex.evt.platform.baz.v1"
            event_type: "Baz"
        """,
    )
    entries = {e.topic: e for e in ContractTopicExtractor().extract(root)}
    baz = entries["onex.evt.platform.baz.v1"]
    assert baz.partitions is None
    assert baz.replication_factor is None
    assert baz.kafka_config is None


@pytest.mark.unit
def test_spec_builder_maps_topic_config_into_topic_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``TopicProvisioner._build_topic_specs`` maps contract config onto specs.

    A topic with ``topic_config`` produces a ``ModelTopicSpec`` with those
    values; a topic without it falls back to the ModelTopicSpec defaults.
    """
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    root = _write_contract(
        tmp_path,
        """
        name: node_x
        published_events:
          - topic: "onex.evt.platform.configured.v1"
            event_type: "Configured"
            topic_config:
              partitions: 1
              replication_factor: 1
              kafka_config:
                retention.ms: "604800000"
          - topic: "onex.evt.platform.defaulted.v1"
            event_type: "Defaulted"
        """,
    )

    from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

    provisioner = TopicProvisioner(contracts_root=root)
    specs = {s.suffix: s for s in provisioner._topic_specs}

    configured = specs["onex.evt.platform.configured.v1"]
    assert configured.partitions == 1
    assert configured.replication_factor == 1
    assert configured.kafka_config is not None
    assert dict(configured.kafka_config) == {"retention.ms": "604800000"}

    defaulted = specs["onex.evt.platform.defaulted.v1"]
    assert defaulted.partitions == DEFAULT_EVENT_TOPIC_PARTITIONS
    assert defaulted.replication_factor == DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR
    assert defaulted.kafka_config is None


@pytest.mark.unit
def test_topic_config_produces_matching_new_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The created NewTopic reflects the contract-declared topic_config.

    Mirrors the ``NewTopic(...)`` construction in
    ``ensure_provisioned_topics_exist`` without requiring a live broker.
    """
    pytest.importorskip("aiokafka")
    from aiokafka.admin import NewTopic

    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    root = _write_contract(
        tmp_path,
        """
        name: node_x
        published_events:
          - topic: "onex.evt.platform.created.v1"
            event_type: "Created"
            topic_config:
              partitions: 1
              replication_factor: 1
              kafka_config:
                retention.ms: "604800000"
                cleanup.policy: "delete"
        """,
    )

    from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

    provisioner = TopicProvisioner(contracts_root=root)
    spec = next(
        s
        for s in provisioner._topic_specs
        if s.suffix == "onex.evt.platform.created.v1"
    )
    new_topic = NewTopic(
        name=spec.suffix,
        num_partitions=spec.partitions,
        replication_factor=spec.replication_factor,
        topic_configs=dict(spec.kafka_config) if spec.kafka_config else {},
    )
    assert new_topic.num_partitions == 1
    assert new_topic.replication_factor == 1
    assert new_topic.topic_configs == {
        "retention.ms": "604800000",
        "cleanup.policy": "delete",
    }


@pytest.mark.unit
def test_merge_sources_keeps_declared_config(tmp_path: Path) -> None:
    """A topic declared in two contracts (one with config) keeps the config."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "contract.yaml").write_text(
        textwrap.dedent(
            """
            name: node_a
            event_bus:
              subscribe_topics:
                - "onex.evt.platform.shared.v1"
            """
        )
    )
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "contract.yaml").write_text(
        textwrap.dedent(
            """
            name: node_b
            published_events:
              - topic: "onex.evt.platform.shared.v1"
                event_type: "Shared"
                topic_config:
                  partitions: 1
                  replication_factor: 1
            """
        )
    )
    entries = {e.topic: e for e in ContractTopicExtractor().extract(tmp_path)}
    shared = entries["onex.evt.platform.shared.v1"]
    assert shared.partitions == 1
    assert shared.replication_factor == 1
    assert len(shared.source_contracts) == 2
