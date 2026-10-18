# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""PublisherContractCorpus reads every publisher declaration shape the topic graph reads.

The fixture used to know four shapes (``event_bus.publish_topics``,
``published_events[].topic``, ``runtime_dispatch.command_topic`` and
``runtime_dispatch.terminal_events``); the contract topic graph reads the union of a
dozen. A topic published through a shape the fixture did not know raised
``PublisherTopicError`` for an edge the graph does have.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.testing.publisher_contract_fixture import (
    PublisherContractCorpus,
    PublisherTopicError,
)
from omnibase_infra.validators.contract_topic_graph import parse_contract

pytestmark = pytest.mark.unit

_SUCCESS = "onex.evt.demo.thing-completed.v1"
_FAILURE = "onex.evt.demo.thing-failed.v1"
_PRODUCED = "onex.evt.demo.thing-produced.v1"
_PUBLICATION = "onex.evt.demo.thing-publication.v1"
_TOPICS_PRODUCES = "onex.evt.demo.thing-listed.v1"
_PLAIN = "onex.evt.demo.thing-plain.v1"
_COMMAND = "onex.cmd.demo.thing-start.v1"

_CONTRACT = f"""\
name: node_demo_publisher
event_bus:
  publish_topics:
    - {_PLAIN}
  publish:
    success_topic: {_SUCCESS}
    failure_topic: {_FAILURE}
produced_events:
  - topic: {_PRODUCED}
publications:
  - {_PUBLICATION}
topics:
  produces:
    - {_TOPICS_PRODUCES}
runtime_dispatch:
  command_topic: {_COMMAND}
"""


def _write(tmp_path: Path, text: str) -> Path:
    node_dir = tmp_path / "nodes" / "node_demo_publisher"
    node_dir.mkdir(parents=True)
    path = node_dir / "contract.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "topic",
    [_PLAIN, _SUCCESS, _FAILURE, _PRODUCED, _PUBLICATION, _TOPICS_PRODUCES, _COMMAND],
)
def test_every_declared_publisher_shape_resolves(tmp_path: Path, topic: str) -> None:
    corpus = PublisherContractCorpus.from_paths([_write(tmp_path, _CONTRACT)])
    assert corpus.publishers_of(topic) == ("node_demo_publisher",)
    assert corpus.event_type_for(topic).startswith("demo.")


def test_corpus_matches_the_topic_graph_parser(tmp_path: Path) -> None:
    """The fixture and the graph agree on what a contract publishes."""
    path = _write(tmp_path, _CONTRACT)
    node = parse_contract(path, "demo")
    assert node is not None
    corpus = PublisherContractCorpus.from_paths([path])
    published_by_graph = set(node.publish_topics)
    assert published_by_graph, "positive control: the graph parser read a publisher"
    assert published_by_graph <= set(corpus.publishers)


def test_undeclared_topic_still_refused(tmp_path: Path) -> None:
    corpus = PublisherContractCorpus.from_paths([_write(tmp_path, _CONTRACT)])
    with pytest.raises(PublisherTopicError):
        corpus.event_type_for("onex.evt.demo.nobody-publishes-this.v1")


def test_invalid_yaml_contract_is_skipped(tmp_path: Path) -> None:
    good = _write(tmp_path, _CONTRACT)
    bad_dir = tmp_path / "nodes" / "node_broken"
    bad_dir.mkdir(parents=True)
    bad = bad_dir / "contract.yaml"
    bad.write_text("name: [unclosed\n", encoding="utf-8")
    corpus = PublisherContractCorpus.from_paths([bad, good])
    assert corpus.publishers_of(_SUCCESS) == ("node_demo_publisher",)
