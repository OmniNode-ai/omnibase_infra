# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contract topic extractor keeps ``tenant-<slug>.<canonical topic>`` (OMN-25548).

The proof topic of the cloud ingest first slice carries the proof tenant's prefix.
The extractor required the first segment to be ``onex``, so the name was dropped
from ``contract.yaml`` and ``topics.yaml`` declarations and never reached the
create-set. Three unprefixed controls pass beside it; every malformed prefix is
still skipped with a warning.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor


def write_contract(tmp_path: Path, name: str, content: str) -> Path:
    node_dir = tmp_path / name
    node_dir.mkdir(parents=True, exist_ok=True)
    contract = node_dir / "contract.yaml"
    contract.write_text(textwrap.dedent(content), encoding="utf-8")
    return contract


_PROOF_TOPIC = "tenant-cli-run-h051.onex.evt.omnimarket.pool-usage-verdict-evaluated.v1"
_CONTROL_TOPICS = (
    "onex.evt.omnimarket.pool-usage-verdict-evaluated.v1",
    "onex.cmd.omnimarket.pool-usage-verdict-requested.v1",
    "onex.snapshot.projection.delegation.quality-gate.v1",
)
_MALFORMED = (
    "tenant-Cli-run.onex.evt.omnimarket.x-done.v1",  # uppercase slug
    f"tenant-{'a' * 33}.onex.evt.omnimarket.x-done.v1",  # slug over 32 chars
    "tenant--a.onex.evt.omnimarket.x-done.v1",  # leading hyphen
    "tenant-a-.onex.evt.omnimarket.x-done.v1",  # trailing hyphen
    "tenant-.onex.evt.omnimarket.x-done.v1",  # empty slug
    "tenant-cli-run-h051.omnimarket.x-done.v1",  # no onex canonical suffix
    "tenant-cli-run-h051.onex.bogus.omnimarket.x-done.v1",  # bad canonical kind
    "tenant-cli-run-h051.",  # nothing after the prefix
    "tenant-a.tenant-b.onex.evt.omnimarket.x-done.v1",  # nested prefix
    "Tenant-a.onex.evt.omnimarket.x-done.v1",  # uppercase marker
)


@pytest.mark.unit
def test_tenant_prefixed_topic_is_kept_in_contract_yaml(tmp_path: Path) -> None:
    """The proof topic and three unprefixed controls all survive contract.yaml."""
    topics = "\n".join(f'    - "{t}"' for t in (*_CONTROL_TOPICS, _PROOF_TOPIC))
    topics = topics.replace("\n", "\n        ").lstrip()
    write_contract(
        tmp_path,
        "node_tenant",
        f"""
        event_bus:
          publish_topics:
            {topics}
        """,
    )
    results = {e.topic: e for e in ContractTopicExtractor().extract(tmp_path)}

    assert set(results) == {*_CONTROL_TOPICS, _PROOF_TOPIC}
    proof = results[_PROOF_TOPIC]
    assert proof.tenant_slug == "cli-run-h051"
    assert (proof.kind, proof.producer, proof.version) == ("evt", "omnimarket", "v1")
    assert proof.event_name == "pool-usage-verdict-evaluated"
    assert all(results[t].tenant_slug == "" for t in _CONTROL_TOPICS)


@pytest.mark.unit
def test_tenant_prefixed_topic_keeps_its_topic_config(tmp_path: Path) -> None:
    """A structured declaration carries topic_config for the tenant topic."""
    write_contract(
        tmp_path,
        "node_tenant_cfg",
        f"""
        consumed_events:
          - topic: "{_PROOF_TOPIC}"
            topic_config:
              partitions: 1
              kafka_config:
                max.message.bytes: 1048576
        """,
    )
    (entry,) = ContractTopicExtractor().extract(tmp_path)

    assert entry.topic == _PROOF_TOPIC
    assert entry.partitions == 1
    assert dict(entry.kafka_config or {}) == {"max.message.bytes": "1048576"}


@pytest.mark.unit
@pytest.mark.parametrize("bad_topic", _MALFORMED)
def test_malformed_tenant_prefixed_topic_is_excluded(
    tmp_path: Path, bad_topic: str, capsys: pytest.CaptureFixture[str]
) -> None:
    write_contract(
        tmp_path,
        "node_tenant_bad",
        f"""
        event_bus:
          publish_topics:
            - "{bad_topic}"
        """,
    )

    assert ContractTopicExtractor().extract(tmp_path) == []
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.unit
def test_tenant_slug_of_exactly_32_and_1_characters_is_accepted(
    tmp_path: Path,
) -> None:
    longest = "a" + "b" * 30 + "c"
    assert len(longest) == 32
    write_contract(
        tmp_path,
        "node_tenant_edges",
        f"""
        event_bus:
          publish_topics:
            - "tenant-{longest}.onex.evt.omnimarket.x-done.v1"
            - "tenant-z.onex.evt.omnimarket.x-done.v1"
        """,
    )

    slugs = sorted(e.tenant_slug for e in ContractTopicExtractor().extract(tmp_path))

    assert slugs == [longest, "z"]


def _write_topics_yaml(tmp_path: Path, topics: tuple[str, ...]) -> Path:
    skill = tmp_path / "skills" / "proof"
    skill.mkdir(parents=True)
    listed = "\n".join(f"  - {t}" for t in topics)
    (skill / "topics.yaml").write_text(f"topics:\n{listed}\n", encoding="utf-8")
    return tmp_path / "skills"


@pytest.mark.unit
def test_tenant_prefixed_topic_is_kept_in_topics_yaml(tmp_path: Path) -> None:
    root = _write_topics_yaml(tmp_path, (*_CONTROL_TOPICS, _PROOF_TOPIC))

    entries = {
        e.topic: e for e in ContractTopicExtractor().extract_from_skill_manifests(root)
    }

    assert set(entries) == {*_CONTROL_TOPICS, _PROOF_TOPIC}
    assert entries[_PROOF_TOPIC].tenant_slug == "cli-run-h051"
    assert all(entries[t].tenant_slug == "" for t in _CONTROL_TOPICS)


@pytest.mark.unit
@pytest.mark.parametrize("bad_topic", _MALFORMED)
def test_malformed_tenant_prefixed_topic_is_skipped_in_topics_yaml(
    tmp_path: Path, bad_topic: str
) -> None:
    root = _write_topics_yaml(tmp_path, (bad_topic,))

    assert ContractTopicExtractor().extract_from_skill_manifests(root) == []
