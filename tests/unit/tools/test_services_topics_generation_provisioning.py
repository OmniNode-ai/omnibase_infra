# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Provisioning coverage for the omnimarket generation command topic.

The runtime kernel's ``TopicProvisioner`` scans ``services/topics.yaml`` (via
``ContractTopicExtractor.extract_from_skill_manifests``) on every lane, so any
cross-repo topic listed there is explicitly provisioned regardless of which
runtime packages happen to be installed/active.

``onex.cmd.omnimarket.node-generation-requested.v1`` is a consumer-only command
topic: it is declared in omnimarket's ``node_generation_consumer`` contract
(``event_bus.subscribe_topics``) but this broker auto-creates topics only on
PRODUCE, not on subscribe. Its only producer (the projection-API
``/api/generate`` route) does not run on the runtime lane, so without an explicit
provisioning entry the topic stays at 0 partitions and the effects-profile
consumer cannot bind (generation golden chains G-01..G-05 blocked).

These tests pin the manifest entry so the provisioning gap cannot silently
regress.

Ticket: OMN-13808 (sibling of OMN-13771)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

_GENERATION_CMD_TOPIC = "onex.cmd.omnimarket.node-generation-requested.v1"

# services/topics.yaml lives at src/omnibase_infra/services/topics.yaml.
# This test file is at tests/unit/tools/, so walk up to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVICES_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "services"
_SERVICES_TOPICS_YAML = _SERVICES_DIR / "topics.yaml"


@pytest.mark.unit
def test_services_manifest_lists_generation_command_topic() -> None:
    """The real services/topics.yaml declares the generation command topic."""
    data = yaml.safe_load(_SERVICES_TOPICS_YAML.read_text(encoding="utf-8"))
    topics = data["topics"]

    assert _GENERATION_CMD_TOPIC in topics, (
        f"{_GENERATION_CMD_TOPIC} must be listed in {_SERVICES_TOPICS_YAML} so the "
        "TopicProvisioner creates it on every lane; the consumer-only command "
        "topic is never auto-created because it is only subscribed, never "
        "produced, on the runtime lane (OMN-13808)."
    )


@pytest.mark.unit
def test_extractor_yields_generation_command_topic_from_services_manifest() -> None:
    """ContractTopicExtractor extracts the generation topic as a cmd topic.

    This proves the TopicProvisioner (which uses the same extractor over the
    ``services`` manifest root) includes the topic in its provisioning target
    set with the correct kind/producer parse.
    """
    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_skill_manifests(_SERVICES_DIR)

    match = next((e for e in entries if e.topic == _GENERATION_CMD_TOPIC), None)
    assert match is not None, (
        f"{_GENERATION_CMD_TOPIC} was not extracted from {_SERVICES_TOPICS_YAML}"
    )
    assert match.kind == "cmd"
    assert match.producer == "omnimarket"
    assert match.event_name == "node-generation-requested"
    assert match.version == "v1"
