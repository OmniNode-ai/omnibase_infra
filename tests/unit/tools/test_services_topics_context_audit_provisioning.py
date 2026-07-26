# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Provisioning coverage for the omniclaude context-audit event topics.

Regression tests for OMN-14842 — the defect class OMN-7114 fixed for four
other topics. The ``ContextAuditConsumer``
(``omnibase_infra.services.observability.context_audit``) subscribes to five
``onex.evt.omniclaude.audit-*.v1`` topics that are produced by the omniclaude
context audit pipeline, which does NOT run on the runtime lane. The broker
auto-creates topics only on PRODUCE, never on subscribe, and since OMN-13238
the contract-first ``TopicProvisioner`` sources topics exclusively from
contract YAML + ``topics.yaml`` manifests (no fallback to the Python registry
in ``platform_topic_suffixes.py`` where these suffixes are declared).

Three of the five topics (audit-return-bounded, audit-context-budget-exceeded,
audit-compression-triggered) were therefore never created on a fresh broker,
and the consumer crash-looped on "topic not found in cluster metadata"
(observed live on the dev lane, OMN-14842). The fix registers all five
consumer-subscribed audit topics in ``services/observability/topics.yaml`` —
scanned by the runtime kernel unconditionally on every lane via the
``services`` manifest root (one-level child scan; OMN-13808 precedent).

The completeness test at the bottom is the one that would have caught this:
it asserts every topic the consumer subscribes to (plus its DLQ topic) is
discoverable on the exact surfaces the ``TopicProvisioner`` scans.

Ticket: OMN-14842 (defect class: OMN-7114; mechanism precedent: OMN-13808)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.services.observability.context_audit.config import (
    ConfigContextAuditConsumer,
)
from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

_AUDIT_EVENT_TOPICS: tuple[str, ...] = (
    "onex.evt.omniclaude.audit-dispatch-validated.v1",
    "onex.evt.omniclaude.audit-scope-violation.v1",
    "onex.evt.omniclaude.audit-context-budget-exceeded.v1",
    "onex.evt.omniclaude.audit-return-bounded.v1",
    "onex.evt.omniclaude.audit-compression-triggered.v1",
)

# This test file is at tests/unit/tools/, so walk up to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src" / "omnibase_infra"
_NODES_ROOT = _SRC_ROOT / "nodes"
_SERVICES_DIR = _SRC_ROOT / "services"
_OBSERVABILITY_TOPICS_YAML = _SERVICES_DIR / "observability" / "topics.yaml"

# The manifest roots service_kernel wires into TopicProvisioner on every lane
# (cli/topics.yaml, services/topics.yaml) — keep in sync with
# runtime/service_kernel.py `_extra_manifest_roots`.
_KERNEL_MANIFEST_ROOTS: tuple[Path, ...] = (
    _SRC_ROOT / "cli",
    _SRC_ROOT / "services",
)


def _consumer_subscribed_topics() -> list[str]:
    """Default topic list of ContextAuditConsumer without touching env/.env.

    Reads the field default_factory directly instead of instantiating the
    BaseSettings class, so the test never depends on ambient environment
    variables or a local .env file.
    """
    factory = ConfigContextAuditConsumer.model_fields["topics"].default_factory
    assert factory is not None, (
        "ConfigContextAuditConsumer.topics lost its default_factory — "
        "update this test to source the consumer's subscribed topics"
    )
    return list(factory())  # type: ignore[call-arg]


def _consumer_dlq_topic() -> str:
    default = ConfigContextAuditConsumer.model_fields["dlq_topic"].default
    assert isinstance(default, str) and default
    return default


@pytest.mark.unit
def test_observability_manifest_lists_all_context_audit_event_topics() -> None:
    """The real services/observability/topics.yaml declares all five audit topics."""
    data = yaml.safe_load(_OBSERVABILITY_TOPICS_YAML.read_text(encoding="utf-8"))
    topics = data["topics"]

    missing = [t for t in _AUDIT_EVENT_TOPICS if t not in topics]
    assert not missing, (
        f"Missing from {_OBSERVABILITY_TOPICS_YAML}: {missing}. The omniclaude "
        "context-audit producer does not run on the runtime lane and the "
        "broker never auto-creates topics on subscribe, so every "
        "ContextAuditConsumer topic must be listed here for the "
        "TopicProvisioner to create it (OMN-14842)."
    )


@pytest.mark.unit
def test_extractor_yields_audit_topics_from_services_manifest() -> None:
    """ContractTopicExtractor extracts the audit topics with correct parse.

    This proves the TopicProvisioner (which uses the same extractor over the
    ``services`` manifest root — the exact root ``service_kernel`` wires in)
    includes the topics in its provisioning target set with the correct
    kind/producer parse, including the one-level child scan that picks up
    ``services/observability/topics.yaml``.
    """
    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_skill_manifests(_SERVICES_DIR)
    by_topic = {e.topic: e for e in entries}

    for topic in _AUDIT_EVENT_TOPICS:
        match = by_topic.get(topic)
        assert match is not None, (
            f"{topic} was not extracted from {_OBSERVABILITY_TOPICS_YAML} "
            f"via the {_SERVICES_DIR} manifest-root scan"
        )
        assert match.kind == "evt"
        assert match.producer == "omniclaude"
        assert match.version == "v1"


@pytest.mark.unit
def test_every_consumer_referenced_topic_is_provisioned() -> None:
    """Completeness: all ContextAuditConsumer topics are provisioner-visible.

    This is the registration-completeness assertion that would have caught
    OMN-14842 (and OMN-7114 before it): every topic the consumer subscribes
    to — plus the DLQ topic it produces to — must be discoverable on the
    surfaces the runtime ``TopicProvisioner`` actually scans on every lane
    (in-repo node contract.yaml files + the cli/services topics.yaml
    manifests). A topic referenced by the consumer but absent from those
    surfaces is never created on a fresh broker, and the consumer crash-loops
    on "topic not found in cluster metadata".
    """
    extractor = ContractTopicExtractor()
    provisioner_visible = {
        e.topic
        for e in extractor.extract_all(
            contracts_root=_NODES_ROOT,
            skill_manifests_roots=[p for p in _KERNEL_MANIFEST_ROOTS if p.is_dir()],
        )
    }

    consumer_referenced = [*_consumer_subscribed_topics(), _consumer_dlq_topic()]
    assert consumer_referenced, "consumer topic list unexpectedly empty"

    unprovisioned = [t for t in consumer_referenced if t not in provisioner_visible]
    assert not unprovisioned, (
        "ContextAuditConsumer references topics the TopicProvisioner never "
        f"creates: {unprovisioned}. Register them in "
        "services/observability/topics.yaml "
        "(consumer-only event topics, OMN-14842) or in an owning node "
        "contract.yaml (like onex.evt.omniclaude.context-audit-dlq.v1 in "
        "node_context_audit_dlq_effect, OMN-7114)."
    )


@pytest.mark.unit
def test_consumer_default_topics_match_pinned_audit_set() -> None:
    """The pinned audit set stays in lockstep with the consumer defaults.

    If a new audit topic is added to (or removed from) the consumer's
    default subscription list, this test forces the pinned tuple above —
    and therefore the manifest assertions — to be updated in the same PR.
    """
    assert set(_consumer_subscribed_topics()) == set(_AUDIT_EVENT_TOPICS)
