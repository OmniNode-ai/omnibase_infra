# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for _topics_for_handler_entry topic assignment [OMN-12848].

Regression: a single-handler contract that declares an ``event_model`` and
subscribes to MORE THAN ONE topic (e.g. ``node_generation_consumer`` subscribing
to both ``node-generation-requested`` and ``node-deploy``) previously fell
through to ``return ()`` — registering ZERO dispatch routes. Commands were
consumed then routed to DLQ with "No dispatcher found for category 'command'".

The fix: a SOLE handler entry unambiguously owns every subscribe topic. The
ambiguity guard (``return ()``) only applies when MULTIPLE handler entries
compete for the same topics without per-handler ``event_type``.
"""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _topics_for_handler_entry,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)


def _entry(
    *,
    event_model_name: str | None = "ModelFoo",
    event_type: str | None = None,
) -> ModelHandlerRoutingEntry:
    kwargs: dict[str, object] = {
        "handler": ModelHandlerRef(name="HandlerFoo", module="fake.module"),
        "operation": None,
    }
    if event_model_name is not None:
        kwargs["event_model"] = ModelHandlerRef(
            name=event_model_name, module="fake.models"
        )
    if event_type is not None:
        kwargs["event_type"] = event_type
    return ModelHandlerRoutingEntry(**kwargs)


def _contract(
    *,
    entries: tuple[ModelHandlerRoutingEntry, ...],
    subscribe_topics: tuple[str, ...],
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_local",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_local",
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe_topics,
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=entries,
        ),
    )


def test_sole_handler_event_model_multi_topic_assigns_all_topics() -> None:
    """OMN-12848: the regression case — must assign ALL topics, not empty."""
    topics = (
        "onex.cmd.omnimarket.node-generation-requested.v1",
        "onex.cmd.omnimarket.node-deploy.v1",
    )
    entry = _entry(event_model_name="ModelNodeGenerationRequest", event_type=None)
    contract = _contract(entries=(entry,), subscribe_topics=topics)

    assigned = _topics_for_handler_entry(contract, entry)

    assert assigned == topics, (
        "A sole handler entry must own every subscribe topic so a dispatch route "
        "is registered; empty assignment causes node-generation-requested to DLQ."
    )


def test_sole_handler_event_model_single_topic_unchanged() -> None:
    """Single-topic single-handler still resolves to its one topic."""
    topics = ("onex.cmd.omnimarket.node-generation-requested.v1",)
    entry = _entry(event_model_name="ModelNodeGenerationRequest")
    contract = _contract(entries=(entry,), subscribe_topics=topics)

    assert _topics_for_handler_entry(contract, entry) == topics


def test_multi_handler_event_model_no_event_type_stays_ambiguous() -> None:
    """Multiple competing handlers without event_type keep the ambiguity guard."""
    topics = (
        "onex.cmd.platform.a.v1",
        "onex.cmd.platform.b.v1",
    )
    e1 = _entry(event_model_name="ModelA", event_type=None)
    e2 = _entry(event_model_name="ModelB", event_type=None)
    contract = _contract(entries=(e1, e2), subscribe_topics=topics)

    # Neither competing entry can deterministically claim a topic.
    assert _topics_for_handler_entry(contract, e1) == ()
    assert _topics_for_handler_entry(contract, e2) == ()


def test_per_handler_event_type_still_matches_specific_topic() -> None:
    """A per-handler event_type continues to select only its matching topic."""
    topics = (
        "onex.cmd.omnimarket.node-generation-requested.v1",
        "onex.cmd.omnimarket.node-deploy.v1",
    )
    e1 = _entry(
        event_model_name="ModelGen",
        event_type="omnimarket.node-generation-requested",
    )
    e2 = _entry(event_model_name="ModelDeploy", event_type="omnimarket.node-deploy")
    contract = _contract(entries=(e1, e2), subscribe_topics=topics)

    assert _topics_for_handler_entry(contract, e1) == (
        "onex.cmd.omnimarket.node-generation-requested.v1",
    )
    assert _topics_for_handler_entry(contract, e2) == (
        "onex.cmd.omnimarket.node-deploy.v1",
    )


def test_no_event_model_returns_all_topics() -> None:
    """An entry without an event_model keeps the legacy all-topics behavior."""
    topics = ("onex.cmd.platform.a.v1", "onex.cmd.platform.b.v1")
    entry = _entry(event_model_name=None)
    contract = _contract(entries=(entry,), subscribe_topics=topics)

    assert _topics_for_handler_entry(contract, entry) == topics
