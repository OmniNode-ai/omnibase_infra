# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""OMN-21053 -- one reconcile command must reach update-plan-emitted over the bus.

THE LIVE FINDING (dev lane, 2026-10-12T12:15:31Z). One
``ModelManualReconcileCommand`` published to the reconcile command topic stopped at
the first hop: the change detector's auto-wired ``HandlerManualTrigger`` exposed no
``handle()`` / ``handle_async()`` entrypoint, so the runtime answered with a
``ModelBoundaryFailureTerminal`` on the change-detected topic and replayed the
command five times. All four nodes had live, Stable consumer groups; every
downstream topic stayed empty. The only test of the pipeline,
``tests/integration/test_artifact_reconciliation_pipeline.py``, calls each handler's
private helper directly and so passed throughout.

WHY THIS TEST IS AT THE REAL SEAM. It loads the four REAL ``contract.yaml`` files,
runs the kernel's own ``wire_from_manifest`` over a real ``MessageDispatchEngine`` and
a real ``EventBusInmemory``, publishes the command envelope the way the lane does, and
asserts on what reaches the declared output topics. A handler that is unreachable
through the runtime's dispatch entrypoint rule, or whose output model is not the next
hop's input model, fails here and nowhere else.

BLAST RADIUS. The command is a MANUAL reconcile: ``manual_plan_request`` is not a PR
trigger, so no PR comment may be requested -- the pr-comment-posted topic must stay
empty. No network client is constructed on this path.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_NODES_DIR = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"

_DETECTOR = "node_artifact_change_detector_effect"
_ANALYZER = "node_impact_analyzer_compute"
_REDUCER = "node_update_plan_reducer"
_ORCHESTRATOR = "node_artifact_reconciliation_orchestrator"
_CHAIN = (_DETECTOR, _ANALYZER, _REDUCER, _ORCHESTRATOR)

#: A repo the packaged artifact registry does not name: the synthetic artifact set of
#: the lab probe, so the run touches no real artifact.
_SYNTHETIC_REPO = "artrecon_synthetic_1d5b"
_SYNTHETIC_FILE = "synthetic/artrecon_1d5b/contract.yaml"


def _contract_doc(node: str) -> dict[str, object]:
    loaded = yaml.safe_load((_NODES_DIR / node / "contract.yaml").read_text())
    assert isinstance(loaded, dict)
    return loaded


def _topics(node: str, key: str) -> tuple[str, ...]:
    event_bus = _contract_doc(node)["event_bus"]
    assert isinstance(event_bus, dict)
    topics = event_bus[key]
    assert isinstance(topics, list)
    return tuple(str(topic) for topic in topics)


def _single(topics: tuple[str, ...], suffix: str) -> str:
    matches = [topic for topic in topics if topic.endswith(suffix)]
    assert len(matches) == 1, (suffix, topics)
    return matches[0]


def _envelope(
    payload: dict[str, object], correlation_id: UUID, event_type: str
) -> bytes:
    envelope = ModelEventEnvelope[object](
        payload=payload, correlation_id=correlation_id, event_type=event_type
    )
    return envelope.model_dump_json().encode("utf-8")


class _Chain:
    """The four real contracts wired to one in-memory bus, with every hop observed."""

    def __init__(self) -> None:
        self.command_topic = _single(
            _topics(_DETECTOR, "subscribe_topics"), "reconcile.v1"
        )
        self.webhook_topic = _single(
            _topics(_DETECTOR, "subscribe_topics"), "pr-webhook.v1"
        )
        self.emitted_topic = _single(
            _topics(_ORCHESTRATOR, "publish_topics"), "update-plan-emitted.v1"
        )
        self.comment_topic = _single(
            _topics(_ORCHESTRATOR, "publish_topics"), "pr-comment-posted.v1"
        )
        self.seen: dict[str, list[ModelEventEnvelope[object]]] = {
            _single(_topics(_DETECTOR, "publish_topics"), "change-detected.v1"): [],
            _single(_topics(_ANALYZER, "publish_topics"), "impact-analyzed.v1"): [],
            _single(_topics(_REDUCER, "publish_topics"), "update-plan-created.v1"): [],
            self.emitted_topic: [],
            self.comment_topic: [],
        }

    def counts(self) -> str:
        return ", ".join(f"{t}={len(v)}" for t, v in sorted(self.seen.items()))

    def failure_reasons(self) -> list[str]:
        return [
            str(envelope.payload["failure_reason"])
            for envelopes in self.seen.values()
            for envelope in envelopes
            if isinstance(envelope.payload, dict)
            and "failure_reason" in envelope.payload
        ]


async def _run_chain(
    chain: _Chain,
    topic: str,
    wire: bytes,
    settled: Callable[[], bool],
) -> None:
    """Wire the four contracts, publish ``wire`` to ``topic``, wait until ``settled``."""
    manifest = discover_contracts_from_paths(
        [_NODES_DIR / node / "contract.yaml" for node in _CHAIN]
    )
    # Positive control: all four real contracts were discovered, none refused.
    assert manifest.errors == ()
    assert {contract.name for contract in manifest.contracts} == set(_CHAIN)

    bus = EventBusInmemory(environment="test", group="omn-21053-chain")
    await bus.start()
    try:

        def collector(
            topic_name: str,
        ) -> Callable[[ModelEventMessage], Awaitable[None]]:
            async def collect(message: ModelEventMessage) -> None:
                chain.seen[topic_name].append(
                    ModelEventEnvelope[object].model_validate_json(message.value)
                )

            return collect

        for topic_name in sorted(chain.seen):
            await bus.subscribe(
                topic_name,
                group_id=f"omn-21053-collector-{topic_name}",
                on_message=collector(topic_name),
            )

        engine = MessageDispatchEngine()
        await wire_from_manifest(manifest, engine, event_bus=bus, environment="local")
        engine.freeze()

        await bus.publish(topic, None, wire, None)

        async def until_settled() -> None:
            while not settled():
                await asyncio.sleep(0.05)

        try:
            await asyncio.wait_for(until_settled(), timeout=10)
        except TimeoutError:
            pytest.fail(
                f"the command never settled; per-topic counts: {chain.counts()}; "
                f"failure terminals: {chain.failure_reasons()}"
            )
        # Let any extra (wrong) publish land before the caller asserts "exactly".
        await asyncio.sleep(0.5)
    finally:
        await bus.close()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("wire_shape", ["envelope", "bare"])
async def test_reconcile_command_reaches_update_plan_emitted_and_posts_no_comment(
    wire_shape: str,
) -> None:
    """``bare`` is what the ``artifact-reconcile`` CLI publishes: the command alone."""
    chain = _Chain()
    correlation_id = uuid4()
    command = {
        "command_id": str(correlation_id),
        "source_repo": _SYNTHETIC_REPO,
        "changed_files": [_SYNTHETIC_FILE],
        "actor": "artrecon-1d5b",
        "reason": "OMN-21053 chain test: synthetic artifact set, no PR",
    }
    wire = (
        _envelope(command, correlation_id, "artifact.reconcile")
        if wire_shape == "envelope"
        else json.dumps(command).encode("utf-8")
    )

    with patch("httpx.AsyncClient") as github:
        await _run_chain(
            chain,
            chain.command_topic,
            wire,
            lambda: bool(chain.seen[chain.emitted_topic]),
        )

    emitted = chain.seen[chain.emitted_topic]
    assert len(emitted) == 1, f"expected exactly one emitted event: {chain.counts()}"
    if wire_shape == "envelope":
        assert emitted[0].correlation_id == correlation_id
    payload = emitted[0].payload
    assert isinstance(payload, dict)
    plan = yaml.safe_load(str(payload["yaml_payload"]))
    assert plan["trigger_type"] == "manual_plan_request"
    assert plan["source_entity_ref"] == f"manual/{_SYNTHETIC_REPO}"
    assert plan["tasks"] == []
    # The manual path is not PR-triggered: nothing asks for a PR comment, and no
    # GitHub client was so much as constructed.
    assert chain.seen[chain.comment_topic] == []
    github.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("wire_shape", ["envelope", "bare"])
async def test_pr_webhook_reaches_update_plan_emitted_and_pr_comment_posted(
    wire_shape: str,
) -> None:
    """Positive control for the comment topic: a PR trigger does post (GitHub mocked).

    Without it, the empty ``pr-comment-posted`` assertion above could pass for a
    chain that never reaches the comment handler at all.
    """
    chain = _Chain()
    correlation_id = uuid4()
    webhook = {
        "action": "opened",
        "repo": "OmniNode-ai/omnibase_infra",
        "pr_number": 123,
        "head_ref": "omn-21053-chain",
        "head_sha": "0" * 40,
        "changed_files": ["src/omnibase_infra/nodes/node_x/contract.yaml"],
        "ticket_ids": [],
        "actor": "artrecon-1d5b",
        "merged": False,
    }

    no_comments = MagicMock()
    no_comments.raise_for_status = MagicMock()
    no_comments.json = MagicMock(return_value=[])
    created = MagicMock()
    created.raise_for_status = MagicMock()
    created.json = MagicMock(return_value={"id": 42})
    client = AsyncMock()
    client.get = AsyncMock(return_value=no_comments)
    client.post = AsyncMock(return_value=created)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=client):
        await _run_chain(
            chain,
            chain.webhook_topic,
            (
                _envelope(webhook, correlation_id, "github.pr-webhook")
                if wire_shape == "envelope"
                else json.dumps(webhook).encode("utf-8")
            ),
            lambda: bool(chain.seen[chain.emitted_topic])
            and bool(chain.seen[chain.comment_topic]),
        )

    assert len(chain.seen[chain.emitted_topic]) == 1, chain.counts()
    assert len(chain.seen[chain.comment_topic]) == 1, chain.counts()
    comment = chain.seen[chain.comment_topic][0].payload
    assert isinstance(comment, dict)
    assert comment["posted"] is True
    assert comment["comment_id"] == 42
    client.post.assert_awaited_once()
