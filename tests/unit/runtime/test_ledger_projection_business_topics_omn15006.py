# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Truth-surface widening for the audit-ledger projection (OMN-15006).

``node_ledger_projection_compute`` is TYPE-AGNOSTIC (it persists whatever
event it is handed — see ``HandlerLedgerProjection._extract_ledger_metadata``
and ``node_ledger_write_effect``'s append path); the blindness documented by
OMN-15002 (build_loop silent loss) is NOT a handler bug. It is a subscription
scoping gap: ``contract.yaml`` names only 7 platform-lifecycle topics and was
never widened to the business command/completion/DLQ topics whose silent loss
OMN-15002 surfaced.

This test module is the RED->GREEN spec named in-ticket for OMN-15006 (child
of OMN-14498). It drives the REAL contract + REAL dispatch engine, not a
surrogate:

1. Contract-seam assertions — every ``subscribe_topics`` entry has exactly one
   ``handler_routing`` entry, every entry declares ``event_model`` (the exact
   OMN-15002 starvation asymmetry — a missing ``event_model`` starved
   build_loop to 100% silent loss and must never be reintroduced here), the 18
   new topics are present, and each entry's ``message_category`` matches its
   topic's segment.
2. Behavior leg — wire the real contract into a real, frozen
   ``MessageDispatchEngine`` and drive three synthetic envelopes (one command,
   one cross-repo completion event, one DLQ envelope) through
   ``HandlerLedgerProjection`` end to end, asserting a typed
   ``ledger.append`` intent is emitted with ``topic`` == the topic actually
   consumed (event_ledger.topic is the CONSUMED SOURCE, never the command
   name — see memory ``reference_event_ledger_topic_is_source_not_command``).

RED today (pre-fix, contract_version 1.0.2): only 7 topics/entries exist, so
every "new topic present" assertion fails and every synthetic dispatch in the
behavior leg returns ``NO_DISPATCHER`` with zero emitted intents.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.enums.generated.enum_omnibase_infra_topic import (
    EnumOmnibaseInfraTopic,
)
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODES_ROOT = _REPO_ROOT / "src" / "omnibase_infra" / "nodes"

CONTRACT_PATH = _NODES_ROOT / "node_ledger_projection_compute" / "contract.yaml"
CONTRACT_NAME = "node_ledger_projection_compute"

# ---------------------------------------------------------------------------
# The 18 topics this ticket adds. Symbols are cited where importable
# (EnumOmnibaseInfraTopic, this repo's own generated enum); OCC- and
# omnimarket-owned topics are literals with a citation comment — infra MUST
# NOT import onex_change_control's or omnimarket's enums (infra ↛ market /
# infra ↛ occ layering).
# ---------------------------------------------------------------------------

# build_loop (3)
TOPIC_CMD_BUILD_LOOP_APPEND = EnumOmnibaseInfraTopic.CMD_BUILD_LOOP_APPEND_V1.value
TOPIC_EVT_BUILD_LOOP_APPENDED = EnumOmnibaseInfraTopic.EVT_BUILD_LOOP_APPENDED_V1.value
# omnimarket node_build_loop_orchestrator.contract.yaml `terminal_event` — cited
# as a literal (infra ↛ market import ban forbids importing EnumOmnimarketTopic).
TOPIC_EVT_BUILD_LOOP_ORCHESTRATOR_COMPLETED = (
    "onex.evt.omnimarket.build-loop-orchestrator-completed.v1"
)

# OCC governance (6) — onex_change_control.kafka.topics.GovernanceTopic, cited
# as literals (infra cannot import OCC's enum).
TOPIC_OCC_NIGHTLY_PROMOTION = "onex.evt.occ.nightly-promotion.v1"
TOPIC_OCC_GOVERNANCE_CHECK_COMPLETED = (
    "onex.evt.onex-change-control.governance-check-completed.v1"
)
TOPIC_OCC_CONTRACT_DRIFT_DETECTED = (
    "onex.evt.onex-change-control.contract-drift-detected.v1"
)
TOPIC_OCC_COSMETIC_COMPLIANCE_SCORED = (
    "onex.evt.onex-change-control.cosmetic-compliance-scored.v1"
)
TOPIC_OCC_RUNTIME_DEPLOYMENT_REQUEST = "onex.cmd.omnimarket.redeploy-start.v1"
TOPIC_OCC_RUNTIME_DEPLOYMENT_PROOF = "onex.evt.omnimarket.runtime-deployment-proof.v1"

# DLQ (9) — EnumOmnibaseInfraTopic, all onex.dlq.omnibase-infra.*
TOPIC_DLQ_COMMANDS = EnumOmnibaseInfraTopic.DLQ_COMMANDS_V1.value
TOPIC_DLQ_EVENTS = EnumOmnibaseInfraTopic.DLQ_EVENTS_V1.value
TOPIC_DLQ_INTENTS = EnumOmnibaseInfraTopic.DLQ_INTENTS_V1.value
TOPIC_DLQ_OMNIBASE_INFRA = EnumOmnibaseInfraTopic.DLQ_OMNIBASE_INFRA_V1.value
TOPIC_DLQ_PLATFORM = EnumOmnibaseInfraTopic.DLQ_PLATFORM_V1.value
TOPIC_DLQ_QUARANTINE = EnumOmnibaseInfraTopic.DLQ_QUARANTINE_V1.value
TOPIC_DLQ_ROUTER = EnumOmnibaseInfraTopic.DLQ_ROUTER_V1.value
TOPIC_DLQ_RSD = EnumOmnibaseInfraTopic.DLQ_RSD_V1.value
TOPIC_DLQ_SKILL = EnumOmnibaseInfraTopic.DLQ_SKILL_V1.value

# Expected message_category per new topic. DLQ topics carry no evt/cmd/intent
# segment (`onex.dlq.omnibase-infra.<category>.v1`); handler_wiring's
# `_derive_message_category()` falls back to "event" for any unrecognized
# segment (source: handler_wiring.py `_derive_message_category`), and
# `EnumMessageCategory` (omnibase_core) has only 3 members (event/command/
# intent) — no DLQ-specific value exists. "event" is therefore the
# source-derived value, not a guess.
EXPECTED_NEW_TOPICS: dict[str, str] = {
    TOPIC_CMD_BUILD_LOOP_APPEND: "command",
    TOPIC_EVT_BUILD_LOOP_APPENDED: "event",
    TOPIC_EVT_BUILD_LOOP_ORCHESTRATOR_COMPLETED: "event",
    TOPIC_OCC_NIGHTLY_PROMOTION: "event",
    TOPIC_OCC_GOVERNANCE_CHECK_COMPLETED: "event",
    TOPIC_OCC_CONTRACT_DRIFT_DETECTED: "event",
    TOPIC_OCC_COSMETIC_COMPLIANCE_SCORED: "event",
    TOPIC_OCC_RUNTIME_DEPLOYMENT_REQUEST: "command",
    TOPIC_OCC_RUNTIME_DEPLOYMENT_PROOF: "event",
    TOPIC_DLQ_COMMANDS: "event",
    TOPIC_DLQ_EVENTS: "event",
    TOPIC_DLQ_INTENTS: "event",
    TOPIC_DLQ_OMNIBASE_INFRA: "event",
    TOPIC_DLQ_PLATFORM: "event",
    TOPIC_DLQ_QUARANTINE: "event",
    TOPIC_DLQ_ROUTER: "event",
    TOPIC_DLQ_RSD: "event",
    TOPIC_DLQ_SKILL: "event",
}

assert len(EXPECTED_NEW_TOPICS) == 18, (
    f"expected exactly 18 new topics named in-ticket, got {len(EXPECTED_NEW_TOPICS)}"
)


def _load_raw_contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# 1. Contract-seam assertions
# ---------------------------------------------------------------------------


def test_all_18_business_topics_are_subscribed() -> None:
    """RED today: none of the 18 business/OCC/DLQ topics are in subscribe_topics."""
    raw = _load_raw_contract()
    subscribed = set(raw.get("event_bus", {}).get("subscribe_topics") or [])

    missing = sorted(set(EXPECTED_NEW_TOPICS) - subscribed)
    assert not missing, (
        f"node_ledger_projection_compute must subscribe to all 18 business/OCC/DLQ "
        f"topics named in OMN-15006; missing: {missing}"
    )


def test_every_subscribed_topic_has_exactly_one_handler_routing_entry() -> None:
    """OMN-14594 pattern: one topic_match entry PER topic, never a shared entry."""
    raw = _load_raw_contract()
    subscribed = raw.get("event_bus", {}).get("subscribe_topics") or []
    handlers = raw.get("handler_routing", {}).get("handlers") or []

    topic_counts: dict[str, int] = {}
    for entry in handlers:
        topic = entry.get("topic")
        if topic is None:
            continue
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    for topic in subscribed:
        assert topic_counts.get(topic, 0) == 1, (
            f"topic {topic!r} must have exactly ONE handler_routing entry "
            f"(topic_match, per OMN-14594), got {topic_counts.get(topic, 0)}"
        )


def test_every_handler_routing_entry_declares_event_model() -> None:
    """A missing event_model is the exact OMN-15002 starvation asymmetry.

    Without event_model, the dispatcher registers with no payload_type_matcher
    (OMN-12416) and the category is derived from subscribe_topics[0] for the
    WHOLE contract (handler_wiring.py `_category_str_early` fallback) rather
    than per-entry — silently mis-categorizing any topic that isn't first in
    the list. This must never regress.
    """
    raw = _load_raw_contract()
    handlers = raw.get("handler_routing", {}).get("handlers") or []

    missing_event_model = [
        entry.get("topic", "<no-topic>")
        for entry in handlers
        if not entry.get("event_model")
    ]
    assert not missing_event_model, (
        "every handler_routing entry MUST declare event_model — missing on: "
        f"{missing_event_model}"
    )


def test_new_topics_message_category_matches_expected() -> None:
    """Per-entry message_category must match the topic's segment (or the
    documented DLQ fallback)."""
    raw = _load_raw_contract()
    handlers = raw.get("handler_routing", {}).get("handlers") or []
    category_by_topic = {
        entry.get("topic"): entry.get("message_category") for entry in handlers
    }

    mismatches = {
        topic: (category_by_topic.get(topic), expected)
        for topic, expected in EXPECTED_NEW_TOPICS.items()
        if category_by_topic.get(topic) != expected
    }
    assert not mismatches, (
        f"message_category mismatch (actual, expected) per topic: {mismatches}"
    )


# ---------------------------------------------------------------------------
# 2. Behavior leg — drive the real dispatch engine
# ---------------------------------------------------------------------------


class _StubResultApplier:
    async def apply(self, *args: object, **kwargs: object) -> None:
        return None


async def _wire_and_freeze() -> MessageDispatchEngine:
    discovered = discover_contracts_from_paths([CONTRACT_PATH])
    contracts = getattr(discovered, "contracts", discovered)
    manifest = ModelAutoWiringManifest(contracts=tuple(contracts))
    engine = MessageDispatchEngine()

    from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest

    await wire_from_manifest(
        manifest,
        engine,
        event_bus=None,
        environment="dev",
        container=ModelONEXContainer(),
        subscribe_immediately=False,
        result_appliers_by_contract={CONTRACT_NAME: _StubResultApplier()},
    )
    engine.freeze()
    return engine


def _raw_wrapper_envelope(topic: str, *, event_type: str) -> ModelEventEnvelope[object]:
    """Build the raw-event-projection envelope shape this node's own subscribe
    callback constructs (envelope.payload = the dumped ModelEventMessage)."""
    message = ModelEventMessage(
        topic=topic,
        value=b'{"synthetic": true}',
        headers=ModelEventHeaders(
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            source="test-producer-omn15006",
            event_type=event_type,
        ),
        partition=1,
        offset="7",
    )
    return ModelEventEnvelope[object](
        payload=message.model_dump(mode="json"),
        correlation_id=uuid4(),
        event_type=event_type,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("topic", "event_type"),
    [
        pytest.param(
            TOPIC_CMD_BUILD_LOOP_APPEND,
            "omnibase-infra.build-loop-append",
            id="build_loop_command",
        ),
        pytest.param(
            TOPIC_EVT_BUILD_LOOP_ORCHESTRATOR_COMPLETED,
            "omnimarket.build-loop-orchestrator-completed",
            id="build_loop_orchestrator_completion",
        ),
        pytest.param(
            # event_type alias derived from the topic by
            # `_derive_event_type_alias_from_topic` (handler_wiring.py):
            # f"{parts[2]}.{parts[3]}" for a standard 5-segment topic, i.e.
            # "omnibase-infra.events" for "onex.dlq.omnibase-infra.events.v1".
            TOPIC_DLQ_EVENTS,
            "omnibase-infra.events",
            id="dlq_events",
        ),
    ],
)
async def test_synthetic_envelope_on_new_topic_produces_ledger_append_intent(
    topic: str, event_type: str
) -> None:
    """RED today: these topics have no dispatcher -> NO_DISPATCHER, 0 intents.

    GREEN post-fix: the dispatch engine routes the synthetic envelope to
    HandlerLedgerProjection, which emits exactly one typed ``ledger.append``
    intent whose ``topic`` equals the topic actually CONSUMED (never the
    command name — event_ledger.topic is the consumed source).
    """
    engine = await _wire_and_freeze()

    result = await engine.dispatch(
        topic, _raw_wrapper_envelope(topic, event_type=event_type)
    )

    assert result.status == EnumDispatchStatus.SUCCESS, (
        f"topic {topic!r} must dispatch to HandlerLedgerProjection "
        f"(status={result.status!r}, error={result.error_message!r})"
    )
    assert len(result.output_intents) == 1, (
        f"expected exactly 1 ledger.append intent for topic {topic!r}, "
        f"got {len(result.output_intents)}"
    )
    intent = result.output_intents[0]
    assert isinstance(intent, ModelIntent)
    assert intent.payload.intent_type == "ledger.append"
    assert intent.payload.topic == topic, (
        "event_ledger.topic must be the CONSUMED SOURCE topic, never the "
        f"command name: got {intent.payload.topic!r}, expected {topic!r}"
    )
