# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The generic LLM inference response has its own topic (OMN-17105).

``onex.evt.omnibase-infra.inference-response.v1`` belongs to the delegation
pipeline: omnimarket's ``node_llm_delegation_call_effect`` publishes
``ModelInferenceResponseData`` there and the delegation orchestrator subscribes.
``node_llm_inference_effect`` published ``ModelLlmInferenceResponse`` on the same
name, so two producers put two incompatible schemas on one topic. The generic
response now lives on a distinct topic and no infra contract publishes the
delegation topic.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_NODES_DIR = Path(__file__).resolve().parents[4] / "src/omnibase_infra/nodes"
_CONTRACT = _NODES_DIR / "node_llm_inference_effect/contract.yaml"

DELEGATION_RESPONSE_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
GENERIC_RESPONSE_TOPIC = "onex.evt.omnibase-infra.llm-inference-response.v1"


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _published_topics(contract: dict[str, Any]) -> set[str]:
    declared = set(contract.get("event_bus", {}).get("publish_topics", []))
    declared |= {e["topic"] for e in contract.get("published_events", [])}
    return declared


class TestGenericResponseTopic:
    def test_generic_response_is_published_on_its_own_topic(self) -> None:
        contract = _load(_CONTRACT)

        assert GENERIC_RESPONSE_TOPIC in contract["event_bus"]["publish_topics"]
        (event,) = [
            e
            for e in contract["published_events"]
            if e["topic"] == GENERIC_RESPONSE_TOPIC
        ]
        assert event["event_type"] == "LlmInferenceResponse"

    def test_generic_inference_effect_does_not_publish_the_delegation_topic(
        self,
    ) -> None:
        assert DELEGATION_RESPONSE_TOPIC not in _published_topics(_load(_CONTRACT))

    def test_golden_path_names_the_generic_response_topic(self) -> None:
        golden = "\n".join(_load(_CONTRACT)["golden_path"])

        assert GENERIC_RESPONSE_TOPIC in golden
        assert DELEGATION_RESPONSE_TOPIC not in golden


def _topics_with_competing_owners(
    contracts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, set[str]]]:
    """Topics published by more than one contract under different event types."""
    owners: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for name, contract in contracts.items():
        for event in contract.get("published_events", []):
            if event.get("event_type"):
                owners[event["topic"]][name].add(event["event_type"])
    return {
        topic: {name: set(types) for name, types in by_owner.items()}
        for topic, by_owner in owners.items()
        if len(by_owner) > 1
        and len({t for types in by_owner.values() for t in types}) > 1
    }


class TestOneSchemaPerPublishedTopic:
    def test_no_infra_contract_publishes_the_delegation_response_topic(self) -> None:
        publishers = [
            path.parent.name
            for path in sorted(_NODES_DIR.glob("*/contract.yaml"))
            if DELEGATION_RESPONSE_TOPIC in _published_topics(_load(path))
        ]

        assert publishers == []

    def test_no_topic_is_published_by_two_contracts_under_different_event_types(
        self,
    ) -> None:
        contracts = {
            p.parent.name: _load(p) for p in sorted(_NODES_DIR.glob("*/contract.yaml"))
        }

        assert _topics_with_competing_owners(contracts) == {}

    def test_audit_reports_two_owners_with_different_event_types(self) -> None:
        # Positive control: the pre-OMN-17105 collision must be reported.
        collision = {
            owner: {
                "published_events": [
                    {"topic": DELEGATION_RESPONSE_TOPIC, "event_type": event_type}
                ]
            }
            for owner, event_type in (
                ("node_llm_delegation_call_effect", "InferenceResponse"),
                ("node_llm_inference_effect", "LlmInferenceResponse"),
            )
        }

        assert _topics_with_competing_owners(collision) == {
            DELEGATION_RESPONSE_TOPIC: {
                "node_llm_delegation_call_effect": {"InferenceResponse"},
                "node_llm_inference_effect": {"LlmInferenceResponse"},
            }
        }


class TestDelegationTopicIsCrossRepoOwned:
    def test_wiring_health_allowlist_names_the_omnimarket_owner(self) -> None:
        from scripts.check_subscribe_wiring_health import (
            _EXTERNAL_PUBLISHER_ALLOWLIST,
        )

        reason = _EXTERNAL_PUBLISHER_ALLOWLIST[DELEGATION_RESPONSE_TOPIC]

        assert "omnimarket node_llm_delegation_call_effect" in reason
        assert GENERIC_RESPONSE_TOPIC not in _EXTERNAL_PUBLISHER_ALLOWLIST

    def test_wiring_health_reports_no_dead_letter_for_the_delegation_topic(
        self,
    ) -> None:
        from scripts.check_subscribe_wiring_health import check_wiring_health

        errors, _ = check_wiring_health([_NODES_DIR])

        assert [e for e in errors if DELEGATION_RESPONSE_TOPIC in e] == []
