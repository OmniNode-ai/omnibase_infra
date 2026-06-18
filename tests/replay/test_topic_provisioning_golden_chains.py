# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden-chain replay for the per-contract boot interleave (OMN-13237 W6).

Five deterministic, replayable fixtures replay the per-contract
provision -> ready -> attach boot sequence as an event chain through a pure
reducer (no live broker). Positive chains terminate at ``attached``; negative
chains terminate at ``not_ready`` with the runtime still live (§3.8).

This is the replay evidence for AC-5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.enum_runtime_readiness_state import (
    EnumRuntimeReadinessState,
)
from omnibase_infra.event_bus.enum_topic_readiness_failure_reason import (
    EnumTopicReadinessFailureReason,
)
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_runtime_attach_readiness import (
    ModelRuntimeAttachReadiness,
)
from omnibase_infra.event_bus.model_topic_readiness_failure import (
    ModelTopicReadinessFailure,
)
from omnibase_infra.event_bus.model_topic_set_readiness import (
    ModelTopicSetReadiness,
)

_FIXTURE_DIR = (
    Path(__file__).parents[1] / "fixtures" / "golden_chains" / "topic_provisioning"
)

_POSITIVE_CHAINS = (
    "single_subscribe_attached",
    "multiple_subscribe_attached",
    "publish_and_subscribe_attached",
)
_NEGATIVE_CHAINS = (
    "readiness_failure_not_ready",
    "invalid_config_not_ready",
)
_ALL_CHAINS = _POSITIVE_CHAINS + _NEGATIVE_CHAINS


def _load_chain(chain_id: str) -> dict[str, object]:
    path = _FIXTURE_DIR / f"{chain_id}.json"
    return json.loads(path.read_text())


def _replay_chain(events: list[dict[str, object]]) -> ModelContractAttachResult:
    """Pure reducer over the provisioning/attach chain events.

    delta(state, event) -> new state. Deterministic and side-effect free so the
    same input replays to the same terminal state.
    """
    contract_name = "unknown"
    status = EnumContractAttachStatus.NOT_READY
    attached_topics: list[str] = []
    failure: ModelTopicReadinessFailure | None = None
    liveness = True

    for event in sorted(events, key=lambda e: int(e["sequence"])):
        kind = event["event_type"]
        if "contract" in event:
            contract_name = str(event["contract"])
        if kind == "consumer_attached":
            attached_topics.append(str(event["topic"]))
        elif kind == "contract_marked_attached":
            status = EnumContractAttachStatus.ATTACHED
        elif kind in ("readiness_failed", "provisioning_failed"):
            failure = ModelTopicReadinessFailure(
                topic=str(event["topic"]),
                reason=EnumTopicReadinessFailureReason(str(event["reason"])),
            )
        elif kind == "contract_marked_not_ready":
            status = EnumContractAttachStatus.NOT_READY
        elif kind == "runtime_remains_live":
            liveness = bool(event["liveness"])

    # Liveness invariant: a not-ready contract never flips liveness false.
    assert liveness is True

    readiness = None
    if failure is not None:
        readiness = ModelTopicSetReadiness(
            topics=(failure.topic,),
            status=EnumTopicReadinessStatus.NOT_READY,
            failures=(failure,),
            attempts=1,
        )

    return ModelContractAttachResult(
        contract_name=contract_name,
        status=status,
        topics_subscribed=tuple(attached_topics),
        readiness=readiness,
    )


@pytest.mark.parametrize("chain_id", _ALL_CHAINS)
def test_chain_replays_to_expected_terminal_state(chain_id: str) -> None:
    chain = _load_chain(chain_id)
    events = list(chain["events"])  # type: ignore[arg-type]
    result = _replay_chain(events)

    expected = str(chain["expected_terminal_state"])
    assert result.status.value == expected


@pytest.mark.parametrize("chain_id", _POSITIVE_CHAINS)
def test_positive_chains_terminate_attached(chain_id: str) -> None:
    chain = _load_chain(chain_id)
    result = _replay_chain(list(chain["events"]))  # type: ignore[arg-type]
    assert result.status is EnumContractAttachStatus.ATTACHED
    assert result.topics_subscribed  # at least one consumer attached


@pytest.mark.parametrize("chain_id", _NEGATIVE_CHAINS)
def test_negative_chains_terminate_not_ready_runtime_degraded(
    chain_id: str,
) -> None:
    chain = _load_chain(chain_id)
    result = _replay_chain(list(chain["events"]))  # type: ignore[arg-type]
    assert result.status is EnumContractAttachStatus.NOT_READY
    assert result.topics_subscribed == ()
    assert result.readiness is not None
    expected_reason = str(chain["expected_failure_reason"])
    assert result.readiness.failures[0].reason.value == expected_reason

    # Runtime aggregate: a single not-ready non-core contract is DEGRADED, live.
    readiness = ModelRuntimeAttachReadiness.from_results((result,))
    assert readiness.state is EnumRuntimeReadinessState.DEGRADED


@pytest.mark.parametrize("chain_id", _ALL_CHAINS)
def test_replay_is_deterministic(chain_id: str) -> None:
    """Replaying the same chain twice yields identical terminal state."""
    chain = _load_chain(chain_id)
    events = list(chain["events"])  # type: ignore[arg-type]
    first = _replay_chain(events)
    second = _replay_chain(events)
    assert first == second


def test_all_five_chains_present() -> None:
    found = {p.stem for p in _FIXTURE_DIR.glob("*.json")}
    assert set(_ALL_CHAINS).issubset(found)
    assert len(_ALL_CHAINS) == 5
