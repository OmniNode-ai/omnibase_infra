# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-backed topic constants for the delegation orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

_CONTRACT_PATH: Final[Path] = Path(__file__).with_name("contract.yaml")


def _load_contract_data() -> dict[str, object]:
    raw = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Delegation contract must be a mapping: {_CONTRACT_PATH}"
        raise RuntimeError(msg)
    return raw


def _subscribe_topics() -> tuple[str, ...]:
    event_bus = _load_contract_data().get("event_bus")
    if not isinstance(event_bus, dict):
        msg = f"Delegation contract is missing event_bus: {_CONTRACT_PATH}"
        raise RuntimeError(msg)

    raw_topics = event_bus.get("subscribe_topics")
    if not isinstance(raw_topics, list):
        msg = f"Delegation contract is missing subscribe_topics: {_CONTRACT_PATH}"
        raise RuntimeError(msg)

    return tuple(topic for topic in raw_topics if isinstance(topic, str))


def _require_unique_subscribe_topic(fragment: str) -> str:
    matches = tuple(topic for topic in _subscribe_topics() if fragment in topic)
    if len(matches) != 1:
        msg = (
            f"Expected one delegation subscribe topic containing {fragment!r}, "
            f"found {len(matches)} in {_CONTRACT_PATH}"
        )
        raise RuntimeError(msg)
    return matches[0]


TOPIC_ID_INVOCATION_COMMAND: Final[str] = _require_unique_subscribe_topic(
    ".invocation."
)
TOPIC_ID_AGENT_TASK_LIFECYCLE: Final[str] = _require_unique_subscribe_topic(
    ".agent-task-lifecycle."
)

__all__ = [
    "TOPIC_ID_AGENT_TASK_LIFECYCLE",
    "TOPIC_ID_INVOCATION_COMMAND",
]
