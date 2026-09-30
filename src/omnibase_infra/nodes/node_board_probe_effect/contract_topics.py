# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read board-probe publication topics from the node contract (OMN-19937)."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONTRACT_PATH = Path(__file__).resolve().parent / "contract.yaml"


def board_probe_result_topic(contract_path: Path = _CONTRACT_PATH) -> str:
    """Return the sole board-probe-result topic declared by the contract."""
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    event_bus = raw.get("event_bus") if isinstance(raw, dict) else None
    topics = event_bus.get("publish_topics") if isinstance(event_bus, dict) else None
    if not isinstance(topics, list) or not all(
        isinstance(topic, str) for topic in topics
    ):
        raise ValueError(
            f"{contract_path}: event_bus.publish_topics must be a list of strings"
        )
    matches = [topic for topic in topics if topic.endswith("board-probe-result.v1")]
    if len(matches) != 1:
        raise ValueError(
            f"{contract_path}: expected exactly one board-probe-result.v1 publish "
            f"topic, found {matches}"
        )
    return matches[0]


__all__ = ["board_probe_result_topic"]
