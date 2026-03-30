# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Parse contract YAML files for runtime verification-relevant fields.

This module extracts the subset of contract.yaml fields needed by the
verification probes: topics, handlers, events, and FSM states. It does
not attempt to parse the full contract schema -- only what runtime
verification needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelParsedContractForVerification(BaseModel):
    """Verification-relevant fields extracted from a contract.yaml.

    Phase 1 extraction surface for runtime-relevant event and FSM metadata.
    Additional fields (projection tables, service dependencies, health endpoints)
    should be added only when new probe classes require them, not speculatively.

    Attributes:
        name: Contract name (e.g., "node_registration_orchestrator").
        node_type: ONEX node type (e.g., "ORCHESTRATOR_GENERIC").
        subscribe_topics: Topics the node subscribes to via event_bus.
        publish_topics: Topics the node publishes to via event_bus.
        handler_names: Handler names from handler_routing section.
        consumed_events: Event types consumed (from consumed_events section).
        published_events: Event types published (from published_events section).
        fsm_states: FSM states from handler_routing state_decision_matrix entries.
        contract_path: Path to the source contract.yaml file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(
        ...,
        description="Contract name (e.g., 'node_registration_orchestrator').",
    )
    node_type: str = Field(
        ...,
        description="ONEX node type (e.g., 'ORCHESTRATOR_GENERIC').",
    )
    subscribe_topics: tuple[str, ...] = Field(
        default=(),
        description="Topics the node subscribes to via event_bus.",
    )
    publish_topics: tuple[str, ...] = Field(
        default=(),
        description="Topics the node publishes to via event_bus.",
    )
    handler_names: tuple[str, ...] = Field(
        default=(),
        description="Handler names from handler_routing section.",
    )
    consumed_events: tuple[str, ...] = Field(
        default=(),
        description="Event types consumed (from consumed_events section).",
    )
    published_events: tuple[str, ...] = Field(
        default=(),
        description="Event types published (from published_events section).",
    )
    fsm_states: tuple[str, ...] = Field(
        default=(),
        description="FSM states from handler_routing state_decision_matrix.",
    )
    contract_path: str = Field(
        default="",
        description="Path to the source contract.yaml file.",
    )


# ONEX_EXCLUDE: any_type - YAML safe_load returns untyped dicts
def _extract_handler_names(
    data: dict[str, Any],
) -> tuple[str, ...]:
    """Extract handler names from handler_routing section."""
    handler_routing = data.get("handler_routing", {})
    if not handler_routing:
        return ()

    handlers = handler_routing.get("handlers", [])
    names: list[str] = []
    for entry in handlers:
        if isinstance(entry, dict):
            # Handler name can be at entry["name"] or entry["handler"]["name"]
            name = entry.get("name", "")
            if not name:
                handler_dict = entry.get("handler", {})
                if isinstance(handler_dict, dict):
                    name = handler_dict.get("name", "")
            if name:
                names.append(name)
    return tuple(names)


# ONEX_EXCLUDE: any_type - YAML safe_load returns untyped dicts
def _extract_fsm_states(data: dict[str, Any]) -> tuple[str, ...]:
    """Extract FSM states from handler_routing state_decision_matrix entries.

    The state_decision_matrix can be either:
    - A dict keyed by state name (legacy format)
    - A list of dicts with 'current_state' keys (current format)
    """
    handler_routing = data.get("handler_routing", {})
    if not handler_routing:
        return ()

    handlers = handler_routing.get("handlers", [])
    states: set[str] = set()
    for entry in handlers:
        if isinstance(entry, dict):
            matrix = entry.get("state_decision_matrix", None)
            if isinstance(matrix, dict):
                # Legacy format: dict keyed by state name
                states.update(matrix.keys())
            elif isinstance(matrix, list):
                # Current format: list of dicts with current_state
                for item in matrix:
                    if isinstance(item, dict):
                        state = item.get("current_state")
                        if state is not None:
                            states.add(str(state))
    return tuple(sorted(states))


# ONEX_EXCLUDE: any_type - YAML safe_load returns untyped dicts
def _extract_event_types(
    events_section: list[dict[str, Any]] | None,
) -> tuple[str, ...]:
    """Extract event type names from consumed_events or published_events."""
    if not events_section:
        return ()

    types: list[str] = []
    for event in events_section:
        if isinstance(event, dict):
            event_type = event.get("event_type", "")
            if event_type:
                types.append(event_type)
    return tuple(types)


def parse_contract_for_verification(
    contract_path: Path,
) -> ModelParsedContractForVerification:
    """Parse a contract.yaml and extract verification-relevant fields.

    Args:
        contract_path: Path to the contract.yaml file.

    Returns:
        A ModelParsedContractForVerification with extracted fields.

    Raises:
        FileNotFoundError: If the contract path does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    with open(contract_path) as f:
        data = yaml.safe_load(f) or {}

    name = data.get("name", contract_path.parent.name)
    node_type = data.get("node_type", "UNKNOWN")

    # Event bus topics
    event_bus = data.get("event_bus", {}) or {}
    subscribe_topics = tuple(event_bus.get("subscribe_topics", []) or [])
    publish_topics = tuple(event_bus.get("publish_topics", []) or [])

    # Handler names and FSM states
    handler_names = _extract_handler_names(data)
    fsm_states = _extract_fsm_states(data)

    # Consumed and published events
    consumed_events = _extract_event_types(data.get("consumed_events"))
    published_events = _extract_event_types(data.get("published_events"))

    return ModelParsedContractForVerification(
        name=name,
        node_type=node_type,
        subscribe_topics=subscribe_topics,
        publish_topics=publish_topics,
        handler_names=handler_names,
        consumed_events=consumed_events,
        published_events=published_events,
        fsm_states=fsm_states,
        contract_path=str(contract_path),
    )


__all__: list[str] = [
    "ModelParsedContractForVerification",
    "parse_contract_for_verification",
]
