# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build a Kafka topic router dict from a contract's published_events section.

Maps Python class names (e.g. "ModelNodeRegistrationAccepted") to their
declared Kafka topics (e.g. "onex.evt.platform.node-registration-accepted.v1").  # onex-topic-allow: pending contract auto-wiring

Convention: Python class name = "Model" + event_type from contract YAML.

Example::

    contract_data = yaml.safe_load(contract_path.read_text())
    router = build_topic_router_from_contract(contract_data)
    # router == {"ModelNodeRegistrationAccepted": "onex.evt.platform.node-registration-accepted.v1", ...}  # onex-topic-allow: pending contract auto-wiring

    applier = DispatchResultApplier(
        event_bus=event_bus,
        output_topic="responses",
        topic_router=router,
    )

OMN-4882 / OMN-4880
"""

from __future__ import annotations


def build_topic_router_from_contract(
    contract_data: dict[str, object],
) -> dict[str, str]:
    """Parse published_events from a loaded contract YAML dict.

    Returns a dict suitable for ``DispatchResultApplier(topic_router=...)``.
    Entries missing ``event_type`` or ``topic`` are silently skipped.

    Args:
        contract_data: Dict loaded from a contract.yaml file via yaml.safe_load.

    Returns:
        Mapping of Python class names to their declared Kafka topics.
        e.g. ``{"ModelNodeRegistrationAccepted": "onex.evt.platform.node-registration-accepted.v1"}``  # onex-topic-allow: pending contract auto-wiring
    """
    router: dict[str, str] = {}
    if not isinstance(contract_data, dict):
        return router
    published_events = contract_data.get("published_events", [])
    if not isinstance(published_events, list):
        return router
    for entry in published_events:
        if not isinstance(entry, dict):
            continue
        event_type = entry.get("event_type")
        topic = entry.get("topic")
        if (
            isinstance(event_type, str)
            and isinstance(topic, str)
            and event_type
            and topic
        ):
            router[f"Model{event_type}"] = topic
    return router
