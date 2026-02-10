# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract-driven intent routing loader.

This module provides a function to load the intent_routing_table from a
contract YAML file. The routing table maps intent_type strings to target
node names, enabling the IntentExecutor to route intents to the
appropriate effect layer adapters.

Architecture:
    contract.yaml -> load_intent_routing_table() -> dict[str, str]
                                                     |
                                                     v
                                              IntentExecutor registration

    The loader reads the ``intent_consumption.intent_routing_table`` section
    from the contract YAML file, which declares the mapping between intent
    types and their target effect nodes.

Contract Format:
    ```yaml
    intent_consumption:
      subscribed_intents:
        - "consul.register"
        - "consul.deregister"
        - "postgres.upsert_registration"
      intent_routing_table:
        "consul.register": "node_registry_effect"
        "consul.deregister": "node_registry_effect"
        "postgres.upsert_registration": "node_registry_effect"
    ```

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - IntentExecutor: Uses the routing table for intent dispatch
    - load_event_bus_subcontract: Similar pattern for event_bus loading
    - contract.yaml: Source of intent routing declarations

.. versionadded:: 0.7.0
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_intent_routing_table(
    contract_path: Path,
    logger_override: logging.Logger | None = None,
) -> dict[str, str]:
    """Load intent routing table from a contract YAML file.

    Reads the ``intent_consumption.intent_routing_table`` section from the
    contract and returns a dict mapping intent_type strings to target node
    names.

    Args:
        contract_path: Path to the contract YAML file.
        logger_override: Optional logger for warnings. If not provided,
            uses the module-level logger.

    Returns:
        Dict mapping intent_type to target node name. Empty dict if the
        section is missing or the file cannot be loaded.

    Example:
        >>> table = load_intent_routing_table(Path("contract.yaml"))
        >>> print(table)
        {'consul.register': 'node_registry_effect',
         'postgres.upsert_registration': 'node_registry_effect'}
    """
    _logger = logger_override or logger

    if not contract_path.exists():
        _logger.warning(
            "Contract file not found: %s",
            contract_path,
        )
        return {}

    try:
        with contract_path.open(encoding="utf-8") as f:
            contract_data = yaml.safe_load(f)

        if contract_data is None:
            _logger.warning(
                "Empty contract file: %s",
                contract_path,
            )
            return {}

        intent_consumption = contract_data.get("intent_consumption")
        if not intent_consumption:
            _logger.debug(
                "No intent_consumption section in contract: %s",
                contract_path,
            )
            return {}

        routing_table = intent_consumption.get("intent_routing_table")
        if not routing_table:
            _logger.debug(
                "No intent_routing_table in intent_consumption: %s",
                contract_path,
            )
            return {}

        if not isinstance(routing_table, dict):
            _logger.warning(
                "intent_routing_table is not a dict in %s: got %s",
                contract_path,
                type(routing_table).__name__,
            )
            return {}

        # Validate all keys and values are strings
        validated: dict[str, str] = {}
        for intent_type, target_node in routing_table.items():
            if not isinstance(intent_type, str) or not isinstance(target_node, str):
                _logger.warning(
                    "Skipping non-string entry in intent_routing_table: %r -> %r in %s",
                    intent_type,
                    target_node,
                    contract_path,
                )
                continue
            validated[intent_type] = target_node

        if validated:
            _logger.debug(
                "Loaded intent routing table with %d entries from %s",
                len(validated),
                contract_path,
            )

        return validated

    except yaml.YAMLError as e:
        _logger.warning(
            "Failed to parse YAML in contract %s: %s",
            contract_path,
            e,
        )
        return {}
    except OSError as e:
        _logger.warning(
            "Failed to read contract file %s: %s",
            contract_path,
            e,
        )
        return {}


__all__: list[str] = ["load_intent_routing_table"]
