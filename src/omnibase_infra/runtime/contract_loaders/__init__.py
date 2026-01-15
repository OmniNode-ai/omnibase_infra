# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract loaders for declarative ONEX configuration.

This module provides utilities for loading contract-driven configuration
from contract.yaml files. These loaders support the ONEX declarative
pattern where behavior is defined in YAML rather than Python code.

Components:
    - handler_routing_loader: Load handler routing subcontracts from contract.yaml

Usage:
    ```python
    from omnibase_infra.runtime.contract_loaders import (
        load_handler_routing_subcontract,
        convert_class_to_handler_key,
    )

    # Load routing from contract.yaml
    routing = load_handler_routing_subcontract(Path("path/to/contract.yaml"))

    # Convert class name to handler key
    key = convert_class_to_handler_key("HandlerNodeIntrospected")
    # Returns: "handler-node-introspected"
    ```
"""

from omnibase_infra.runtime.contract_loaders.handler_routing_loader import (
    convert_class_to_handler_key,
    load_handler_routing_subcontract,
)

__all__ = [
    "convert_class_to_handler_key",
    "load_handler_routing_subcontract",
]
