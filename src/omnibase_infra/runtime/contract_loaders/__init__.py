# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract loaders for declarative ONEX configuration.

This module provides utilities for loading contract-driven configuration
from contract.yaml files. These loaders support the ONEX declarative
pattern where behavior is defined in YAML rather than Python code.

Components:
    - handler_routing_loader: Load handler routing subcontracts from contract.yaml
    - operation_bindings_loader: Load operation bindings subcontracts from contract.yaml

Usage:
    ```python
    from omnibase_infra.runtime.contract_loaders import (
        load_handler_routing_subcontract,
        load_operation_bindings_subcontract,
        convert_class_to_handler_key,
    )

    # Load routing from contract.yaml
    routing = load_handler_routing_subcontract(Path("path/to/contract.yaml"))

    # Load bindings from contract.yaml
    bindings = load_operation_bindings_subcontract(Path("path/to/contract.yaml"))

    # Convert class name to handler key
    key = convert_class_to_handler_key("HandlerNodeIntrospected")
    # Returns: "handler-node-introspected"
    ```
"""

from omnibase_infra.runtime.contract_loaders.handler_routing_loader import (
    MAX_CONTRACT_FILE_SIZE_BYTES,
    VALID_ROUTING_STRATEGIES,
    convert_class_to_handler_key,
    load_handler_class_info_from_contract,
    load_handler_routing_subcontract,
)
from omnibase_infra.runtime.contract_loaders.operation_bindings_loader import (
    load_operation_bindings_subcontract,
)

__all__ = [
    "MAX_CONTRACT_FILE_SIZE_BYTES",
    "VALID_ROUTING_STRATEGIES",
    "convert_class_to_handler_key",
    "load_handler_class_info_from_contract",
    "load_handler_routing_subcontract",
    "load_operation_bindings_subcontract",
]
