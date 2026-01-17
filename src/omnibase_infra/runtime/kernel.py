# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Kernel - Contract-driven bootstrap entrypoint.

This module provides the public API for the ONEX runtime kernel,
re-exporting the core functions from service_kernel.py.

Functions:
    bootstrap: Initialize the ONEX runtime and execute the contract-driven bootstrap.
    main: Entry point for the ONEX runtime.
    load_runtime_config: Load runtime configuration from contract and environment.

Example:
    >>> from pathlib import Path
    >>> from omnibase_infra.runtime.kernel import load_runtime_config
    >>>
    >>> # Load configuration from contracts directory
    >>> contracts_dir = Path("./contracts")
    >>> config = load_runtime_config(contracts_dir)
    >>> print(config.input_topic)
    requests
    >>> print(config.event_bus.type)
    inmemory

Note:
    This module serves as a stable public API. The implementation resides
    in service_kernel.py, following the ONEX naming convention for service modules.
"""

from omnibase_infra.runtime.service_kernel import (
    bootstrap,
    load_runtime_config,
    main,
)

__all__: list[str] = [
    "bootstrap",
    "load_runtime_config",
    "main",
]
