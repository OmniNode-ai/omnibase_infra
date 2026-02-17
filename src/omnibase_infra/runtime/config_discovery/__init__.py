# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract-driven config discovery for ONEX Infrastructure.

This package provides the machinery to extract configuration requirements
from ONEX contracts, map them to Infisical secret paths, and prefetch
values during runtime bootstrap.

.. versionadded:: 0.10.0
    Created as part of OMN-2287.
"""

from omnibase_infra.runtime.config_discovery.config_prefetcher import (
    ConfigPrefetcher,
)
from omnibase_infra.runtime.config_discovery.contract_config_extractor import (
    ContractConfigExtractor,
)
from omnibase_infra.runtime.config_discovery.transport_config_map import (
    TransportConfigMap,
)

__all__ = [
    "ConfigPrefetcher",
    "ContractConfigExtractor",
    "TransportConfigMap",
]
