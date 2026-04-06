# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract auto-discovery and auto-wiring for ONEX runtime.

Scans ``onex.nodes`` entry points, loads contract.yaml files, and builds
a :class:`ModelAutoWiringManifest` describing all discoverable nodes and
their event bus wiring.

Part of OMN-7653.
"""

from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_from_paths,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelDiscoveryError,
    ModelEventBusWiring,
)

__all__ = [
    "ModelAutoWiringManifest",
    "ModelContractVersion",
    "ModelDiscoveredContract",
    "ModelDiscoveryError",
    "ModelEventBusWiring",
    "discover_contracts",
    "discover_contracts_from_paths",
]
