# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract auto-discovery and auto-wiring for ONEX runtime.

Scans ``onex.nodes`` entry points, loads contract.yaml files, and builds
a :class:`ModelAutoWiringManifest` describing all discoverable nodes and
their event bus wiring.

Part of OMN-7653 (discovery) and OMN-7654 (wiring).
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
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
    ModelDuplicateTopicOwnership,
)
from omnibase_infra.runtime.auto_wiring.wiring import (
    wire_from_manifest,
)

__all__ = [
    "EnumWiringOutcome",
    "ModelAutoWiringManifest",
    "ModelAutoWiringReport",
    "ModelContractVersion",
    "ModelContractWiringResult",
    "ModelDiscoveredContract",
    "ModelDiscoveryError",
    "ModelDuplicateTopicOwnership",
    "ModelEventBusWiring",
    "ModelHandlerRef",
    "ModelHandlerRouting",
    "ModelHandlerRoutingEntry",
    "discover_contracts",
    "discover_contracts_from_paths",
    "wire_from_manifest",
]
