# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-wiring module for contract lifecycle hooks and auto-discovery.

Provides:
- Contract lifecycle hooks (on_start, validate_handshake, on_shutdown) that
  replace Plugin.initialize() and Plugin.shutdown() with declarative,
  contract-driven lifecycle management (OMN-7655).
- Contract auto-discovery from onex.nodes entry points, building an
  auto-wiring manifest for event bus wiring (OMN-7653).
- Handler auto-wiring engine that wires discovered contracts into the
  dispatch engine and event bus (OMN-7654).
"""

from omnibase_infra.runtime.auto_wiring.config import ModelLifecycleHookConfig
from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_from_paths,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    wire_from_manifest,
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
    ModelLifecycleHooks,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
    ModelDuplicateTopicOwnership,
)
from omnibase_infra.runtime.auto_wiring.result import ModelLifecycleHookResult
from omnibase_infra.runtime.auto_wiring.wiring import LifecycleHookExecutor

__all__ = [
    "EnumWiringOutcome",
    "LifecycleHookExecutor",
    "ModelAutoWiringContext",
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
    "ModelLifecycleHookConfig",
    "ModelLifecycleHookResult",
    "ModelLifecycleHooks",
    "discover_contracts",
    "discover_contracts_from_paths",
    "wire_from_manifest",
]
