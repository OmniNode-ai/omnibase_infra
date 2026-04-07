# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract auto-discovery, handler auto-wiring, and lifecycle hooks.

Combines:
- Discovery engine: scans onex.nodes entry points (OMN-7653)
- Handler wiring: wires handlers into MessageDispatchEngine (OMN-7654)
- Lifecycle hooks: on_start, validate_handshake, on_shutdown (OMN-7655)
- Handshake validation: retry + quarantine semantics (OMN-7657)
- Kernel integration: unified auto-wiring for service_kernel (OMN-7656)
"""

from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_from_paths,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.lifecycle import LifecycleHookExecutor
from omnibase_infra.runtime.auto_wiring.models import (
    HandshakeFailureReason,
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelDiscoveryError,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
    ModelHandshakeConfig,
    ModelLifecycleHookConfig,
    ModelLifecycleHookResult,
    ModelLifecycleHooks,
    ModelQuarantineRecord,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
    ModelDuplicateTopicOwnership,
)

__all__ = [
    "EnumWiringOutcome",
    "HandshakeFailureReason",
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
    "ModelHandshakeConfig",
    "ModelLifecycleHookConfig",
    "ModelLifecycleHookResult",
    "ModelLifecycleHooks",
    "ModelQuarantineRecord",
    "discover_contracts",
    "discover_contracts_from_paths",
    "wire_from_manifest",
]
