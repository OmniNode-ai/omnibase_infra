# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pydantic models for contract auto-discovery, auto-wiring, and lifecycle hooks.

Includes:
- Discovery models: contract manifest, handler routing refs (OMN-7653, OMN-7654)
- Lifecycle hook models: hook config, results, handshake, quarantine (OMN-7655, OMN-7657)
"""

from omnibase_infra.runtime.auto_wiring.models.enum_handshake_failure_reason import (
    HandshakeFailureReason,
)
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovery_error import (
    ModelDiscoveryError,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_ref import (
    ModelHandlerRef,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing import (
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.auto_wiring.models.model_handshake_config import (
    ModelHandshakeConfig,
)
from omnibase_infra.runtime.auto_wiring.models.model_lifecycle_hook_config import (
    ModelLifecycleHookConfig,
)
from omnibase_infra.runtime.auto_wiring.models.model_lifecycle_hook_result import (
    ModelLifecycleHookResult,
)
from omnibase_infra.runtime.auto_wiring.models.model_lifecycle_hooks import (
    ModelLifecycleHooks,
)
from omnibase_infra.runtime.auto_wiring.models.model_quarantine_record import (
    ModelQuarantineRecord,
)
from omnibase_infra.runtime.auto_wiring.models.model_runtime_profile_policy import (
    ModelRuntimeProfilePolicy,
)

__all__ = [
    "HandshakeFailureReason",
    "ModelAutoWiringManifest",
    "ModelContractVersion",
    "ModelDiscoveredContract",
    "ModelDiscoveryError",
    "ModelEventBusWiring",
    "ModelHandlerRef",
    "ModelHandlerRouting",
    "ModelHandlerRoutingEntry",
    "ModelHandshakeConfig",
    "ModelLifecycleHookConfig",
    "ModelLifecycleHookResult",
    "ModelLifecycleHooks",
    "ModelQuarantineRecord",
    "ModelRuntimeProfilePolicy",
]
