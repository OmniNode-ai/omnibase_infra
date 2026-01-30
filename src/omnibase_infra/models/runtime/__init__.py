# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Models for Plugin Loading.

This module exports runtime-specific models used by the Plugin Loader
and related runtime components.

.. versionadded:: 0.7.0
    Created as part of OMN-1132 Plugin Loader implementation.
"""

from omnibase_infra.models.runtime.model_contract_security_config import (
    ModelContractSecurityConfig,
)
from omnibase_infra.models.runtime.model_discovery_error import ModelDiscoveryError
from omnibase_infra.models.runtime.model_discovery_result import ModelDiscoveryResult
from omnibase_infra.models.runtime.model_discovery_warning import ModelDiscoveryWarning
from omnibase_infra.models.runtime.model_failed_plugin_load import (
    ModelFailedPluginLoad,
)
from omnibase_infra.models.runtime.model_handler_contract import ModelHandlerContract
from omnibase_infra.models.runtime.model_loaded_handler import ModelLoadedHandler
from omnibase_infra.models.runtime.model_plugin_load_context import (
    ModelPluginLoadContext,
)
from omnibase_infra.models.runtime.model_plugin_load_summary import (
    ModelPluginLoadSummary,
)

# ModelContractLoadResult and ModelRuntimeContractConfig are exported from
# omnibase_infra.runtime.models (canonical location for runtime loader models)
from omnibase_infra.runtime.models import (
    ModelContractLoadResult,
    ModelRuntimeContractConfig,
)

__all__ = [
    "ModelContractLoadResult",
    "ModelContractSecurityConfig",
    "ModelDiscoveryError",
    "ModelDiscoveryResult",
    "ModelDiscoveryWarning",
    "ModelFailedPluginLoad",
    "ModelHandlerContract",
    "ModelLoadedHandler",
    "ModelPluginLoadContext",
    "ModelPluginLoadSummary",
    "ModelRuntimeContractConfig",
]
