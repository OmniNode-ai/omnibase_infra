# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for contract-driven config discovery.

.. versionadded:: 0.10.0
    Created as part of OMN-2287.
"""

from omnibase_infra.runtime.config_discovery.models.model_config_requirement import (
    ModelConfigRequirement,
)
from omnibase_infra.runtime.config_discovery.models.model_config_requirements import (
    ModelConfigRequirements,
)
from omnibase_infra.runtime.config_discovery.models.model_transport_config_spec import (
    ModelTransportConfigSpec,
)
from omnibase_infra.runtime.config_discovery.models.protocol_secret_resolver import (
    ProtocolSecretResolver,
)

__all__ = [
    "ModelConfigRequirement",
    "ModelConfigRequirements",
    "ModelTransportConfigSpec",
    "ProtocolSecretResolver",
]
