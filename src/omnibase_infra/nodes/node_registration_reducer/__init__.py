# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Node Registration Reducer - FSM-driven declarative reducer for node registration.

Also exports RegistrationReducer (pure-function reducer) migrated from
nodes.reducers (OMN-3989).
"""

from omnibase_infra.nodes.node_registration_reducer.models import ModelValidationResult
from omnibase_infra.nodes.node_registration_reducer.node import NodeRegistrationReducer
from omnibase_infra.nodes.node_registration_reducer.registration_reducer import (
    RegistrationReducer,
)
from omnibase_infra.nodes.node_registration_reducer.registry import (
    RegistryInfraNodeRegistrationReducer,
)

__all__ = [
    "ModelValidationResult",
    "NodeRegistrationReducer",
    "RegistrationReducer",
    "RegistryInfraNodeRegistrationReducer",
]
