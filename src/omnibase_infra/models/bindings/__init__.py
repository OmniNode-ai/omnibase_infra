# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Operation bindings models for declarative handler parameter mapping.

This module provides models for binding expressions used in declarative
handler parameter resolution from contract.yaml.

.. versionadded:: 0.2.6
    Added ModelOperationBindingsSubcontract and ModelBindingResolutionResult
    as part of OMN-1518 - Declarative operation bindings.
"""

from omnibase_infra.models.bindings.model_binding_resolution_result import (
    ModelBindingResolutionResult,
)
from omnibase_infra.models.bindings.model_operation_binding import (
    ModelOperationBinding,
)
from omnibase_infra.models.bindings.model_operation_bindings_subcontract import (
    ModelOperationBindingsSubcontract,
)
from omnibase_infra.models.bindings.model_parsed_binding import (
    ModelParsedBinding,
)

__all__: list[str] = [
    "ModelBindingResolutionResult",
    "ModelOperationBinding",
    "ModelOperationBindingsSubcontract",
    "ModelParsedBinding",
]
