# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Operation bindings models for declarative handler parameter mapping.

This module provides models for binding expressions used in declarative
handler parameter resolution from contract.yaml.

.. versionadded:: 0.2.6
    Added ModelOperationBindingsSubcontract and ModelBindingResolutionResult
    as part of OMN-1518 - Declarative operation bindings.
"""

from omnibase_infra.models.bindings.constants import (
    DEFAULT_JSON_RECURSION_DEPTH,
    EXPRESSION_PATTERN,
    MAX_EXPRESSION_LENGTH,
    MAX_JSON_RECURSION_DEPTH,
    MAX_PATH_SEGMENTS,
    MIN_JSON_RECURSION_DEPTH,
    VALID_CONTEXT_PATHS,
    VALID_SOURCES,
)
from omnibase_infra.models.bindings.model_binding_resolution_result import (
    ModelBindingResolutionResult,
)
from omnibase_infra.models.bindings.model_operation_binding import (
    ModelOperationBinding,
)
from omnibase_infra.models.bindings.model_operation_bindings_subcontract import (
    MAX_EXPRESSION_LENGTH_LIMIT,
    MAX_PATH_SEGMENTS_LIMIT,
    MIN_EXPRESSION_LENGTH,
    MIN_PATH_SEGMENTS,
    ModelOperationBindingsSubcontract,
)
from omnibase_infra.models.bindings.model_parsed_binding import (
    ModelParsedBinding,
)

__all__: list[str] = [
    # Constants
    "DEFAULT_JSON_RECURSION_DEPTH",
    "EXPRESSION_PATTERN",
    "MAX_EXPRESSION_LENGTH",
    "MAX_EXPRESSION_LENGTH_LIMIT",
    "MAX_JSON_RECURSION_DEPTH",
    "MAX_PATH_SEGMENTS",
    "MAX_PATH_SEGMENTS_LIMIT",
    "MIN_EXPRESSION_LENGTH",
    "MIN_JSON_RECURSION_DEPTH",
    "MIN_PATH_SEGMENTS",
    "VALID_CONTEXT_PATHS",
    "VALID_SOURCES",
    # Models
    "ModelBindingResolutionResult",
    "ModelOperationBinding",
    "ModelOperationBindingsSubcontract",
    "ModelParsedBinding",
]
