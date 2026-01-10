# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Projector Schema Models Module.

Provides Pydantic models for projector schema definitions, used by
ProjectorSchemaValidator for schema validation and migration SQL generation.

NOTE: Schema models are temporarily defined here until omnibase_core provides
them at omnibase_core.models.projectors. Once available, this module should
re-export from omnibase_core instead.

Exports:
    ModelProjectorSchema: Complete projection table schema definition
    ModelProjectorColumn: Individual column definition within a schema
    ModelProjectorIndex: Index definition for projection tables
    ModelProjectorValidationError: Structured error for validation failures
    ModelProjectorDiscoveryResult: Result of projector contract discovery

Related Tickets:
    - OMN-1168: ProjectorPluginLoader contract discovery loading
"""

from omnibase_infra.models.projectors.model_projector_column import (
    ModelProjectorColumn,
)
from omnibase_infra.models.projectors.model_projector_discovery_result import (
    ModelProjectorDiscoveryResult,
)
from omnibase_infra.models.projectors.model_projector_index import ModelProjectorIndex
from omnibase_infra.models.projectors.model_projector_schema import ModelProjectorSchema
from omnibase_infra.models.projectors.model_projector_validation_error import (
    ModelProjectorValidationError,
)

__all__ = [
    "ModelProjectorColumn",
    "ModelProjectorDiscoveryResult",
    "ModelProjectorIndex",
    "ModelProjectorSchema",
    "ModelProjectorValidationError",
]
