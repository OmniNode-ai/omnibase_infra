#!/usr/bin/env python3

# Node-specific Consul projector models
from .model_consul_projector_input import ModelConsulProjectorInput
from .model_consul_projector_output import ModelConsulProjectorOutput
from .model_consul_projections import (
    ModelConsulServiceProjection,
    ModelConsulHealthProjection,
    ModelConsulKVProjection,
    ModelConsulTopologyProjection,
)

# Import shared Consul models for convenience
from omnibase_infra.models.consul import (
    ModelConsulKVResponse,
    ModelConsulServiceListResponse,
    ModelConsulHealthResponse,
)

__all__ = [
    # Node-specific models
    "ModelConsulProjectorInput",
    "ModelConsulProjectorOutput",
    "ModelConsulServiceProjection",
    "ModelConsulHealthProjection",
    "ModelConsulKVProjection",
    "ModelConsulTopologyProjection",
    
    # Shared models (re-exported for convenience)
    "ModelConsulKVResponse",
    "ModelConsulServiceListResponse",
    "ModelConsulHealthResponse",
]