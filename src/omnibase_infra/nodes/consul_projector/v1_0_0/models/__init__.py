#!/usr/bin/env python3

# Node-specific Consul projector models
from .model_consul_projector_input import ModelConsulProjectorInput
from .model_consul_projector_output import ModelConsulProjectorOutput
from .model_consul_service_projection import ModelConsulServiceProjection
from .model_consul_health_projection import ModelConsulHealthProjection
from .model_consul_kv_projection import ModelConsulKVProjection
from .model_consul_topology_projection import ModelConsulTopologyProjection

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