#!/usr/bin/env python3

# Node-specific Consul projector models
# Import shared Consul models for convenience
from omnibase_infra.models.consul import (
    ModelConsulHealthResponse,
    ModelConsulKVResponse,
    ModelConsulServiceListResponse,
)

from .model_consul_cache_entry import (
    ModelConsulHealthCacheEntry,
    ModelConsulKVCacheEntry,
    ModelConsulServiceCacheEntry,
)
from .model_consul_health_projection import ModelConsulHealthProjection
from .model_consul_kv_projection import ModelConsulKVProjection
from .model_consul_projector_input import ModelConsulProjectorInput
from .model_consul_projector_output import ModelConsulProjectorOutput
from .model_consul_service_projection import ModelConsulServiceProjection
from .model_consul_topology_projection import ModelConsulTopologyProjection

__all__ = [
    # Node-specific models
    "ModelConsulProjectorInput",
    "ModelConsulProjectorOutput",
    "ModelConsulServiceProjection",
    "ModelConsulHealthProjection",
    "ModelConsulKVProjection",
    "ModelConsulTopologyProjection",

    # Cache models
    "ModelConsulServiceCacheEntry",
    "ModelConsulHealthCacheEntry",
    "ModelConsulKVCacheEntry",

    # Shared models (re-exported for convenience)
    "ModelConsulKVResponse",
    "ModelConsulServiceListResponse",
    "ModelConsulHealthResponse",
]
