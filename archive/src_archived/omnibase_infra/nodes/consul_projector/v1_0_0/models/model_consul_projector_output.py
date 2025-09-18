#!/usr/bin/env python3


from pydantic import BaseModel

from omnibase_infra.models.consul.model_consul_health_response import (
    ModelConsulHealthResponse,
)
from omnibase_infra.models.consul.model_consul_kv_response import ModelConsulKvResponse

# Import shared Consul models
from omnibase_infra.models.consul.model_consul_service_response import (
    ModelConsulServiceResponse,
)

from .model_consul_topology_metrics import ModelConsulTopologyMetrics


class ModelConsulProjectorOutput(BaseModel):
    """Output model for Consul projector operation results.

    Node-specific model for returning projection operation results through effect outputs.
    """

    projection_result: (
        ModelConsulServiceResponse
        | ModelConsulHealthResponse
        | ModelConsulKvResponse
        | ModelConsulTopologyMetrics
        | dict[str, str | int | bool | list[str]]
        | list[dict[str, str | int | bool]]
        | str
    )
    projection_type: str
    timestamp: str  # ISO format datetime
    metadata: dict | None = None
