#!/usr/bin/env python3


from pydantic import BaseModel

from omnibase_infra.models.consul.model_consul_health_response import (
    ModelConsulHealthResponse,
)
from omnibase_infra.models.consul.model_consul_kv_response import ModelConsulKvResponse
from omnibase_infra.models.consul.model_consul_service_list_response import (
    ModelConsulServiceListResponse,
)

# Import shared Consul models
from omnibase_infra.models.consul.model_consul_service_response import (
    ModelConsulServiceResponse,
)


class ModelConsulAdapterOutput(BaseModel):
    """Output model for Consul adapter operation results.
    
    Node-specific model for returning Consul operation results through effect outputs.
    """

    consul_operation_result: ModelConsulServiceResponse | ModelConsulHealthResponse | ModelConsulKvResponse | ModelConsulServiceListResponse | dict[str, str | int | bool] | list[dict[str, str | int | bool]] | str | bool
    success: bool
    operation_type: str
