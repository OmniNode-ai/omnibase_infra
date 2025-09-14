#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Union, Dict, List

# Import shared Consul models
from omnibase_infra.models.consul.model_consul_service_response import ModelConsulServiceResponse
from omnibase_infra.models.consul.model_consul_health_response import ModelConsulHealthResponse
from omnibase_infra.models.consul.model_consul_kv_response import ModelConsulKvResponse
from omnibase_infra.models.consul.model_consul_service_list_response import ModelConsulServiceListResponse


class ModelConsulAdapterOutput(BaseModel):
    """Output model for Consul adapter operation results.
    
    Node-specific model for returning Consul operation results through effect outputs.
    """
    
    consul_operation_result: Union[
        ModelConsulServiceResponse,
        ModelConsulHealthResponse,
        ModelConsulKvResponse,
        ModelConsulServiceListResponse,
        Dict[str, Union[str, int, bool]],
        List[Dict[str, Union[str, int, bool]]],
        str,
        bool
    ]
    success: bool
    operation_type: str