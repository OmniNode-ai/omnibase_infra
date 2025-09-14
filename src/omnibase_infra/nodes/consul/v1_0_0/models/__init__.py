#!/usr/bin/env python3

# Node-specific Consul adapter models
from .model_consul_adapter_input import ModelConsulAdapterInput
from .model_consul_adapter_output import ModelConsulAdapterOutput
from .model_consul_value_data import ModelConsulValueData
from .model_consul_service_config import ModelConsulServiceConfig

# Import shared Consul models for convenience
from omnibase_infra.models.consul.model_consul_kv_request import ModelConsulKVRequest
from omnibase_infra.models.consul.model_consul_kv_response import ModelConsulKVResponse
from omnibase_infra.models.consul.model_consul_service_registration import (
    ModelConsulServiceRegistration,
    ModelConsulHealthCheck,
)
from omnibase_infra.models.consul.model_consul_service_response import ModelConsulServiceResponse
from omnibase_infra.models.consul.model_consul_service_list_response import (
    ModelConsulServiceListResponse,
    ModelConsulServiceInfo,
)
from omnibase_infra.models.consul.model_consul_health_response import (
    ModelConsulHealthResponse,
    ModelConsulHealthCheckNode,
)

__all__ = [
    # Node-specific models
    "ModelConsulAdapterInput",
    "ModelConsulAdapterOutput",
    "ModelConsulValueData",
    "ModelConsulServiceConfig",
    
    # Shared models (re-exported for backward compatibility)
    "ModelConsulKVRequest",
    "ModelConsulKVResponse", 
    "ModelConsulServiceRegistration",
    "ModelConsulHealthCheck",
    "ModelConsulServiceResponse",
    "ModelConsulServiceListResponse",
    "ModelConsulServiceInfo",
    "ModelConsulHealthResponse",
    "ModelConsulHealthCheckNode",
]