#!/usr/bin/env python3

# Node-specific Consul adapter models
from omnibase_infra.models.consul.model_consul_health_response import (
    ModelConsulHealthCheckNode,
    ModelConsulHealthResponse,
)

# Import shared Consul models for convenience
from omnibase_infra.models.consul.model_consul_kv_request import ModelConsulKVRequest
from omnibase_infra.models.consul.model_consul_kv_response import ModelConsulKVResponse
from omnibase_infra.models.consul.model_consul_service_list_response import (
    ModelConsulServiceInfo,
    ModelConsulServiceListResponse,
)
from omnibase_infra.models.consul.model_consul_service_registration import (
    ModelConsulHealthCheck,
    ModelConsulServiceRegistration,
)
from omnibase_infra.models.consul.model_consul_service_response import (
    ModelConsulServiceResponse,
)

from .model_consul_adapter_input import ModelConsulAdapterInput
from .model_consul_adapter_output import ModelConsulAdapterOutput
from .model_consul_service_config import ModelConsulServiceConfig
from .model_consul_value_data import ModelConsulValueData

__all__ = [
    # Node-specific models
    "ModelConsulAdapterInput",
    "ModelConsulAdapterOutput",
    "ModelConsulHealthCheck",
    "ModelConsulHealthCheckNode",
    "ModelConsulHealthResponse",
    # Shared models (re-exported for backward compatibility)
    "ModelConsulKVRequest",
    "ModelConsulKVResponse",
    "ModelConsulServiceConfig",
    "ModelConsulServiceInfo",
    "ModelConsulServiceListResponse",
    "ModelConsulServiceRegistration",
    "ModelConsulServiceResponse",
    "ModelConsulValueData",
]
