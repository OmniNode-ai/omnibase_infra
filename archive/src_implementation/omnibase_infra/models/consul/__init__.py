#!/usr/bin/env python3

# Shared Consul models for infrastructure nodes
from .model_consul_health_response import (
    ModelConsulHealthCheckNode,
    ModelConsulHealthResponse,
)
from .model_consul_kv_request import ModelConsulKVRequest
from .model_consul_kv_response import ModelConsulKVResponse
from .model_consul_service_list_response import (
    ModelConsulServiceInfo,
    ModelConsulServiceListResponse,
)
from .model_consul_service_registration import (
    ModelConsulHealthCheck,
    ModelConsulServiceRegistration,
)
from .model_consul_service_response import ModelConsulServiceResponse

__all__ = [
    "ModelConsulHealthCheck",
    "ModelConsulHealthCheckNode",
    "ModelConsulHealthResponse",
    "ModelConsulKVRequest",
    "ModelConsulKVResponse",
    "ModelConsulServiceInfo",
    "ModelConsulServiceListResponse",
    "ModelConsulServiceRegistration",
    "ModelConsulServiceResponse",
]
