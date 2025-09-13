#!/usr/bin/env python3

# Shared Consul models for infrastructure nodes
from .model_consul_kv_request import ModelConsulKVRequest
from .model_consul_kv_response import ModelConsulKVResponse
from .model_consul_service_registration import (
    ModelConsulServiceRegistration,
    ModelConsulHealthCheck,
)
from .model_consul_service_response import ModelConsulServiceResponse
from .model_consul_service_list_response import (
    ModelConsulServiceListResponse,
    ModelConsulServiceInfo,
)
from .model_consul_health_response import (
    ModelConsulHealthResponse,
    ModelConsulHealthCheckNode,
)

__all__ = [
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