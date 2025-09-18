#!/usr/bin/env python3

from enum import Enum


class ModelConsulProjectionType(Enum):
    """Enumeration of Consul projection types."""

    SERVICE_STATE = "service_state"
    HEALTH_STATE = "health_state"
    KV_STATE = "kv_state"
    TOPOLOGY = "topology"
