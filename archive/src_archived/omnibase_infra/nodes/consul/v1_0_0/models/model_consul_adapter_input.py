#!/usr/bin/env python3

from typing import Literal

from pydantic import BaseModel

from .model_consul_service_config import ModelConsulServiceConfig
from .model_consul_value_data import ModelConsulValueData


class ModelConsulAdapterInput(BaseModel):
    """Input model for Consul adapter operations from event envelopes.

    Node-specific model for processing event envelope payloads into Consul operations.
    """

    action: Literal[
        "consul_kv_get",
        "consul_kv_put",
        "consul_kv_delete",
        "consul_service_register",
        "consul_service_deregister",
        "consul_service_list",
        "consul_health_check",
    ]
    key_path: str | None = None
    value_data: ModelConsulValueData | None = None
    service_config: ModelConsulServiceConfig | None = None
    recurse: bool = False
