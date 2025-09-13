#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal


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
        "consul_health_check"
    ]
    key_path: Optional[str] = None
    value_data: Optional[Dict[str, Any]] = None
    service_config: Optional[Dict[str, Any]] = None
    recurse: bool = False