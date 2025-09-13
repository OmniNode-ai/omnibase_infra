#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Optional, List, Literal


class ModelConsulProjectorInput(BaseModel):
    """Input model for Consul projector operations from event envelopes.
    
    Node-specific model for processing event envelope payloads into projection operations.
    """
    
    projection_type: Literal["service_state", "health_state", "kv_state", "topology"]
    target_services: Optional[List[str]] = None
    include_metadata: bool = True
    aggregation_window: int = 300  # seconds