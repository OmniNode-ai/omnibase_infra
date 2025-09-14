#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Any, Optional


class ModelConsulProjectorOutput(BaseModel):
    """Output model for Consul projector operation results.
    
    Node-specific model for returning projection operation results through effect outputs.
    """
    
    projection_result: Any
    projection_type: str
    timestamp: str  # ISO format datetime
    metadata: Optional[dict] = None