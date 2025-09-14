#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Any


class ModelConsulAdapterOutput(BaseModel):
    """Output model for Consul adapter operation results.
    
    Node-specific model for returning Consul operation results through effect outputs.
    """
    
    consul_operation_result: Any
    success: bool
    operation_type: str