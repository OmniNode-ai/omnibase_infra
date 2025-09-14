#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Literal


class ModelConsulServiceResponse(BaseModel):
    """Response for Consul service operations.
    
    Shared model used across Consul infrastructure nodes for service operation responses.
    """
    
    status: Literal["success", "failed", "not_found"]
    service_id: str
    service_name: str