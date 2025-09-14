#!/usr/bin/env python3

from pydantic import BaseModel
from typing import List


class ModelConsulServiceInfo(BaseModel):
    """Information about a Consul service."""
    
    id: str
    name: str
    port: int
    address: str
    tags: List[str]


class ModelConsulServiceListResponse(BaseModel):
    """Response for Consul service list operations.
    
    Shared model used across Consul infrastructure nodes for service listing responses.
    """
    
    status: str
    services: List[ModelConsulServiceInfo]
    count: int