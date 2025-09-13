#!/usr/bin/env python3

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ModelConsulHealthCheckNode(BaseModel):
    """Health check information for a specific node."""
    
    node: Optional[str] = None
    service_id: Optional[str] = None
    service_name: Optional[str] = None
    status: str


class ModelConsulHealthResponse(BaseModel):
    """Response for Consul health check operations.
    
    Shared model used across Consul infrastructure nodes for health check responses.
    """
    
    status: str
    service_name: Optional[str] = None
    health_checks: Optional[List[ModelConsulHealthCheckNode]] = None
    health_summary: Optional[Dict[str, Any]] = None