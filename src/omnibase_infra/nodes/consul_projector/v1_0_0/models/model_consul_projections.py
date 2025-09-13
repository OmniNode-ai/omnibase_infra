#!/usr/bin/env python3

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ModelConsulServiceProjection(BaseModel):
    """Service state projection result model."""
    
    services: List[Dict[str, Any]]
    total_services: int


class ModelConsulHealthProjection(BaseModel):
    """Health state projection result model."""
    
    health_summary: Dict[str, int]
    service_health: List[Dict[str, Any]]


class ModelConsulKVProjection(BaseModel):
    """KV state projection result model."""
    
    key_summary: Dict[str, Any]
    key_details: Optional[List[Dict[str, Any]]] = None


class ModelConsulTopologyProjection(BaseModel):
    """Service topology projection result model."""
    
    topology_graph: Dict[str, Any]
    metrics: Dict[str, Any]