#!/usr/bin/env python3

from pydantic import BaseModel, Field
from .model_consul_topology_graph import ModelConsulTopologyGraph
from .model_consul_topology_metrics import ModelConsulTopologyMetrics


class ModelConsulTopologyProjection(BaseModel):
    """Service topology projection result model."""
    
    topology_graph: ModelConsulTopologyGraph = Field(..., description="Strongly typed topology graph")
    metrics: ModelConsulTopologyMetrics = Field(..., description="Strongly typed topology metrics")