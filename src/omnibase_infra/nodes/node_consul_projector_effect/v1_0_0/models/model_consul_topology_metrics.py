#!/usr/bin/env python3

from pydantic import BaseModel, Field


class ModelConsulTopologyMetrics(BaseModel):
    """Topology metrics with strongly typed details."""

    total_nodes: int = Field(..., description="Total number of nodes in topology")
    total_services: int = Field(..., description="Total number of services in topology")
    total_connections: int = Field(..., description="Total number of service connections")
    average_connections_per_service: float = Field(..., description="Average connections per service")
    cluster_coefficient: float = Field(..., description="Clustering coefficient of the topology")
    max_path_length: int = Field(..., description="Maximum path length between services")
