#!/usr/bin/env python3

from uuid import UUID

from pydantic import BaseModel, Field


class ModelConsulTopologyNode(BaseModel):
    """Topology node representation."""

    node_id: UUID = Field(..., description="Unique node identifier")
    node_name: str = Field(..., description="Node name")
    services: list[str] = Field(
        ..., description="List of services running on this node",
    )


class ModelConsulTopologyEdge(BaseModel):
    """Topology edge representing service connections."""

    source_service: str = Field(..., description="Source service name")
    target_service: str = Field(..., description="Target service name")
    connection_type: str = Field(
        ..., description="Type of connection (HTTP, gRPC, etc.)",
    )


class ModelConsulTopologyGraph(BaseModel):
    """Service topology graph with strongly typed details."""

    nodes: list[ModelConsulTopologyNode] = Field(
        ..., description="List of topology nodes",
    )
    edges: list[ModelConsulTopologyEdge] = Field(
        ..., description="List of topology edges",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional topology metadata",
    )
