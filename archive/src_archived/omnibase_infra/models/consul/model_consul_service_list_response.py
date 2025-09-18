#!/usr/bin/env python3


from pydantic import BaseModel


class ModelConsulServiceInfo(BaseModel):
    """Information about a Consul service."""

    id: str
    name: str
    port: int
    address: str
    tags: list[str]


class ModelConsulServiceListResponse(BaseModel):
    """Response for Consul service list operations.

    Shared model used across Consul infrastructure nodes for service listing responses.
    """

    status: str
    services: list[ModelConsulServiceInfo]
    count: int
