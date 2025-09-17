#!/usr/bin/env python3

from typing import Literal

from pydantic import BaseModel


class ModelConsulServiceResponse(BaseModel):
    """Response for Consul service operations.
    
    Shared model used across Consul infrastructure nodes for service operation responses.
    """

    status: Literal["success", "failed", "not_found"]
    service_id: str
    service_name: str
