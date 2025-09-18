#!/usr/bin/env python3

from pydantic import BaseModel, Field, HttpUrl


class ModelConsulUrl(BaseModel):
    """URL model for Consul configurations."""

    url: HttpUrl = Field(
        ..., description="HTTP URL for health checks or service addresses",
    )
