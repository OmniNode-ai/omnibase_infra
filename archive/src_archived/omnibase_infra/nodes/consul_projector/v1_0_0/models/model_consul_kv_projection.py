#!/usr/bin/env python3


from pydantic import BaseModel, Field

from .model_consul_kv_details import ModelConsulKvDetails
from .model_consul_kv_summary import ModelConsulKvSummary


class ModelConsulKVProjection(BaseModel):
    """KV state projection result model."""

    key_summary: ModelConsulKvSummary = Field(
        ..., description="Strongly typed KV summary details",
    )
    key_details: list[ModelConsulKvDetails] | None = Field(
        None, description="List of strongly typed KV key details",
    )
