"""Get health input model for Kafka producer pool EFFECT node."""

from pydantic import BaseModel


class ModelGetHealthInput(BaseModel):
    """Input model for get_health operation (empty)."""
    pass