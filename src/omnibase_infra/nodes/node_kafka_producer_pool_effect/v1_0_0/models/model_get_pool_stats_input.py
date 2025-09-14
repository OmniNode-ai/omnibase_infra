"""Get pool stats input model for Kafka producer pool EFFECT node."""

from pydantic import BaseModel


class ModelGetPoolStatsInput(BaseModel):
    """Input model for get_pool_stats operation (empty)."""
    pass