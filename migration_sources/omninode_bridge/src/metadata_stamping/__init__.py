"""
MetadataStampingService - Phase 3: Distributed Architecture and Scaling

High-performance microservice for generating cryptographic metadata stamps
with distributed architecture capabilities, auto-scaling, and multi-region support.
"""

__version__ = "3.0.0"
__author__ = "OmniNode Team"
__email__ = "team@omninode.ai"
__license__ = "MIT"

from .distributed import circuit_breaker, sharding
from .telemetry import opentelemetry

__all__ = [
    "sharding",
    "circuit_breaker",
    "opentelemetry",
]
