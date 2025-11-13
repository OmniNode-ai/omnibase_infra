"""Data models for metadata stamping service."""

from .database import (
    HashMetricRecord,
    LegacyMetadataStampRecord,
    MetadataStampRecord,
    ProtocolHandlerRecord,
)
from .requests import StampOptions, StampRequest, StampType, ValidationOptions
from .responses import (
    HashResponse,
    HealthResponse,
    PerformanceMetrics,
    StampResponse,
    ValidationResponse,
)

__all__ = [
    # Request models
    "StampRequest",
    "StampOptions",
    "ValidationOptions",
    "StampType",
    # Response models
    "StampResponse",
    "ValidationResponse",
    "HashResponse",
    "HealthResponse",
    "PerformanceMetrics",
    # Database models
    "MetadataStampRecord",
    "LegacyMetadataStampRecord",
    "ProtocolHandlerRecord",
    "HashMetricRecord",
]
