"""Database components for metadata stamping service."""

from .client import DatabaseConfig, MetadataStampingPostgresClient
from .operations import (
    AdvancedMetadataOperations,
    BatchOperationResult,
    BatchUpsertResult,
    ConflictResolutionStrategy,
    UpsertResult,
    VersionedMetadata,
)

__all__ = [
    "MetadataStampingPostgresClient",
    "DatabaseConfig",
    "AdvancedMetadataOperations",
    "ConflictResolutionStrategy",
    "BatchOperationResult",
    "UpsertResult",
    "BatchUpsertResult",
    "VersionedMetadata",
]
