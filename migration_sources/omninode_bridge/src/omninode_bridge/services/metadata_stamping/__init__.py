"""MetadataStampingService - Phase 1 Implementation.

High-performance metadata stamping service with BLAKE3 hashing
and omnibase_core protocol compliance.

Updated with compliance fields from omnibase_3 and ai-dev patterns:
- intelligence_data: JSONB field for enhanced processing intelligence
- version: Schema version tracking (default: 1)
- op_id: Unique operation identifier for traceability
- namespace: Service namespace (default: omninode.services.metadata)
- metadata_version: Metadata format version (default: 0.1)

Backward Compatibility:
- Legacy interfaces maintained through compatibility module
- Existing code continues to work without modifications
- Optional migration to new compliance-aware interfaces
"""

from .compatibility import (
    BackwardCompatibleClient,
    create_stamp_legacy,
    ensure_compliance_fields,
    get_stamp_legacy,
    migrate_legacy_client,
    strip_compliance_fields,
)
from .database.client import MetadataStampingPostgresClient
from .service import MetadataStampingService

__all__ = [
    # Core service
    "MetadataStampingService",
    # Database client
    "MetadataStampingPostgresClient",
    # Backward compatibility
    "BackwardCompatibleClient",
    "migrate_legacy_client",
    "create_stamp_legacy",
    "get_stamp_legacy",
    # Utilities
    "ensure_compliance_fields",
    "strip_compliance_fields",
]
