"""Backward compatibility layer for MetadataStampingService.

This module provides wrapper functions and utilities to ensure existing
code continues to work with the new compliance fields while maintaining
the same API surface.
"""

import logging
import uuid
from typing import Optional

from .database.client import MetadataStampingPostgresClient
from .models.database import LegacyMetadataStampRecord, MetadataStampRecord

logger = logging.getLogger(__name__)


class BackwardCompatibleClient:
    """Backward compatible wrapper for MetadataStampingPostgresClient.

    This class provides the same interface as the original client but
    automatically handles the new compliance fields with sensible defaults.
    """

    def __init__(self, client: MetadataStampingPostgresClient):
        """Initialize the compatibility wrapper.

        Args:
            client: The underlying database client
        """
        self._client = client

    async def create_metadata_stamp_legacy(
        self,
        file_hash: str,
        file_path: str,
        file_size: int,
        content_type: str,
        stamp_data: dict,
        protocol_version: str = "1.0",
    ) -> dict:
        """Create metadata stamp with legacy interface.

        This method maintains the exact same signature as the original
        create_metadata_stamp method, automatically providing compliance
        field defaults.

        Args:
            file_hash: BLAKE3 hash of the file
            file_path: Path to the file
            file_size: Size of the file in bytes
            content_type: MIME content type
            stamp_data: Stamp metadata as dictionary
            protocol_version: Protocol version

        Returns:
            Created stamp information (legacy format)
        """
        # Call the new method with compliance defaults
        result = await self._client.create_metadata_stamp(
            file_hash=file_hash,
            file_path=file_path,
            file_size=file_size,
            content_type=content_type,
            stamp_data=stamp_data,
            protocol_version=protocol_version,
            intelligence_data={},  # Default empty intelligence
            version=1,  # Default version
            op_id=str(uuid.uuid4()),  # Generate operation ID
            namespace="omninode.services.metadata",  # Default namespace
            metadata_version="0.1",  # Default metadata version
        )

        # Return legacy format (without compliance fields for backward compatibility)
        return {
            "id": result["id"],
            "file_hash": result["file_hash"],
            "created_at": result["created_at"],
        }

    async def get_metadata_stamp_legacy(self, file_hash: str) -> Optional[dict]:
        """Retrieve metadata stamp with legacy interface.

        This method returns the stamp data in the legacy format,
        filtering out the new compliance fields.

        Args:
            file_hash: BLAKE3 hash to look up

        Returns:
            Stamp record in legacy format, None if not found
        """
        result = await self._client.get_metadata_stamp(file_hash)

        if not result:
            return None

        # Create legacy record model to filter compliance fields
        full_record = MetadataStampRecord(**result)
        legacy_record = LegacyMetadataStampRecord.from_full_record(full_record)

        return legacy_record.model_dump()

    async def batch_insert_stamps_legacy(self, stamps_data: list[dict]) -> list[str]:
        """Batch insert stamps with legacy interface.

        This method accepts stamp data in the legacy format and
        automatically adds compliance field defaults.

        Args:
            stamps_data: List of stamp data dictionaries (legacy format)

        Returns:
            List of inserted stamp IDs
        """
        # Transform legacy stamp data to include compliance fields
        enhanced_stamps_data = []

        for stamp in stamps_data:
            enhanced_stamp = {
                **stamp,  # Original fields
                # Add compliance defaults if not present
                "intelligence_data": stamp.get("intelligence_data", {}),
                "version": stamp.get("version", 1),
                "op_id": stamp.get("op_id", str(uuid.uuid4())),
                "namespace": stamp.get("namespace", "omninode.services.metadata"),
                "metadata_version": stamp.get("metadata_version", "0.1"),
            }
            enhanced_stamps_data.append(enhanced_stamp)

        return await self._client.batch_insert_stamps(enhanced_stamps_data)

    # Delegate other methods unchanged
    async def record_performance_metric(self, *args, **kwargs):
        """Delegate performance metric recording (unchanged interface)."""
        return await self._client.record_performance_metric(*args, **kwargs)

    async def get_performance_statistics(self):
        """Delegate performance statistics (unchanged interface)."""
        return await self._client.get_performance_statistics()

    async def health_check(self):
        """Delegate health check (unchanged interface)."""
        return await self._client.health_check()

    async def close(self):
        """Delegate connection closure (unchanged interface)."""
        return await self._client.close()


def ensure_compliance_fields(stamp_data: dict) -> dict:
    """Ensure stamp data has all required compliance fields.

    This utility function can be used to upgrade legacy stamp data
    dictionaries to include the new compliance fields with defaults.

    Args:
        stamp_data: Original stamp data dictionary

    Returns:
        Enhanced stamp data with compliance fields
    """
    return {
        **stamp_data,
        # Add compliance fields with defaults if not present
        "intelligence_data": stamp_data.get("intelligence_data", {}),
        "version": stamp_data.get("version", 1),
        "op_id": stamp_data.get("op_id", str(uuid.uuid4())),
        "namespace": stamp_data.get("namespace", "omninode.services.metadata"),
        "metadata_version": stamp_data.get("metadata_version", "0.1"),
    }


def strip_compliance_fields(stamp_data: dict) -> dict:
    """Strip compliance fields from stamp data for legacy compatibility.

    This utility function removes the new compliance fields from
    stamp data dictionaries to maintain backward compatibility.

    Args:
        stamp_data: Enhanced stamp data with compliance fields

    Returns:
        Legacy stamp data without compliance fields
    """
    # List of compliance fields to remove
    compliance_fields = {
        "intelligence_data",
        "version",
        "op_id",
        "namespace",
        "metadata_version",
    }

    return {
        key: value for key, value in stamp_data.items() if key not in compliance_fields
    }


def migrate_legacy_client(
    client: MetadataStampingPostgresClient,
) -> BackwardCompatibleClient:
    """Create backward compatible wrapper for existing client.

    This function provides an easy migration path for existing code
    that uses the MetadataStampingPostgresClient directly.

    Args:
        client: Existing database client

    Returns:
        Backward compatible wrapper
    """
    logger.info(
        "Creating backward compatible wrapper for MetadataStampingPostgresClient"
    )
    return BackwardCompatibleClient(client)


# Alias functions for even simpler migration
async def create_stamp_legacy(
    client: MetadataStampingPostgresClient,
    file_hash: str,
    file_path: str,
    file_size: int,
    content_type: str,
    stamp_data: dict,
    protocol_version: str = "1.0",
) -> dict:
    """Legacy-compatible stamp creation function.

    Direct function interface for minimal code changes.
    """
    compat_client = BackwardCompatibleClient(client)
    return await compat_client.create_metadata_stamp_legacy(
        file_hash, file_path, file_size, content_type, stamp_data, protocol_version
    )


async def get_stamp_legacy(
    client: MetadataStampingPostgresClient, file_hash: str
) -> Optional[dict]:
    """Legacy-compatible stamp retrieval function.

    Direct function interface for minimal code changes.
    """
    compat_client = BackwardCompatibleClient(client)
    return await compat_client.get_metadata_stamp_legacy(file_hash)
