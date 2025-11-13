"""AsyncMetadataStampingClient for MetadataStampingService integration.

Provides async client for:
    - Hash generation (BLAKE3)
    - Metadata stamp creation
    - Stamp validation
    - Batch operations
    - Health monitoring
"""

import logging
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .base_client import BaseServiceClient, ClientError

logger = logging.getLogger(__name__)


# Request/Response Models


class HashGenerationRequest(BaseModel):
    """Request model for hash generation."""

    file_data: bytes = Field(..., description="File data to hash")
    namespace: str = Field(default="default", description="Namespace for the hash")


class HashGenerationResponse(BaseModel):
    """Response model for hash generation."""

    status: str
    data: dict[str, Any]
    message: Optional[str] = None


class StampRequest(BaseModel):
    """Request model for stamp creation."""

    file_hash: str = Field(..., description="File hash (BLAKE3)")
    file_path: str = Field(..., description="File path")
    file_size: int = Field(..., description="File size in bytes")
    content_type: Optional[str] = Field(None, description="Content type")
    stamp_data: dict[str, Any] = Field(..., description="Stamp metadata")
    namespace: str = Field(
        default="omninode.services.metadata", description="Namespace"
    )


class StampResponse(BaseModel):
    """Response model for stamp creation."""

    status: str
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None


class ValidationRequest(BaseModel):
    """Request model for stamp validation."""

    content: str = Field(..., description="Content containing stamps to validate")
    namespace: Optional[str] = Field(None, description="Namespace filter")


class ValidationResponse(BaseModel):
    """Response model for stamp validation."""

    status: str
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class BatchStampRequest(BaseModel):
    """Request model for batch stamping."""

    items: list[StampRequest] = Field(..., description="Items to stamp")
    namespace: str = Field(
        default="omninode.services.metadata", description="Namespace"
    )


class AsyncMetadataStampingClient(BaseServiceClient):
    """Async client for MetadataStampingService.

    Features:
        - BLAKE3 hash generation
        - Metadata stamp creation with O.N.E. v0.1 compliance
        - Stamp validation
        - Batch operations
        - Circuit breaker protection
        - Retry logic with exponential backoff
        - Correlation ID propagation

    Example:
        async with AsyncMetadataStampingClient("http://192.168.86.200:8057") as client:
            hash_result = await client.generate_hash(
                file_data=b"content",
                correlation_id=correlation_id
            )

            stamp = await client.create_stamp(
                file_hash=hash_result["hash"],
                file_path="/path/to/file",
                file_size=len(file_data),
                stamp_data={"key": "value"},
                correlation_id=correlation_id
            )
    """

    def __init__(
        self,
        base_url: str = "http://192.168.86.200:8057",  # Remote Metadata Stamping Service (via hostname resolution)
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize MetadataStamping client.

        Args:
            base_url: Base URL for MetadataStampingService (defaults to remote infrastructure at 192.168.86.200:8057)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        super().__init__(
            base_url=base_url,
            service_name="MetadataStampingService",
            timeout=timeout,
            max_retries=max_retries,
        )

    async def _validate_connection(self) -> bool:
        """Validate connection to MetadataStampingService.

        Returns:
            True if service is reachable and healthy
        """
        try:
            health = await self.health_check()
            return health.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    async def generate_hash(
        self,
        file_data: bytes,
        namespace: str = "default",
        correlation_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """Generate BLAKE3 hash for file data.

        Args:
            file_data: File data bytes to hash
            namespace: Namespace for the hash operation
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Dictionary with hash result:
            {
                "hash": "blake3_hash_string",
                "execution_time_ms": 1.23,
                "file_size_bytes": 1024,
                "performance_grade": "A"
            }

        Raises:
            ClientError: If hash generation fails
        """
        try:
            # Use multipart form data for file upload
            files = {"file": ("data", file_data, "application/octet-stream")}
            params = {"namespace": namespace}

            response = await self._make_request_with_retry(
                method="POST",
                endpoint="/hash",
                correlation_id=correlation_id,
                files=files,
                params=params,
            )

            result = response.json()

            if result.get("status") == "success":
                return result.get("data", {})
            else:
                raise ClientError(
                    f"Hash generation failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(
                f"Hash generation failed: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "file_size": len(file_data),
                },
            )
            raise

    async def create_stamp(
        self,
        file_hash: str,
        file_path: str,
        file_size: int,
        stamp_data: dict[str, Any],
        content_type: Optional[str] = None,
        namespace: str = "omninode.services.metadata",
        correlation_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """Create metadata stamp for a file.

        Args:
            file_hash: BLAKE3 hash of the file
            file_path: Path to the file
            file_size: File size in bytes
            stamp_data: Additional stamp metadata
            content_type: Optional content type
            namespace: Namespace for the stamp
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Dictionary with stamp details:
            {
                "stamp_id": "uuid",
                "file_hash": "blake3_hash",
                "stamped_content": "...",
                "stamp": "...",
                "created_at": "2025-10-02T...",
                "op_id": "uuid",
                "namespace": "omninode.services.metadata",
                "version": 1
            }

        Raises:
            ClientError: If stamp creation fails
        """
        try:
            request = StampRequest(
                file_hash=file_hash,
                file_path=file_path,
                file_size=file_size,
                content_type=content_type,
                stamp_data=stamp_data,
                namespace=namespace,
            )

            response = await self._make_request_with_retry(
                method="POST",
                endpoint="/stamp",
                correlation_id=correlation_id,
                json=request.model_dump(mode="json"),
            )

            result = response.json()

            if result.get("status") == "success":
                return result.get("data", {})
            else:
                raise ClientError(
                    f"Stamp creation failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(
                f"Stamp creation failed: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "file_hash": file_hash,
                    "namespace": namespace,
                },
            )
            raise

    async def validate_stamp(
        self,
        content: str,
        namespace: Optional[str] = None,
        correlation_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """Validate stamps in content.

        Args:
            content: Content containing stamps to validate
            namespace: Optional namespace filter
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Dictionary with validation results:
            {
                "validation_result": bool,
                "stamps_found": int,
                "valid_stamps": int,
                "invalid_stamps": int,
                "details": [...]
            }

        Raises:
            ClientError: If validation fails
        """
        try:
            request = ValidationRequest(
                content=content,
                namespace=namespace,
            )

            response = await self._make_request_with_retry(
                method="POST",
                endpoint="/validate",
                correlation_id=correlation_id,
                json=request.model_dump(mode="json"),
            )

            result = response.json()

            if result.get("status") in ("success", "partial"):
                return result.get("data", {})
            else:
                raise ClientError(
                    f"Stamp validation failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(
                f"Stamp validation failed: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "namespace": namespace,
                },
            )
            raise

    async def get_stamp(
        self,
        file_hash: str,
        namespace: Optional[str] = None,
        correlation_id: Optional[UUID] = None,
    ) -> Optional[dict[str, Any]]:
        """Retrieve metadata stamp by file hash.

        Args:
            file_hash: BLAKE3 hash to lookup
            namespace: Optional namespace filter
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Stamp details or None if not found

        Raises:
            ClientError: If retrieval fails
        """
        try:
            params = {}
            if namespace:
                params["namespace"] = namespace

            response = await self._make_request_with_retry(
                method="GET",
                endpoint=f"/stamp/{file_hash}",
                correlation_id=correlation_id,
                allow_statuses=(404,),
                params=params,
            )

            # Handle 404 as "not found" rather than error
            if response.status_code == 404:
                return None

            result = response.json()

            if result.get("status") == "success":
                return result.get("data")
            else:
                # Other non-success cases
                return None

        except Exception as e:
            logger.error(
                f"Stamp retrieval failed: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "file_hash": file_hash,
                    "namespace": namespace,
                },
            )
            raise

    async def batch_create_stamps(
        self,
        items: list[dict[str, Any]],
        namespace: str = "omninode.services.metadata",
        correlation_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """Create multiple stamps in a batch operation.

        Args:
            items: List of stamp requests (each with file_hash, file_path, etc.)
            namespace: Namespace for all stamps
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Dictionary with batch results:
            {
                "total": int,
                "successful": int,
                "failed": int,
                "results": [...]
            }

        Raises:
            ClientError: If batch operation fails
        """
        try:
            stamp_requests = [
                StampRequest(**item, namespace=namespace) for item in items
            ]

            request = BatchStampRequest(
                items=stamp_requests,
                namespace=namespace,
            )

            response = await self._make_request_with_retry(
                method="POST",
                endpoint="/batch",
                correlation_id=correlation_id,
                json=request.model_dump(mode="json"),
            )

            result = response.json()

            if result.get("status") in ("success", "partial"):
                return result.get("data", {})
            else:
                raise ClientError(
                    f"Batch operation failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(
                f"Batch stamp creation failed: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "batch_size": len(items),
                    "namespace": namespace,
                },
            )
            raise

    async def get_service_metrics(
        self,
        correlation_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """Get service performance metrics.

        Args:
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Dictionary with service metrics

        Raises:
            ClientError: If metrics retrieval fails
        """
        try:
            response = await self._make_request_with_retry(
                method="GET",
                endpoint="/metrics",
                correlation_id=correlation_id,
            )

            return response.json()

        except Exception as e:
            logger.error(
                f"Metrics retrieval failed: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                },
            )
            raise
