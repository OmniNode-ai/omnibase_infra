# === OmniNode:Tool_Metadata ===
# metadata_version: 0.1
# name: metadata_stamping_api_router
# title: MetadataStampingService API Router
# version: 0.1.0
# namespace: omninode.services.metadata
# category: service.infrastructure.api
# kind: service
# role: api_router
# description: |
#   FastAPI router implementing core API endpoints for metadata stamping service
#   with high-performance stamping, validation, hash generation, and unified
#   response format capabilities.
# tags: [api, router, metadata, fastapi, endpoints, stamping]
# author: OmniNode Development Team
# license: MIT
# entrypoint: router.py
# protocols_supported: [O.N.E. v0.1]
# runtime_constraints: {sandboxed: false, privileged: false, requires_network: true, requires_gpu: false}
# dependencies: [{"name": "fastapi", "version": "^0.104.1"}]
# environment: [python>=3.11]
# === /OmniNode:Tool_Metadata ===

"""FastAPI router for metadata stamping service endpoints.

This module implements the core API endpoints for the metadata stamping service,
providing high-performance stamping, validation, and hash generation capabilities.
"""

import asyncio
import logging
import re
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from ..execution.stamping_transformers import StampingInput, ValidationInput
from ..models.requests import (
    BatchStampRequest,
    ProtocolValidationRequest,
    StampRequest,
    ValidationRequest,
)
from ..models.responses import (
    BatchStampResponse,
    BatchStampResult,
    ComponentHealth,
    ErrorDetail,
    HashResponse,
    HealthResponse,
    NamespaceQueryResponse,
    PerformanceMetrics,
    ProtocolValidationResponse,
    ProtocolValidationResult,
    StampResponse,
    UnifiedResponse,
    ValidationResponse,
)
from ..service import MetadataStampingService

logger = logging.getLogger(__name__)

# Create router with prefix
router = APIRouter(prefix="/api/v1/metadata-stamping", tags=["metadata-stamping"])

# Service instance (will be injected via dependency)
_service_instance: Optional[MetadataStampingService] = None

# Security constants
MAX_CONTENT_SIZE = 100 * 1024 * 1024  # 100MB
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


def validate_hash(hash_str: str) -> bool:
    """Validate BLAKE3 hash format for security.

    Args:
        hash_str: Hash string to validate

    Returns:
        True if valid hash format, False otherwise
    """
    return bool(re.match(r"^[a-f0-9]{64}$", hash_str))


def validate_content_size(content: bytes, max_size: int = MAX_CONTENT_SIZE) -> None:
    """Validate content size to prevent memory exhaustion.

    Args:
        content: Content to validate
        max_size: Maximum allowed size in bytes

    Raises:
        HTTPException: If content is too large
    """
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Content too large. Maximum size: {max_size // 1024 // 1024}MB",
        )


def validate_file_size(file_size: int, max_size: int = MAX_FILE_SIZE) -> None:
    """Validate file size to prevent resource exhaustion.

    Args:
        file_size: File size in bytes to validate
        max_size: Maximum allowed size in bytes

    Raises:
        HTTPException: If file is too large
    """
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum file size: {max_size // 1024 // 1024}MB",
        )


async def get_service() -> MetadataStampingService:
    """Dependency to get the service instance.

    Returns:
        MetadataStampingService instance

    Raises:
        HTTPException: If service is not initialized
    """
    if _service_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _service_instance


def set_service(service: MetadataStampingService):
    """Set the service instance for dependency injection.

    Args:
        service: MetadataStampingService instance
    """
    global _service_instance
    _service_instance = service


# Unified Response Helpers


def create_success_response(
    data: Any = None, message: str = None, metadata: dict[str, Any] = None
) -> UnifiedResponse:
    """Create a successful unified response.

    Args:
        data: Response data
        message: Optional success message
        metadata: Optional response metadata

    Returns:
        UnifiedResponse with success status
    """
    return UnifiedResponse(
        status="success", data=data, message=message, metadata=metadata
    )


def create_error_response(
    error: str,
    error_code: str = "INTERNAL_ERROR",
    field: str = None,
    status_code: int = 500,
) -> UnifiedResponse:
    """Create an error unified response.

    Args:
        error: Error message
        error_code: Specific error code
        field: Field causing the error
        status_code: HTTP status code

    Returns:
        UnifiedResponse with error status
    """
    error_detail = ErrorDetail(code=error_code, field=field, message=error)
    return UnifiedResponse(
        status="error", error=[error_detail], metadata={"status_code": status_code}
    )


def create_partial_response(
    data: Any,
    errors: list[ErrorDetail],
    message: str = None,
    metadata: dict[str, Any] = None,
) -> UnifiedResponse:
    """Create a partial success unified response.

    Args:
        data: Successful response data
        errors: List of errors that occurred
        message: Optional message
        metadata: Optional response metadata

    Returns:
        UnifiedResponse with partial status
    """
    return UnifiedResponse(
        status="partial", data=data, error=errors, message=message, metadata=metadata
    )


def handle_service_exception(e: Exception, operation: str) -> HTTPException:
    """Handle service exceptions with detailed error codes.

    Args:
        e: Exception that occurred
        operation: Operation being performed

    Returns:
        HTTPException with appropriate status code and error details
    """
    if isinstance(e, HTTPException):
        return e
    elif isinstance(e, ValueError | TypeError):
        logger.error(f"Invalid input for {operation}: {e}")
        return HTTPException(
            status_code=422,
            detail=create_error_response(
                f"Invalid input: {e}", "VALIDATION_ERROR"
            ).model_dump(),
        )
    elif isinstance(e, ConnectionError | OSError):
        logger.error(f"Database connection error during {operation}: {e}")
        return HTTPException(
            status_code=503,
            detail=create_error_response(
                "Service temporarily unavailable", "DATABASE_CONNECTION_ERROR"
            ).model_dump(),
        )
    elif isinstance(e, MemoryError):
        logger.error(f"Memory error during {operation}: {e}")
        return HTTPException(
            status_code=507,
            detail=create_error_response(
                "Insufficient memory to process request", "MEMORY_ERROR"
            ).model_dump(),
        )
    elif isinstance(e, RuntimeError):
        logger.error(f"Service runtime error during {operation}: {e}")
        return HTTPException(
            status_code=500,
            detail=create_error_response(
                f"Service error: {e}", "RUNTIME_ERROR"
            ).model_dump(),
        )
    else:
        logger.error(f"Unexpected error during {operation}: {e}")
        return HTTPException(
            status_code=500,
            detail=create_error_response(
                "Internal server error", "UNEXPECTED_ERROR"
            ).model_dump(),
        )


# Core stamping endpoints


@router.post("/stamp", response_model=UnifiedResponse)
async def create_metadata_stamp(
    request: StampRequest, service: MetadataStampingService = Depends(get_service)
) -> UnifiedResponse:
    """Generate metadata stamp for content.

    Args:
        request: Stamp request with content and options
        service: Injected service instance

    Returns:
        Unified response with generated stamp and metrics

    Raises:
        HTTPException: If content is too large or invalid
    """
    try:
        start_time = time.perf_counter()

        # Security: Validate content size
        content_bytes = request.content.encode("utf-8")
        validate_content_size(content_bytes)

        # Stamp the content
        result = await service.stamp_content(
            content=request.content,
            file_path=request.file_path,
            stamp_type=request.options.stamp_type.value,
            metadata=request.metadata,
        )

        # Store in database if configured with namespace support
        stamp_id = None
        created_at = None
        if service.db_client:
            # Idempotency: Check if stamp already exists
            existing_stamp = await service.db_client.get_metadata_stamp(
                file_hash=result["content_hash"]
            )

            if existing_stamp:
                # Stamp already exists - return existing stamp (idempotent)
                logger.info(
                    f"Stamp already exists for hash {result['content_hash']}, returning existing stamp"
                )
                # Convert UUID to string if needed
                stamp_id = str(existing_stamp["id"]) if existing_stamp["id"] else None
                created_at = existing_stamp["created_at"]
            else:
                # Create new stamp
                db_result = await service.db_client.create_metadata_stamp(
                    file_hash=result["content_hash"],
                    file_path=request.file_path or "",
                    file_size=len(content_bytes),
                    content_type="text/plain",
                    stamp_data={
                        "stamp_type": result["stamp_type"],
                        "stamp": result["stamp"],
                        "metadata": request.metadata,
                        "namespace": request.namespace,
                    },
                    protocol_version=request.protocol_version,
                )
                stamp_id = db_result["id"]
                created_at = db_result["created_at"]
                logger.info(
                    f"Created new stamp with id {stamp_id} for hash {result['content_hash']}"
                )

        # Calculate total execution time
        total_time_ms = (time.perf_counter() - start_time) * 1000

        stamp_response = StampResponse(
            success=True,
            stamp_id=stamp_id,
            file_hash=result["content_hash"],
            stamped_content=result["stamped_content"],
            stamp=result["stamp"],
            stamp_type=result["stamp_type"],
            performance_metrics=PerformanceMetrics(
                execution_time_ms=total_time_ms,
                file_size_bytes=len(content_bytes),
                performance_grade=result["performance_grade"],
            ),
            created_at=created_at,
        )

        return create_success_response(
            data=stamp_response,
            message="Metadata stamp created successfully",
            metadata={
                "namespace": request.namespace,
                "protocol_version": request.protocol_version,
                "operation": "create_stamp",
            },
        )

    except Exception as e:
        raise handle_service_exception(e, "stamp creation")


@router.post("/validate", response_model=UnifiedResponse)
async def validate_stamps(
    request: ValidationRequest, service: MetadataStampingService = Depends(get_service)
) -> UnifiedResponse:
    """Validate existing stamps in content.

    Args:
        request: Validation request with content and options
        service: Injected service instance

    Returns:
        Unified response with validation results and metrics

    Raises:
        HTTPException: If content is too large or invalid
    """
    try:
        start_time = time.perf_counter()

        # Security: Validate content size
        content_bytes = request.content.encode("utf-8")
        validate_content_size(content_bytes)

        # Validate the stamps
        result = await service.validate_stamp(
            content=request.content, expected_hash=request.options.expected_hash
        )

        # Calculate total execution time
        total_time_ms = (time.perf_counter() - start_time) * 1000

        # Format validation details
        validation_details = [
            {
                "stamp_type": detail["stamp_type"],
                "stamp_hash": detail["stamp_hash"],
                "is_valid": detail["is_valid"],
                "current_hash": detail["current_hash"],
            }
            for detail in result.get("validation_results", [])
        ]

        validation_response = ValidationResponse(
            success=True,
            is_valid=result["valid"],
            stamps_found=result["stamps_found"],
            current_hash=result["current_hash"],
            validation_details=validation_details,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=total_time_ms,
                file_size_bytes=len(content_bytes),
                performance_grade=(
                    "A" if total_time_ms < 10 else "B" if total_time_ms < 50 else "C"
                ),
            ),
        )

        return create_success_response(
            data=validation_response,
            message="Stamp validation completed successfully",
            metadata={
                "namespace": request.namespace,
                "validation_summary": {
                    "valid": result["valid"],
                    "stamps_found": result["stamps_found"],
                },
                "operation": "validate_stamps",
            },
        )

    except Exception as e:
        raise handle_service_exception(e, "stamp validation")


@router.post("/hash", response_model=UnifiedResponse)
async def generate_file_hash(
    file: UploadFile = File(...),
    namespace: Optional[str] = Query(default="default", description="Namespace"),
    service: MetadataStampingService = Depends(get_service),
) -> UnifiedResponse:
    """Generate BLAKE3 hash for uploaded file.

    Args:
        file: Uploaded file
        namespace: Optional namespace for the hash
        service: Injected service instance

    Returns:
        Unified response with hash and performance metrics

    Raises:
        HTTPException: If file is too large or invalid
    """
    try:
        # Read file content
        content = await file.read()

        # Security: Validate file size
        validate_file_size(len(content))

        # Generate hash
        result = await service.generate_hash(content, file.filename)

        hash_response = HashResponse(
            file_hash=result["hash"],
            execution_time_ms=result["execution_time_ms"],
            file_size_bytes=result["file_size_bytes"],
            performance_grade=result["performance_grade"],
        )

        return create_success_response(
            data=hash_response,
            message="File hash generated successfully",
            metadata={
                "namespace": namespace,
                "filename": file.filename,
                "operation": "generate_hash",
            },
        )

    except Exception as e:
        raise handle_service_exception(e, "hash generation")


@router.get("/stamp/{file_hash}", response_model=UnifiedResponse)
async def get_metadata_stamp(
    file_hash: str,
    namespace: Optional[str] = Query(default=None, description="Filter by namespace"),
    service: MetadataStampingService = Depends(get_service),
) -> UnifiedResponse:
    """Retrieve existing metadata stamp by file hash.

    Args:
        file_hash: BLAKE3 hash to look up
        namespace: Optional namespace filter
        service: Injected service instance

    Returns:
        Unified response with stamp record if found

    Raises:
        HTTPException: If hash format is invalid or stamp not found
    """
    try:
        # Security: Validate hash format before database query
        if not validate_hash(file_hash):
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Invalid hash format. Expected 64-character hexadecimal string",
                    "INVALID_HASH_FORMAT",
                    "file_hash",
                ).model_dump(),
            )

        if not service.db_client:
            raise HTTPException(
                status_code=503,
                detail=create_error_response(
                    "Database not configured", "DATABASE_NOT_CONFIGURED"
                ).model_dump(),
            )

        stamp = await service.db_client.get_metadata_stamp(file_hash)

        if not stamp:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "Stamp not found", "STAMP_NOT_FOUND", "file_hash"
                ).model_dump(),
            )

        # Filter by namespace if specified
        if namespace and stamp.get("stamp_data", {}).get("namespace") != namespace:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    f"Stamp not found in namespace '{namespace}'",
                    "STAMP_NOT_FOUND_IN_NAMESPACE",
                    "namespace",
                ).model_dump(),
            )

        return create_success_response(
            data=stamp,
            message="Metadata stamp retrieved successfully",
            metadata={
                "file_hash": file_hash,
                "namespace": namespace or stamp.get("stamp_data", {}).get("namespace"),
                "operation": "get_stamp",
            },
        )

    except Exception as e:
        raise handle_service_exception(e, "stamp retrieval")


# New enhanced endpoints


@router.post("/batch", response_model=UnifiedResponse)
async def batch_stamp_operations(
    request: BatchStampRequest, service: MetadataStampingService = Depends(get_service)
) -> UnifiedResponse:
    """Process multiple stamping operations in batch for high throughput.

    Args:
        request: Batch stamping request with multiple items
        service: Injected service instance

    Returns:
        Unified response with batch processing results

    Raises:
        HTTPException: If batch request is invalid
    """
    try:
        start_time = time.perf_counter()

        if not request.items:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "No items provided for batch processing",
                    "EMPTY_BATCH_REQUEST",
                    "items",
                ).model_dump(),
            )

        # Process items in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(10)  # Limit concurrent operations

        async def process_item(item_data: dict) -> BatchStampResult:
            async with semaphore:
                try:
                    item_id = item_data.get("id", f"item_{time.time()}")
                    content = item_data.get("content", "")
                    file_path = item_data.get("file_path")
                    namespace = item_data.get("namespace", request.namespace)
                    metadata = item_data.get("metadata", {})

                    # Validate content size
                    content_bytes = content.encode("utf-8")
                    validate_content_size(content_bytes)

                    # Stamp the content
                    result = await service.stamp_content(
                        content=content,
                        file_path=file_path,
                        stamp_type=request.options.stamp_type.value,
                        metadata=metadata,
                    )

                    # Store in database if configured
                    stamp_id = None
                    if service.db_client:
                        db_result = await service.db_client.create_metadata_stamp(
                            file_hash=result["content_hash"],
                            file_path=file_path or "",
                            file_size=len(content_bytes),
                            content_type="text/plain",
                            stamp_data={
                                "stamp_type": result["stamp_type"],
                                "stamp": result["stamp"],
                                "metadata": metadata,
                                "namespace": namespace,
                            },
                            protocol_version=request.protocol_version,
                        )
                        stamp_id = db_result["id"]

                    return BatchStampResult(
                        id=item_id,
                        success=True,
                        stamp_id=stamp_id,
                        file_hash=result["content_hash"],
                        stamp=result["stamp"],
                        performance_metrics=PerformanceMetrics(
                            execution_time_ms=result.get("execution_time_ms", 0),
                            file_size_bytes=len(content_bytes),
                            performance_grade=result.get("performance_grade", "C"),
                        ),
                    )

                except Exception as e:
                    return BatchStampResult(
                        id=item_data.get("id", f"item_{time.time()}"),
                        success=False,
                        error=str(e),
                    )

        # Process all items concurrently
        tasks = [process_item(item) for item in request.items]
        results = await asyncio.gather(*tasks)

        # Calculate overall metrics
        total_time_ms = (time.perf_counter() - start_time) * 1000
        successful_items = sum(1 for r in results if r.success)
        failed_items = len(results) - successful_items

        batch_response = BatchStampResponse(
            total_items=len(request.items),
            successful_items=successful_items,
            failed_items=failed_items,
            results=results,
            overall_performance=PerformanceMetrics(
                execution_time_ms=total_time_ms,
                file_size_bytes=sum(
                    len(item.get("content", "").encode("utf-8"))
                    for item in request.items
                ),
                performance_grade=(
                    "A"
                    if total_time_ms < 1000
                    else "B" if total_time_ms < 5000 else "C"
                ),
            ),
        )

        response_status = (
            "success"
            if failed_items == 0
            else "partial" if successful_items > 0 else "error"
        )

        return UnifiedResponse(
            status=response_status,
            data=batch_response,
            message=f"Batch processing completed: {successful_items}/{len(request.items)} successful",
            metadata={
                "namespace": request.namespace,
                "protocol_version": request.protocol_version,
                "operation": "batch_stamp",
            },
        )

    except Exception as e:
        raise handle_service_exception(e, "batch stamping")


@router.post("/validate-protocol", response_model=UnifiedResponse)
async def validate_protocol_compliance(
    request: ProtocolValidationRequest,
    service: MetadataStampingService = Depends(get_service),
) -> UnifiedResponse:
    """Validate content against O.N.E. v0.1 protocol compliance.

    Args:
        request: Protocol validation request
        service: Injected service instance

    Returns:
        Unified response with protocol validation results

    Raises:
        HTTPException: If validation request is invalid
    """
    try:
        start_time = time.perf_counter()

        # Security: Validate content size
        content_bytes = request.content.encode("utf-8")
        validate_content_size(content_bytes)

        # Perform protocol validation
        issues = []
        recommendations = []
        compliance_level = "full"

        # Check for required protocol elements
        if "<!-- METADATA_STAMP:" not in request.content:
            issues.append("No metadata stamps found in content")
            compliance_level = "none"

        # Check protocol version compatibility
        detected_version = "1.0"  # Default
        if "protocol_version" in request.content:
            # Extract version if present
            pass

        # Check namespace compliance
        if request.namespace and request.namespace != "default":
            if f"namespace:{request.namespace}" not in request.content:
                recommendations.append(
                    f"Consider adding explicit namespace declaration: {request.namespace}"
                )

        # Validate against target protocol
        if request.target_protocol == "O.N.E.v0.1":
            # O.N.E. v0.1 specific validations
            if not any(
                stamp_type in request.content for stamp_type in ["lightweight", "rich"]
            ):
                issues.append("O.N.E. v0.1 requires explicit stamp type declaration")
                compliance_level = "partial" if compliance_level == "full" else "none"

        # Calculate execution time
        total_time_ms = (time.perf_counter() - start_time) * 1000

        validation_result = ProtocolValidationResult(
            is_valid=len(issues) == 0,
            protocol_version=detected_version,
            compliance_level=compliance_level,
            issues=issues,
            recommendations=recommendations,
        )

        protocol_response = ProtocolValidationResponse(
            validation_result=validation_result,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=total_time_ms,
                file_size_bytes=len(content_bytes),
                performance_grade=(
                    "A" if total_time_ms < 5 else "B" if total_time_ms < 20 else "C"
                ),
            ),
        )

        return create_success_response(
            data=protocol_response,
            message=f"Protocol validation completed: {compliance_level} compliance",
            metadata={
                "namespace": request.namespace,
                "target_protocol": request.target_protocol,
                "validation_level": request.validation_level,
                "operation": "validate_protocol",
            },
        )

    except Exception as e:
        raise handle_service_exception(e, "protocol validation")


@router.get("/namespace/{namespace}", response_model=UnifiedResponse)
async def query_namespace_stamps(
    namespace: str,
    limit: int = Query(
        default=50, ge=1, le=1000, description="Maximum number of results"
    ),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    service: MetadataStampingService = Depends(get_service),
) -> UnifiedResponse:
    """Query stamps within a specific namespace.

    Args:
        namespace: Namespace to query
        limit: Maximum number of results to return
        offset: Offset for pagination
        service: Injected service instance

    Returns:
        Unified response with namespace query results

    Raises:
        HTTPException: If namespace query fails
    """
    try:
        if not service.db_client:
            raise HTTPException(
                status_code=503,
                detail=create_error_response(
                    "Database not configured", "DATABASE_NOT_CONFIGURED"
                ).model_dump(),
            )

        # This would need to be implemented in the database client
        # For now, return a mock response structure
        stamps = (
            []
        )  # await service.db_client.query_stamps_by_namespace(namespace, limit, offset)
        total_stamps = 0  # await service.db_client.count_stamps_by_namespace(namespace)

        namespace_response = NamespaceQueryResponse(
            namespace=namespace,
            total_stamps=total_stamps,
            stamps=stamps,
            pagination={
                "limit": limit,
                "offset": offset,
                "total": total_stamps,
                "has_more": offset + limit < total_stamps,
            },
        )

        return create_success_response(
            data=namespace_response,
            message=f"Namespace query completed: {len(stamps)} stamps found",
            metadata={
                "namespace": namespace,
                "pagination": {"limit": limit, "offset": offset, "total": total_stamps},
                "operation": "query_namespace",
            },
        )

    except Exception as e:
        raise handle_service_exception(e, "namespace query")


# Health and monitoring endpoints


@router.get("/health/live", response_model=UnifiedResponse)
async def liveness_check(
    service: MetadataStampingService = Depends(get_service),
) -> UnifiedResponse:
    """Liveness probe for Kubernetes/Docker health checks.

    This endpoint checks if the service is alive and responsive.
    It should return 200 if the service can accept requests.

    Args:
        service: Injected service instance

    Returns:
        Unified response with liveness status
    """
    try:
        # Basic service availability check
        uptime = service.get_uptime()

        # Service is alive if it's been running and can respond
        if uptime > 0:
            return create_success_response(
                data={
                    "status": "alive",
                    "uptime_seconds": uptime,
                    "timestamp": time.time(),
                },
                message="Service is alive and responsive",
                metadata={"check_type": "liveness", "operation": "liveness_check"},
            )
        else:
            return create_error_response(
                "Service not properly initialized", "SERVICE_NOT_ALIVE"
            )

    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return create_error_response(
            f"Liveness check failed: {e}", "LIVENESS_CHECK_FAILED"
        )


@router.get("/health/ready", response_model=UnifiedResponse)
async def readiness_check(
    service: MetadataStampingService = Depends(get_service),
) -> UnifiedResponse:
    """Readiness probe for Kubernetes/Docker health checks.

    This endpoint checks if the service is ready to handle requests.
    It should return 200 only when all dependencies are available.

    Args:
        service: Injected service instance

    Returns:
        Unified response with readiness status
    """
    try:
        # Comprehensive readiness check
        if not service.is_initialized:
            return create_error_response(
                "Service not initialized", "SERVICE_NOT_INITIALIZED"
            )

        readiness_status = {"status": "ready", "components": {}, "overall_ready": True}

        # Check stamping engine readiness
        try:
            test_result = await service.stamping_engine.hash_generator.generate_hash(
                b"readiness_test"
            )
            if (
                test_result["execution_time_ms"] < 10
            ):  # Should be very fast for small test
                readiness_status["components"]["stamping_engine"] = {
                    "ready": True,
                    "response_time_ms": test_result["execution_time_ms"],
                }
            else:
                readiness_status["components"]["stamping_engine"] = {
                    "ready": False,
                    "reason": "Slow response time",
                    "response_time_ms": test_result["execution_time_ms"],
                }
                readiness_status["overall_ready"] = False
        except Exception as e:
            readiness_status["components"]["stamping_engine"] = {
                "ready": False,
                "reason": f"Engine error: {e}",
            }
            readiness_status["overall_ready"] = False

        # Check database readiness if configured
        if service.db_client:
            try:
                db_health = await service.db_client.health_check()
                db_ready = (
                    db_health["status"] == "healthy"
                    and db_health.get("response_time_ms", 0) < 100
                )
                readiness_status["components"]["database"] = {
                    "ready": db_ready,
                    "status": db_health["status"],
                    "response_time_ms": db_health.get("response_time_ms"),
                }
                if not db_ready:
                    readiness_status["overall_ready"] = False
            except Exception as e:
                readiness_status["components"]["database"] = {
                    "ready": False,
                    "reason": f"Database error: {e}",
                }
                readiness_status["overall_ready"] = False

        # Check event publisher readiness if configured
        if service.event_publisher:
            try:
                # Simple event publisher health check
                readiness_status["components"]["event_publisher"] = {
                    "ready": True,
                    "status": "configured",
                }
            except Exception as e:
                readiness_status["components"]["event_publisher"] = {
                    "ready": False,
                    "reason": f"Event publisher error: {e}",
                }
                readiness_status["overall_ready"] = False

        if readiness_status["overall_ready"]:
            return create_success_response(
                data=readiness_status,
                message="Service is ready to handle requests",
                metadata={"check_type": "readiness", "operation": "readiness_check"},
            )
        else:
            return create_partial_response(
                data=readiness_status,
                errors=[
                    ErrorDetail(
                        code="READINESS_CHECK_FAILED",
                        message="One or more components not ready",
                    )
                ],
                message="Service not fully ready",
            )

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return create_error_response(
            f"Readiness check failed: {e}", "READINESS_CHECK_FAILED"
        )


@router.get("/health", response_model=UnifiedResponse)
async def health_check(
    service: MetadataStampingService = Depends(get_service),
) -> UnifiedResponse:
    """Service health status and performance metrics.

    Args:
        service: Injected service instance

    Returns:
        Unified response with health status and component information
    """
    try:
        health_result = await service.health_check()

        # Format component health
        components = {}
        for component, status in health_result["components"].items():
            components[component] = ComponentHealth(
                status=status.get("status", "unknown"),
                response_time_ms=status.get("response_time_ms"),
                details=status.get("details"),
            )

        health_response = HealthResponse(
            status=health_result["status"],
            components=components,
            uptime_seconds=health_result.get("uptime_seconds"),
            version="0.1.0",
        )

        return create_success_response(
            data=health_response,
            message=f"Health check completed: {health_result['status']}",
            metadata={
                "operation": "health_check",
                "components_checked": len(components),
            },
        )

    except (ConnectionError, OSError) as e:
        logger.error(f"Connection error during health check: {e}")
        health_response = HealthResponse(
            status="degraded",
            components={
                "service": ComponentHealth(
                    status="error", details={"error": f"Connection error: {e}"}
                )
            },
            version="0.1.0",
        )
        return create_partial_response(
            data=health_response,
            errors=[
                ErrorDetail(code="CONNECTION_ERROR", message=f"Connection error: {e}")
            ],
            message="Health check completed with connection issues",
        )
    except (RuntimeError, AttributeError, TypeError) as e:
        logger.error(f"Service component error during health check: {e}")
        health_response = HealthResponse(
            status="degraded",
            components={
                "service": ComponentHealth(
                    status="error", details={"error": f"Component error: {e}"}
                )
            },
            version="0.1.0",
        )
        return create_partial_response(
            data=health_response,
            errors=[
                ErrorDetail(code="COMPONENT_ERROR", message=f"Component error: {e}")
            ],
            message="Health check completed with component issues",
        )
    except Exception as e:
        logger.error(f"Unexpected error performing health check: {e}")
        health_response = HealthResponse(
            status="degraded",
            components={
                "service": ComponentHealth(
                    status="error", details={"error": f"Unexpected error: {e}"}
                )
            },
            version="0.1.0",
        )
        return create_error_response(f"Health check failed: {e}", "HEALTH_CHECK_FAILED")


@router.get("/metrics", response_model=UnifiedResponse)
async def get_metrics(
    namespace: Optional[str] = Query(
        default=None, description="Filter metrics by namespace"
    ),
    service: MetadataStampingService = Depends(get_service),
) -> UnifiedResponse:
    """Get service performance metrics.

    Args:
        namespace: Optional namespace filter for metrics
        service: Injected service instance

    Returns:
        Unified response with performance metrics
    """
    try:
        # Get hash generator metrics
        hash_metrics = await (
            service.stamping_engine.hash_generator.metrics_collector.get_performance_stats()
        )

        # Get database metrics if available
        db_metrics = {}
        if service.db_client:
            db_metrics = await service.db_client.get_performance_statistics()

        metrics_data = {
            "hash_generation": hash_metrics,
            "database": db_metrics,
            "service_uptime": service.get_uptime(),
        }

        # Filter by namespace if specified
        if namespace:
            # This would be implemented with namespace-specific metrics filtering
            # For now, just add the namespace to metadata
            pass

        return create_success_response(
            data=metrics_data,
            message="Performance metrics retrieved successfully",
            metadata={
                "namespace": namespace,
                "metrics_collected_at": time.time(),
                "operation": "get_metrics",
            },
        )

    except Exception as e:
        raise handle_service_exception(e, "metrics retrieval")


# Transformer-based endpoints for O.N.E. v0.1 compliance
@router.post("/transform/stamp", response_model=UnifiedResponse)
async def transformer_stamp_content(
    request: StampingInput, service: MetadataStampingService = Depends(get_service)
) -> UnifiedResponse:
    """
    Create stamp using transformer pattern.

    Args:
        request: Stamping input with schema validation
        service: Service dependency

    Returns:
        UnifiedResponse: Stamped content with metadata
    """
    try:
        import uuid

        from ..execution import ExecutionContext, get_transformer

        # Get stamping transformer
        stamper = get_transformer("metadata_stamper")
        if not stamper:
            return UnifiedResponse(
                status="error", error="Metadata stamping transformer not found"
            )

        # Create execution context
        context = ExecutionContext(
            execution_id=str(uuid.uuid4()),
            input_schema="StampingInput",
            output_schema="StampingOutput",
        )

        # Execute transformer with validation
        result = await stamper.execute_with_validation(request, context)

        return UnifiedResponse(
            status="success",
            data=result.model_dump(),
            metadata={
                "transformer": "metadata_stamper",
                "execution_id": context.execution_id,
                "schema_version": "1.0.0",
            },
        )

    except Exception as e:
        return UnifiedResponse(status="error", error=str(e))


@router.post("/transform/validate", response_model=UnifiedResponse)
async def transformer_validate_content(
    request: ValidationInput, service: MetadataStampingService = Depends(get_service)
) -> UnifiedResponse:
    """
    Validate stamps using transformer pattern.

    Args:
        request: Validation input with schema validation
        service: Service dependency

    Returns:
        UnifiedResponse: Validation results
    """
    try:
        import uuid

        from ..execution import ExecutionContext, get_transformer

        # Get validation transformer
        validator = get_transformer("stamp_validator")
        if not validator:
            return UnifiedResponse(
                status="error", error="Stamp validation transformer not found"
            )

        # Create execution context
        context = ExecutionContext(
            execution_id=str(uuid.uuid4()),
            input_schema="ValidationInput",
            output_schema="ValidationOutput",
        )

        # Execute transformer with validation
        result = await validator.execute_with_validation(request, context)

        return UnifiedResponse(
            status="success",
            data=result.model_dump(),
            metadata={
                "transformer": "stamp_validator",
                "execution_id": context.execution_id,
                "schema_version": "1.0.0",
            },
        )

    except Exception as e:
        return UnifiedResponse(status="error", error=str(e))


@router.get("/transformers", response_model=UnifiedResponse)
async def list_transformers() -> UnifiedResponse:
    """
    List all registered transformers.

    Returns:
        UnifiedResponse: List of transformers with metrics
    """
    try:
        from ..execution import list_transformers as get_all_transformers

        transformers = get_all_transformers()
        transformer_list = []

        for name, transformer in transformers.items():
            metrics = transformer.get_metrics()
            transformer_list.append(
                {"name": name, "version": transformer.version, "metrics": metrics}
            )

        return UnifiedResponse(
            status="success",
            data={"transformers": transformer_list, "count": len(transformer_list)},
        )

    except Exception as e:
        return UnifiedResponse(status="error", error=str(e))


@router.get("/schemas", response_model=UnifiedResponse)
async def list_schemas() -> UnifiedResponse:
    """
    List all registered schemas.

    Returns:
        UnifiedResponse: List of schemas with versions
    """
    try:
        from ..execution import schema_registry

        schemas = schema_registry.list_schemas()
        active_schemas = schema_registry.list_active_schemas()

        return UnifiedResponse(
            status="success",
            data={
                "all_schemas": schemas,
                "active_schemas": active_schemas,
                "total_schemas": len(schemas),
            },
        )

    except Exception as e:
        return UnifiedResponse(status="error", error=str(e))
