"""
MetadataStampingService Phase 2 - Main Application Module

Integrates all Phase 2 components including Redis caching, Kafka streaming,
advanced metadata extraction, batch processing, and comprehensive monitoring.
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Database client
from omninode_bridge.services.metadata_stamping.database.client import (
    DatabaseConfig,
    MetadataStampingPostgresClient,
)

from .batch_processor import (
    BatchConfig,
    BatchProcessingRequest,
    BatchProcessor,
    FileProcessingRequest,
    create_batch_processor,
)

# Phase 2 components
from .cache.redis_cache import (
    CacheConfig,
    MetadataStampingRedisCache,
    create_redis_cache,
)
from .extractors.advanced_extractors import (
    ExtractionConfig,
    MetadataExtractionEngine,
    create_extraction_engine,
)
from .monitoring.metrics_collector import MetricsCollector, create_metrics_collector
from .streaming.kafka_handler import (
    KafkaConfig,
    MetadataStampingKafkaHandler,
    create_kafka_handler,
)

logger = logging.getLogger(__name__)


# Global service components
cache: Optional[MetadataStampingRedisCache] = None
kafka_handler: Optional[MetadataStampingKafkaHandler] = None
extraction_engine: Optional[MetadataExtractionEngine] = None
metrics_collector: Optional[MetricsCollector] = None
batch_processor: Optional[BatchProcessor] = None
db_client: Optional[MetadataStampingPostgresClient] = None


class ServiceConfig(BaseModel):
    """Main service configuration."""

    # Service settings
    service_name: str = "MetadataStampingService"
    service_version: str = "0.2.0"
    service_phase: str = "Phase 2 - Advanced Features"

    # Component enable/disable flags
    enable_redis_cache: bool = True
    enable_kafka_streaming: bool = True
    enable_batch_processing: bool = True
    enable_advanced_extraction: bool = True
    enable_monitoring: bool = True

    # Performance settings
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 30

    # Component configurations
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    kafka_config: KafkaConfig = Field(default_factory=KafkaConfig)
    extraction_config: ExtractionConfig = Field(default_factory=ExtractionConfig)
    batch_config: BatchConfig = Field(default_factory=BatchConfig)
    database_config: DatabaseConfig = Field(
        default_factory=lambda: DatabaseConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "metadata_stamping_dev"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
    )


def load_config() -> ServiceConfig:
    """Load service configuration from environment variables."""
    config = ServiceConfig()

    # Redis configuration
    if os.getenv("REDIS_HOST"):
        config.cache_config.host = os.getenv("REDIS_HOST")
    if os.getenv("REDIS_PORT"):
        config.cache_config.port = int(os.getenv("REDIS_PORT"))
    if os.getenv("REDIS_PASSWORD"):
        config.cache_config.password = os.getenv("REDIS_PASSWORD")

    # Kafka configuration
    if os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        config.kafka_config.bootstrap_servers = os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS"
        ).split(",")
    if os.getenv("KAFKA_CONSUMER_GROUP_ID"):
        config.kafka_config.consumer_group_id = os.getenv("KAFKA_CONSUMER_GROUP_ID")

    # Feature flags
    config.enable_redis_cache = (
        os.getenv("ENABLE_REDIS_CACHE", "true").lower() == "true"
    )
    config.enable_kafka_streaming = (
        os.getenv("ENABLE_KAFKA_STREAMING", "true").lower() == "true"
    )
    config.enable_batch_processing = (
        os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"
    )

    return config


service_config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global cache, kafka_handler, extraction_engine, metrics_collector, batch_processor, db_client

    logger.info(
        f"Starting {service_config.service_name} {service_config.service_version}"
    )

    try:
        # Initialize database client
        try:
            db_client = MetadataStampingPostgresClient(service_config.database_config)
            await db_client.initialize()
            logger.info("Database client initialized")
        except Exception as e:
            logger.error(f"Database client initialization failed: {e}")
            db_client = None

        # Initialize metrics collector
        if service_config.enable_monitoring:
            metrics_collector = create_metrics_collector()
            await metrics_collector.start_monitoring()
            logger.info("Metrics collector initialized")

        # Initialize Redis cache
        if service_config.enable_redis_cache:
            try:
                cache = await create_redis_cache(service_config.cache_config)
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
                cache = None

        # Initialize Kafka handler
        if service_config.enable_kafka_streaming:
            try:
                kafka_handler = await create_kafka_handler(service_config.kafka_config)
                logger.info("Kafka handler initialized")
            except Exception as e:
                logger.warning(f"Kafka handler initialization failed: {e}")
                kafka_handler = None

        # Initialize metadata extraction engine
        if service_config.enable_advanced_extraction:
            extraction_engine = create_extraction_engine(
                service_config.extraction_config
            )
            logger.info("Metadata extraction engine initialized")

        # Initialize batch processor
        if service_config.enable_batch_processing:
            batch_processor = create_batch_processor(
                config=service_config.batch_config,
                cache=cache,
                kafka_handler=kafka_handler,
                extraction_engine=extraction_engine,
                metrics_collector=metrics_collector,
            )
            await batch_processor.start()
            logger.info("Batch processor initialized")

        yield

    finally:
        # Cleanup
        logger.info("Shutting down services...")

        if batch_processor:
            await batch_processor.stop()

        if kafka_handler:
            await kafka_handler.close()

        if cache:
            await cache.close()

        if db_client:
            await db_client.close()

        if metrics_collector:
            await metrics_collector.stop_monitoring()

        logger.info("All services shut down")


# FastAPI application
app = FastAPI(
    title=service_config.service_name,
    version=service_config.service_version,
    description=f"{service_config.service_name} - {service_config.service_phase}",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
async def get_cache() -> Optional[MetadataStampingRedisCache]:
    """Get Redis cache instance."""
    return cache


async def get_kafka_handler() -> Optional[MetadataStampingKafkaHandler]:
    """Get Kafka handler instance."""
    return kafka_handler


async def get_extraction_engine() -> Optional[MetadataExtractionEngine]:
    """Get metadata extraction engine instance."""
    return extraction_engine


async def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get metrics collector instance."""
    return metrics_collector


async def get_batch_processor() -> Optional[BatchProcessor]:
    """Get batch processor instance."""
    return batch_processor


async def get_db_client() -> Optional[MetadataStampingPostgresClient]:
    """Get database client instance."""
    return db_client


# API Models
class StampRequest(BaseModel):
    """Request model for single file stamping."""

    file_hash: str
    file_path: str
    file_data: bytes
    content_type: str = "application/octet-stream"
    extract_metadata: bool = True
    cache_result: bool = True

    class Config:
        arbitrary_types_allowed = True


class StampResponse(BaseModel):
    """Response model for stamping operations."""

    stamp_id: str
    file_hash: str
    stamp_data: dict
    execution_time_ms: float
    cache_hit: bool = False
    metadata_extracted: bool = False


class BatchStampRequest(BaseModel):
    """Request model for batch stamping."""

    files: list[StampRequest]
    priority: str = "normal"
    strategy: str = "adaptive"
    max_concurrency: int = 10


class BatchStampResponse(BaseModel):
    """Response model for batch operations."""

    batch_id: str
    status: str
    message: str


# API Endpoints


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": service_config.service_name,
        "version": service_config.service_version,
        "phase": service_config.service_phase,
        "status": "operational",
        "features": {
            "database": db_client is not None,
            "redis_cache": service_config.enable_redis_cache and cache is not None,
            "kafka_streaming": service_config.enable_kafka_streaming
            and kafka_handler is not None,
            "batch_processing": service_config.enable_batch_processing
            and batch_processor is not None,
            "advanced_extraction": service_config.enable_advanced_extraction
            and extraction_engine is not None,
            "monitoring": service_config.enable_monitoring
            and metrics_collector is not None,
        },
    }


@app.get("/health")
async def health_check(
    cache_client: Optional[MetadataStampingRedisCache] = Depends(get_cache),
    kafka_client: Optional[MetadataStampingKafkaHandler] = Depends(get_kafka_handler),
    batch_proc: Optional[BatchProcessor] = Depends(get_batch_processor),
):
    """Comprehensive health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "service": service_config.service_name,
        "version": service_config.service_version,
        "components": {},
    }

    # Check cache health
    if cache_client:
        try:
            cache_health = await cache_client.health_check()
            health_status["components"]["cache"] = cache_health
        except Exception as e:
            health_status["components"]["cache"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "degraded"

    # Check Kafka health
    if kafka_client:
        try:
            kafka_health = await kafka_client.health_check()
            health_status["components"]["kafka"] = kafka_health
        except Exception as e:
            health_status["components"]["kafka"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "degraded"

    # Check batch processor health
    if batch_proc:
        batch_stats = batch_proc.get_statistics()
        health_status["components"]["batch_processor"] = {
            "status": "healthy",
            "queue_size": batch_stats.get("current_queue_size", 0),
            "processing_batches": batch_stats.get("current_processing_batches", 0),
        }

    return health_status


@app.post("/stamp", response_model=StampResponse)
async def create_stamp(
    request: StampRequest,
    background_tasks: BackgroundTasks,
    extraction: Optional[MetadataExtractionEngine] = Depends(get_extraction_engine),
    cache_client: Optional[MetadataStampingRedisCache] = Depends(get_cache),
    kafka_client: Optional[MetadataStampingKafkaHandler] = Depends(get_kafka_handler),
    metrics: Optional[MetricsCollector] = Depends(get_metrics_collector),
    database: Optional[MetadataStampingPostgresClient] = Depends(get_db_client),
):
    """Create metadata stamp for a single file with idempotency support."""
    import time
    import uuid

    import asyncpg

    start_time = time.perf_counter()

    try:
        # Extract business logic into inner function for optional metrics wrapping
        async def _process_stamp_request() -> StampResponse:
            # Check cache first
            cache_hit = False
            if cache_client:
                cached_entry = await cache_client.get_stamp(request.file_hash)
                if cached_entry:
                    execution_time = (time.perf_counter() - start_time) * 1000
                    return StampResponse(
                        stamp_id=cached_entry.stamp_id,
                        file_hash=request.file_hash,
                        stamp_data=cached_entry.stamp_data,
                        execution_time_ms=execution_time,
                        cache_hit=True,
                        metadata_extracted=False,
                    )

            # Check database for existing stamp (idempotency - prevents duplicate crash)
            if database:
                try:
                    existing_stamp = await database.get_metadata_stamp(
                        request.file_hash
                    )
                    if existing_stamp:
                        logger.info(
                            f"Stamp already exists for hash {request.file_hash}, returning existing"
                        )
                        execution_time = (time.perf_counter() - start_time) * 1000
                        return StampResponse(
                            stamp_id=existing_stamp["id"],
                            file_hash=request.file_hash,
                            stamp_data=existing_stamp.get("stamp_data", {}),
                            execution_time_ms=execution_time,
                            cache_hit=False,
                            metadata_extracted=False,
                        )
                except Exception as e:
                    logger.warning(
                        f"Database lookup failed: {e}, continuing with creation"
                    )

            # Extract metadata if enabled
            metadata_result = None
            if request.extract_metadata and extraction:
                metadata_result = await extraction.extract_metadata(
                    request.file_data, request.file_path, request.content_type
                )

            # Create stamp data
            stamp_data = {
                "file_hash": request.file_hash,
                "file_path": request.file_path,
                "file_size": len(request.file_data),
                "content_type": request.content_type,
                "processed_at": time.time(),
                "metadata": (metadata_result.model_dump() if metadata_result else None),
            }

            stamp_id = str(uuid.uuid4())

            # Store in database if available (with duplicate handling)
            if database:
                try:
                    db_result = await database.create_metadata_stamp(
                        file_hash=request.file_hash,
                        file_path=request.file_path,
                        file_size=len(request.file_data),
                        content_type=request.content_type,
                        stamp_data=stamp_data,
                        protocol_version="1.0",
                        intelligence_data={},
                        version=1,
                        op_id=stamp_id,
                        namespace="omninode.services.metadata",
                        metadata_version="0.1",
                    )
                    stamp_id = db_result["id"]
                    logger.info(f"Stamp stored in database with ID: {stamp_id}")
                except asyncpg.exceptions.UniqueViolationError:
                    # Race condition: stamp was created by concurrent request
                    logger.warning(
                        f"Duplicate stamp detected (race condition) for hash {request.file_hash}"
                    )
                    # Fetch the existing stamp
                    try:
                        existing_stamp = await database.get_metadata_stamp(
                            request.file_hash
                        )
                        if existing_stamp:
                            execution_time = (time.perf_counter() - start_time) * 1000
                            return StampResponse(
                                stamp_id=existing_stamp["id"],
                                file_hash=request.file_hash,
                                stamp_data=existing_stamp.get("stamp_data", {}),
                                execution_time_ms=execution_time,
                                cache_hit=False,
                                metadata_extracted=False,
                            )
                    except Exception as fetch_error:
                        logger.error(f"Failed to fetch duplicate stamp: {fetch_error}")
                        # Fall through - return newly created stamp data without database ID
                except Exception as e:
                    logger.error(f"Database insert failed: {e}")
                    # Continue without database - service remains operational

            # Cache result if enabled
            if cache_client and request.cache_result:
                from .cache.redis_cache import StampCacheEntry

                cache_entry = StampCacheEntry(
                    stamp_id=stamp_id,
                    file_hash=request.file_hash,
                    stamp_data=stamp_data,
                )
                background_tasks.add_task(cache_client.cache_stamp, cache_entry)

            # Publish to Kafka if available
            if kafka_client:
                from .streaming.kafka_handler import StampResultEvent

                result_event = StampResultEvent(
                    request_event_id=str(uuid.uuid4()),
                    stamp_id=stamp_id,
                    file_hash=request.file_hash,
                    stamp_data=stamp_data,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    hash_generation_time_ms=5.0,  # Placeholder
                    metadata_extraction_time_ms=10.0,  # Placeholder
                    database_operation_time_ms=2.0,  # Placeholder
                )
                background_tasks.add_task(
                    kafka_client.publish_stamp_result, result_event
                )

            execution_time = (time.perf_counter() - start_time) * 1000

            return StampResponse(
                stamp_id=stamp_id,
                file_hash=request.file_hash,
                stamp_data=stamp_data,
                execution_time_ms=execution_time,
                cache_hit=cache_hit,
                metadata_extracted=metadata_result is not None,
            )

        # Conditionally wrap with metrics timing if available
        if metrics:
            async with metrics.time_request("POST", "/stamp"):
                return await _process_stamp_request()
        else:
            # Execute without metrics - service continues to function
            return await _process_stamp_request()

    except Exception as e:
        logger.error(f"Error creating stamp: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stamp/batch", response_model=BatchStampResponse)
async def create_batch_stamp(
    request: BatchStampRequest,
    batch_proc: Optional[BatchProcessor] = Depends(get_batch_processor),
):
    """Submit a batch of files for stamping."""
    if not batch_proc:
        raise HTTPException(status_code=503, detail="Batch processing not available")

    try:
        # Convert request to internal format
        files = [
            FileProcessingRequest(
                file_hash=file_req.file_hash,
                file_path=file_req.file_path,
                file_data=file_req.file_data,
                file_size=len(file_req.file_data),
                content_type=file_req.content_type,
                extract_metadata=file_req.extract_metadata,
                cache_result=file_req.cache_result,
            )
            for file_req in request.files
        ]

        # Map priority
        priority_map = {
            "low": "LOW",
            "normal": "NORMAL",
            "high": "HIGH",
            "urgent": "URGENT",
        }

        # Map strategy
        strategy_map = {
            "sequential": "SEQUENTIAL",
            "parallel": "PARALLEL",
            "adaptive": "ADAPTIVE",
            "pipeline": "PIPELINE",
        }

        from .batch_processor import BatchPriority, ProcessingStrategy

        batch_request = BatchProcessingRequest(
            files=files,
            priority=BatchPriority[priority_map.get(request.priority, "NORMAL")],
            strategy=ProcessingStrategy[strategy_map.get(request.strategy, "ADAPTIVE")],
            max_concurrency=request.max_concurrency,
        )

        batch_id = await batch_proc.submit_batch(batch_request)

        return BatchStampResponse(
            batch_id=batch_id,
            status="submitted",
            message=f"Batch submitted successfully with {len(files)} files",
        )

    except Exception as e:
        logger.error(f"Error submitting batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stamp/batch/{batch_id}")
async def get_batch_status(
    batch_id: str, batch_proc: Optional[BatchProcessor] = Depends(get_batch_processor)
):
    """Get status of a batch processing job."""
    if not batch_proc:
        raise HTTPException(status_code=503, detail="Batch processing not available")

    status = await batch_proc.get_batch_status(batch_id)
    if not status:
        raise HTTPException(status_code=404, detail="Batch not found")

    return status


@app.get("/metrics")
async def get_metrics(
    metrics: Optional[MetricsCollector] = Depends(get_metrics_collector),
):
    """Get Prometheus metrics."""
    if not metrics:
        raise HTTPException(status_code=503, detail="Metrics collection not available")

    metrics_data = metrics.get_metrics()
    return Response(content=metrics_data, media_type="text/plain")


@app.get("/metrics/summary")
async def get_metrics_summary(
    metrics: Optional[MetricsCollector] = Depends(get_metrics_collector),
    batch_proc: Optional[BatchProcessor] = Depends(get_batch_processor),
    cache_client: Optional[MetadataStampingRedisCache] = Depends(get_cache),
    kafka_client: Optional[MetadataStampingKafkaHandler] = Depends(get_kafka_handler),
):
    """Get performance metrics summary."""
    summary = {}

    if metrics:
        summary["performance"] = metrics.get_performance_summary()

    if batch_proc:
        summary["batch_processing"] = batch_proc.get_statistics()

    if cache_client:
        summary["cache"] = await cache_client.get_cache_stats()

    if kafka_client:
        summary["kafka"] = await kafka_client.get_metrics()

    return summary


@app.get("/status")
async def get_service_status():
    """Get detailed service status."""
    return {
        "service": service_config.service_name,
        "version": service_config.service_version,
        "phase": service_config.service_phase,
        "uptime": asyncio.get_event_loop().time(),
        "configuration": {
            "redis_cache_enabled": service_config.enable_redis_cache,
            "kafka_streaming_enabled": service_config.enable_kafka_streaming,
            "batch_processing_enabled": service_config.enable_batch_processing,
            "advanced_extraction_enabled": service_config.enable_advanced_extraction,
            "monitoring_enabled": service_config.enable_monitoring,
        },
        "components": {
            "database": db_client is not None,
            "cache": cache is not None,
            "kafka": kafka_handler is not None,
            "extraction": extraction_engine is not None,
            "batch_processor": batch_processor is not None,
            "metrics": metrics_collector is not None,
        },
    }


@app.get("/stamp/{file_hash}")
async def get_stamp(
    file_hash: str,
    namespace: Optional[str] = None,
    database: Optional[MetadataStampingPostgresClient] = Depends(get_db_client),
):
    """Retrieve metadata stamp by file hash.

    Args:
        file_hash: BLAKE3 hash of the file
        namespace: Optional namespace filter (not yet implemented in database)
        database: Database client dependency

    Returns:
        Stamp record if found

    Raises:
        HTTPException: 503 if database unavailable, 404 if stamp not found
    """
    if not database:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # Get stamp from database
        stamp = await database.get_metadata_stamp(file_hash)

        if not stamp:
            raise HTTPException(
                status_code=404, detail=f"Stamp not found for file_hash: {file_hash}"
            )

        # Filter by namespace if provided
        if namespace and stamp.get("namespace") != namespace:
            raise HTTPException(
                status_code=404, detail=f"Stamp not found in namespace: {namespace}"
            )

        # Return stamp as dict (already a dict from database)
        return stamp

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving stamp {file_hash}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stamp: {e!s}")


# Signal handling for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8053, log_level="info", access_log=True)
