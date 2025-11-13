"""
Batch processing optimizations for MetadataStampingService Phase 2.

Provides high-performance batch processing capabilities with intelligent
scheduling, resource management, and optimized throughput for large-scale
metadata stamping operations.
"""

import asyncio
import heapq
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .cache.redis_cache import MetadataStampingRedisCache, StampCacheEntry
from .extractors.advanced_extractors import MetadataExtractionEngine, MetadataResult
from .monitoring.metrics_collector import MetricsCollector
from .streaming.kafka_handler import MetadataStampingKafkaHandler

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Batch processing priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class BatchStatus(Enum):
    """Batch processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingStrategy(Enum):
    """Batch processing strategies."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    PIPELINE = "pipeline"


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Configuration values are environment-aware defaults.
    Batch size can be tuned based on workload characteristics.
    """

    max_batch_size: int = 100  # Default batch size (tune based on workload)
    max_concurrent_batches: int = 5
    max_concurrent_files_per_batch: int = 10
    default_timeout_seconds: int = 300  # 5 minutes

    # Resource management
    max_memory_usage_mb: int = 1024
    max_cpu_usage_percent: float = 80.0

    # Performance optimization
    enable_cache_warming: bool = True
    enable_prefetching: bool = True
    enable_result_streaming: bool = True

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0

    # Quality settings
    enable_quality_checks: bool = True
    quality_threshold: float = 0.8


class FileProcessingRequest(BaseModel):
    """Request for processing a single file in a batch."""

    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_hash: str
    file_path: str
    file_data: bytes
    file_size: int
    content_type: str
    priority: BatchPriority = BatchPriority.NORMAL

    # Processing options
    extract_metadata: bool = True
    cache_result: bool = True
    generate_thumbnail: bool = False

    # Retry tracking
    retry_count: int = 0
    last_error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BatchProcessingRequest(BaseModel):
    """Request for batch processing."""

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    files: list[FileProcessingRequest]
    priority: BatchPriority = BatchPriority.NORMAL
    strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE
    timeout_seconds: int = 300

    # Processing options
    max_concurrency: int = 10
    enable_early_termination: bool = True
    quality_threshold: float = 0.8

    # Callbacks
    progress_callback: Optional[Callable] = None
    completion_callback: Optional[Callable] = None

    # Timestamps
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class FileProcessingResult(BaseModel):
    """Result of processing a single file."""

    file_id: str
    file_hash: str
    status: str  # "success", "failed", "skipped"
    execution_time_ms: float

    # Results
    stamp_data: Optional[dict[str, Any]] = None
    metadata_result: Optional[MetadataResult] = None
    cache_hit: bool = False

    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0


class BatchProcessingResult(BaseModel):
    """Result of batch processing."""

    batch_id: str
    status: BatchStatus
    total_files: int
    successful_files: int
    failed_files: int
    skipped_files: int

    # Timing
    execution_time_ms: float
    average_file_time_ms: float

    # Results
    file_results: list[FileProcessingResult]

    # Performance metrics
    cache_hit_rate: float
    throughput_files_per_second: float
    memory_peak_mb: float
    cpu_peak_percent: float

    # Quality metrics
    quality_score: float
    quality_grade: str


class ResourceMonitor:
    """Monitor system resources during batch processing."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self._monitor_task = asyncio.create_task(self._monitor_resources())

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_resources(self):
        """Background resource monitoring."""
        try:
            import psutil

            process = psutil.Process()

            while self.monitoring:
                try:
                    # Memory usage
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

                    # CPU usage
                    cpu_percent = process.cpu_percent()
                    self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)

                    # Check limits
                    if memory_mb > self.config.max_memory_usage_mb:
                        logger.warning(
                            f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.config.max_memory_usage_mb}MB)"
                        )

                    if cpu_percent > self.config.max_cpu_usage_percent:
                        logger.warning(
                            f"CPU usage ({cpu_percent:.1f}%) exceeds limit ({self.config.max_cpu_usage_percent}%)"
                        )

                    await asyncio.sleep(1)  # Check every second

                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Resource monitor failed: {e}")


class BatchProcessor:
    """
    High-performance batch processor for MetadataStampingService.

    Features:
    - Intelligent batch scheduling and prioritization
    - Adaptive concurrency control based on system resources
    - Cache-aware processing with result streaming
    - Pipeline optimization for maximum throughput
    - Quality monitoring and early termination
    - Comprehensive metrics and monitoring
    """

    def __init__(
        self,
        config: BatchConfig,
        cache: Optional[MetadataStampingRedisCache] = None,
        kafka_handler: Optional[MetadataStampingKafkaHandler] = None,
        extraction_engine: Optional[MetadataExtractionEngine] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self.config = config
        self.cache = cache
        self.kafka_handler = kafka_handler
        self.extraction_engine = extraction_engine
        self.metrics_collector = metrics_collector

        # Processing queues (priority-based)
        self._batch_queue: list[tuple[int, float, BatchProcessingRequest]] = (
            []
        )  # (priority, timestamp, request)
        self._processing_batches: dict[str, BatchProcessingRequest] = {}

        # Resource management
        self.resource_monitor = ResourceMonitor(config)
        self._processing_semaphore = asyncio.Semaphore(config.max_concurrent_batches)

        # Performance tracking
        self._processing_stats = {
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Background tasks
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the batch processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_batches())
        logger.info("Batch processor started")

    async def stop(self):
        """Stop the batch processor."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        await self.resource_monitor.stop_monitoring()
        logger.info("Batch processor stopped")

    async def submit_batch(self, request: BatchProcessingRequest) -> str:
        """Submit a batch for processing."""
        # Validate request
        if not request.files:
            raise ValueError("Batch must contain at least one file")

        if len(request.files) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size ({len(request.files)}) exceeds maximum ({self.config.max_batch_size})"
            )

        # Add to priority queue
        priority_value = request.priority.value
        timestamp = time.time()

        heapq.heappush(self._batch_queue, (-priority_value, timestamp, request))

        logger.info(
            f"Submitted batch {request.batch_id} with {len(request.files)} files (priority: {request.priority.value})"
        )

        return request.batch_id

    async def get_batch_status(self, batch_id: str) -> Optional[dict[str, Any]]:
        """Get status of a batch."""
        if batch_id in self._processing_batches:
            batch = self._processing_batches[batch_id]
            return {
                "batch_id": batch_id,
                "status": "processing",
                "total_files": len(batch.files),
                "started_at": batch.started_at,
                "elapsed_time": time.time() - (batch.started_at or time.time()),
            }

        # Check if batch is in queue
        for _, _, queued_batch in self._batch_queue:
            if queued_batch.batch_id == batch_id:
                return {
                    "batch_id": batch_id,
                    "status": "queued",
                    "total_files": len(queued_batch.files),
                    "queue_position": self._get_queue_position(batch_id),
                }

        return None

    def _get_queue_position(self, batch_id: str) -> int:
        """Get position of batch in queue."""
        for i, (_, _, batch) in enumerate(self._batch_queue):
            if batch.batch_id == batch_id:
                return i + 1
        return -1

    async def _process_batches(self):
        """Main batch processing loop."""
        while self._running:
            try:
                if not self._batch_queue:
                    await asyncio.sleep(0.1)
                    continue

                # Get next batch from priority queue
                _, _, batch_request = heapq.heappop(self._batch_queue)

                # Process batch asynchronously
                task = asyncio.create_task(self._process_single_batch(batch_request))

                # Don't wait for completion - allows concurrent processing

            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(1)

    async def _process_single_batch(
        self, request: BatchProcessingRequest
    ) -> BatchProcessingResult:
        """Process a single batch."""
        async with self._processing_semaphore:
            batch_id = request.batch_id
            start_time = time.perf_counter()

            try:
                # Add to processing batches
                request.started_at = time.time()
                self._processing_batches[batch_id] = request

                # Start resource monitoring
                await self.resource_monitor.start_monitoring()

                logger.info(
                    f"Processing batch {batch_id} with {len(request.files)} files"
                )

                # Choose processing strategy
                if request.strategy == ProcessingStrategy.ADAPTIVE:
                    strategy = self._choose_optimal_strategy(request)
                else:
                    strategy = request.strategy

                # Process files based on strategy
                if strategy == ProcessingStrategy.SEQUENTIAL:
                    file_results = await self._process_sequential(request)
                elif strategy == ProcessingStrategy.PARALLEL:
                    file_results = await self._process_parallel(request)
                elif strategy == ProcessingStrategy.PIPELINE:
                    file_results = await self._process_pipeline(request)
                else:
                    file_results = await self._process_parallel(
                        request
                    )  # Default fallback

                # Calculate results
                execution_time = (time.perf_counter() - start_time) * 1000
                successful_files = sum(1 for r in file_results if r.status == "success")
                failed_files = sum(1 for r in file_results if r.status == "failed")
                skipped_files = sum(1 for r in file_results if r.status == "skipped")
                cache_hits = sum(1 for r in file_results if r.cache_hit)

                # Quality assessment
                quality_score = (
                    successful_files / len(file_results) if file_results else 0.0
                )
                quality_grade = self._calculate_quality_grade(quality_score)

                # Create result
                result = BatchProcessingResult(
                    batch_id=batch_id,
                    status=(
                        BatchStatus.COMPLETED
                        if successful_files > 0
                        else BatchStatus.FAILED
                    ),
                    total_files=len(request.files),
                    successful_files=successful_files,
                    failed_files=failed_files,
                    skipped_files=skipped_files,
                    execution_time_ms=execution_time,
                    average_file_time_ms=(
                        execution_time / len(file_results) if file_results else 0.0
                    ),
                    file_results=file_results,
                    cache_hit_rate=(
                        cache_hits / len(file_results) if file_results else 0.0
                    ),
                    throughput_files_per_second=(
                        len(file_results) / (execution_time / 1000)
                        if execution_time > 0
                        else 0.0
                    ),
                    memory_peak_mb=self.resource_monitor.peak_memory_mb,
                    cpu_peak_percent=self.resource_monitor.peak_cpu_percent,
                    quality_score=quality_score,
                    quality_grade=quality_grade,
                )

                # Update stats
                self._update_stats(result)

                # Notify completion callback
                if request.completion_callback:
                    try:
                        await request.completion_callback(result)
                    except Exception as e:
                        logger.error(f"Error in completion callback: {e}")

                logger.info(
                    f"Completed batch {batch_id}: {successful_files}/{len(request.files)} files successful in {execution_time:.1f}ms"
                )

                return result

            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.error(f"Batch {batch_id} failed: {e}")

                return BatchProcessingResult(
                    batch_id=batch_id,
                    status=BatchStatus.FAILED,
                    total_files=len(request.files),
                    successful_files=0,
                    failed_files=len(request.files),
                    skipped_files=0,
                    execution_time_ms=execution_time,
                    average_file_time_ms=0.0,
                    file_results=[],
                    cache_hit_rate=0.0,
                    throughput_files_per_second=0.0,
                    memory_peak_mb=self.resource_monitor.peak_memory_mb,
                    cpu_peak_percent=self.resource_monitor.peak_cpu_percent,
                    quality_score=0.0,
                    quality_grade="F",
                )

            finally:
                # Cleanup
                self._processing_batches.pop(batch_id, None)
                await self.resource_monitor.stop_monitoring()

    def _choose_optimal_strategy(
        self, request: BatchProcessingRequest
    ) -> ProcessingStrategy:
        """Choose optimal processing strategy based on batch characteristics."""
        file_count = len(request.files)
        avg_file_size = (
            sum(f.file_size for f in request.files) / file_count
            if file_count > 0
            else 0
        )

        # Small batches with small files - sequential is efficient
        if file_count <= 10 and avg_file_size < 1024 * 1024:  # < 1MB
            return ProcessingStrategy.SEQUENTIAL

        # Large files - pipeline processing for better resource utilization
        if avg_file_size > 10 * 1024 * 1024:  # > 10MB
            return ProcessingStrategy.PIPELINE

        # Default to parallel processing
        return ProcessingStrategy.PARALLEL

    async def _process_sequential(
        self, request: BatchProcessingRequest
    ) -> list[FileProcessingResult]:
        """Process files sequentially."""
        results = []

        for i, file_req in enumerate(request.files):
            try:
                # Progress callback
                if request.progress_callback:
                    await request.progress_callback(i + 1, len(request.files))

                result = await self._process_single_file(file_req)
                results.append(result)

                # Early termination check
                if request.enable_early_termination and self._should_terminate_early(
                    results, request
                ):
                    logger.warning(
                        f"Early termination triggered for batch {request.batch_id}"
                    )
                    break

            except Exception as e:
                logger.error(f"Error processing file {file_req.file_id}: {e}")
                results.append(
                    FileProcessingResult(
                        file_id=file_req.file_id,
                        file_hash=file_req.file_hash,
                        status="failed",
                        execution_time_ms=0.0,
                        error_message=str(e),
                        error_type=type(e).__name__,
                    )
                )

        return results

    async def _process_parallel(
        self, request: BatchProcessingRequest
    ) -> list[FileProcessingResult]:
        """Process files in parallel."""
        semaphore = asyncio.Semaphore(request.max_concurrency)

        async def process_with_semaphore(
            file_req: FileProcessingRequest,
        ) -> FileProcessingResult:
            async with semaphore:
                return await self._process_single_file(file_req)

        # Create tasks for all files
        tasks = [
            asyncio.create_task(process_with_semaphore(file_req))
            for file_req in request.files
        ]

        # Process with progress tracking
        results = []
        completed = 0

        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
                completed += 1

                # Progress callback
                if request.progress_callback:
                    await request.progress_callback(completed, len(request.files))

                # Early termination check
                if request.enable_early_termination and self._should_terminate_early(
                    results, request
                ):
                    logger.warning(
                        f"Early termination triggered for batch {request.batch_id}"
                    )
                    # Cancel remaining tasks
                    for remaining_task in tasks:
                        if not remaining_task.done():
                            remaining_task.cancel()
                    break

            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                results.append(
                    FileProcessingResult(
                        file_id="unknown",
                        file_hash="",
                        status="failed",
                        execution_time_ms=0.0,
                        error_message=str(e),
                        error_type=type(e).__name__,
                    )
                )

        return results

    async def _process_pipeline(
        self, request: BatchProcessingRequest
    ) -> list[FileProcessingResult]:
        """Process files using pipeline strategy."""
        # Pipeline stages: Cache Check -> Hash Generation -> Metadata Extraction -> Store

        stage1_queue = asyncio.Queue(maxsize=request.max_concurrency)
        stage2_queue = asyncio.Queue(maxsize=request.max_concurrency)
        stage3_queue = asyncio.Queue(maxsize=request.max_concurrency)
        results_queue = asyncio.Queue()

        # Pipeline workers
        async def cache_check_worker():
            """Stage 1: Check cache for existing results."""
            while True:
                try:
                    file_req = await stage1_queue.get()
                    if file_req is None:  # Sentinel to stop
                        await stage2_queue.put(None)
                        break

                    # Check cache
                    cached_result = None
                    if self.cache:
                        cached_result = await self.cache.get_stamp(file_req.file_hash)

                    await stage2_queue.put((file_req, cached_result))
                    stage1_queue.task_done()

                except Exception as e:
                    logger.error(f"Cache check worker error: {e}")

        async def hash_worker():
            """Stage 2: Generate hash if not cached."""
            while True:
                try:
                    item = await stage2_queue.get()
                    if item is None:  # Sentinel to stop
                        await stage3_queue.put(None)
                        break

                    file_req, cached_result = item

                    if cached_result:
                        # Cache hit - skip to results
                        result = FileProcessingResult(
                            file_id=file_req.file_id,
                            file_hash=file_req.file_hash,
                            status="success",
                            execution_time_ms=1.0,  # Minimal time for cache hit
                            stamp_data=(
                                cached_result.stamp_data.dict()
                                if hasattr(cached_result, "stamp_data")
                                else {}
                            ),
                            cache_hit=True,
                        )
                        await results_queue.put(result)
                    else:
                        await stage3_queue.put((file_req, None))

                    stage2_queue.task_done()

                except Exception as e:
                    logger.error(f"Hash worker error: {e}")

        async def extraction_worker():
            """Stage 3: Extract metadata and generate final result."""
            while True:
                try:
                    item = await stage3_queue.get()
                    if item is None:  # Sentinel to stop
                        break

                    file_req, _ = item
                    result = await self._process_single_file(file_req)
                    await results_queue.put(result)
                    stage3_queue.task_done()

                except Exception as e:
                    logger.error(f"Extraction worker error: {e}")

        # Start pipeline workers
        workers = [
            asyncio.create_task(cache_check_worker()),
            asyncio.create_task(hash_worker()),
            asyncio.create_task(extraction_worker()),
        ]

        # Feed files into pipeline
        for file_req in request.files:
            await stage1_queue.put(file_req)

        # Send sentinel to stop pipeline
        await stage1_queue.put(None)

        # Collect results
        results = []
        completed = 0

        while completed < len(request.files):
            try:
                result = await asyncio.wait_for(results_queue.get(), timeout=10.0)
                results.append(result)
                completed += 1

                # Progress callback
                if request.progress_callback:
                    await request.progress_callback(completed, len(request.files))

            except TimeoutError:
                logger.warning("Pipeline processing timeout")
                break

        # Wait for workers to complete
        await asyncio.gather(*workers, return_exceptions=True)

        return results

    async def _process_single_file(
        self, file_req: FileProcessingRequest
    ) -> FileProcessingResult:
        """Process a single file."""
        start_time = time.perf_counter()

        try:
            # Check cache first
            cache_hit = False
            if self.cache and file_req.cache_result:
                cached_entry = await self.cache.get_stamp(file_req.file_hash)
                if cached_entry:
                    self._processing_stats["cache_hits"] += 1
                    return FileProcessingResult(
                        file_id=file_req.file_id,
                        file_hash=file_req.file_hash,
                        status="success",
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        stamp_data=cached_entry.stamp_data,
                        cache_hit=True,
                    )

            self._processing_stats["cache_misses"] += 1

            # Extract metadata if requested
            metadata_result = None
            if file_req.extract_metadata and self.extraction_engine:
                metadata_result = await self.extraction_engine.extract_metadata(
                    file_req.file_data, file_req.file_path, file_req.content_type
                )

            # Create stamp data
            stamp_data = {
                "file_hash": file_req.file_hash,
                "file_path": file_req.file_path,
                "file_size": file_req.file_size,
                "content_type": file_req.content_type,
                "processed_at": time.time(),
                "metadata": metadata_result.model_dump() if metadata_result else None,
            }

            # Cache result if requested
            if self.cache and file_req.cache_result:
                cache_entry = StampCacheEntry(
                    stamp_id=str(uuid.uuid4()),
                    file_hash=file_req.file_hash,
                    stamp_data=stamp_data,
                )
                await self.cache.cache_stamp(cache_entry)

            execution_time = (time.perf_counter() - start_time) * 1000

            return FileProcessingResult(
                file_id=file_req.file_id,
                file_hash=file_req.file_hash,
                status="success",
                execution_time_ms=execution_time,
                stamp_data=stamp_data,
                metadata_result=metadata_result,
                cache_hit=cache_hit,
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error processing file {file_req.file_id}: {e}")

            return FileProcessingResult(
                file_id=file_req.file_id,
                file_hash=file_req.file_hash,
                status="failed",
                execution_time_ms=execution_time,
                error_message=str(e),
                error_type=type(e).__name__,
                retry_count=file_req.retry_count,
            )

    def _should_terminate_early(
        self, results: list[FileProcessingResult], request: BatchProcessingRequest
    ) -> bool:
        """Check if batch should terminate early due to quality issues."""
        if (
            not request.enable_early_termination or len(results) < 10
        ):  # Need minimum sample size
            return False

        # Calculate current success rate
        successful = sum(1 for r in results if r.status == "success")
        success_rate = successful / len(results)

        # Terminate if success rate is below threshold
        return success_rate < request.quality_threshold

    def _calculate_quality_grade(self, quality_score: float) -> str:
        """Calculate quality grade based on success rate."""
        if quality_score >= 0.95:
            return "A"
        elif quality_score >= 0.85:
            return "B"
        elif quality_score >= 0.70:
            return "C"
        elif quality_score >= 0.50:
            return "D"
        else:
            return "F"

    def _update_stats(self, result: BatchProcessingResult):
        """Update processing statistics."""
        self._processing_stats["total_batches"] += 1
        self._processing_stats["total_files"] += result.total_files
        self._processing_stats["successful_files"] += result.successful_files
        self._processing_stats["failed_files"] += result.failed_files
        self._processing_stats["total_processing_time"] += result.execution_time_ms

        if result.status == BatchStatus.COMPLETED:
            self._processing_stats["successful_batches"] += 1
        else:
            self._processing_stats["failed_batches"] += 1

        # Update cache stats
        for file_result in result.file_results:
            if file_result.cache_hit:
                self._processing_stats["cache_hits"] += 1
            else:
                self._processing_stats["cache_misses"] += 1

    def get_statistics(self) -> dict[str, Any]:
        """Get batch processing statistics."""
        stats = self._processing_stats.copy()

        # Calculate derived metrics
        if stats["total_batches"] > 0:
            stats["batch_success_rate"] = (
                stats["successful_batches"] / stats["total_batches"]
            )
            stats["average_batch_time_ms"] = (
                stats["total_processing_time"] / stats["total_batches"]
            )

        if stats["total_files"] > 0:
            stats["file_success_rate"] = (
                stats["successful_files"] / stats["total_files"]
            )
            stats["average_file_time_ms"] = (
                stats["total_processing_time"] / stats["total_files"]
            )

        if (stats["cache_hits"] + stats["cache_misses"]) > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / (
                stats["cache_hits"] + stats["cache_misses"]
            )

        # Current state
        stats["current_queue_size"] = len(self._batch_queue)
        stats["current_processing_batches"] = len(self._processing_batches)

        return stats


# Factory function
def create_batch_processor(
    config: Optional[BatchConfig] = None,
    cache: Optional[MetadataStampingRedisCache] = None,
    kafka_handler: Optional[MetadataStampingKafkaHandler] = None,
    extraction_engine: Optional[MetadataExtractionEngine] = None,
    metrics_collector: Optional[MetricsCollector] = None,
) -> BatchProcessor:
    """Factory function to create batch processor."""
    if config is None:
        config = BatchConfig()

    return BatchProcessor(
        config=config,
        cache=cache,
        kafka_handler=kafka_handler,
        extraction_engine=extraction_engine,
        metrics_collector=metrics_collector,
    )
