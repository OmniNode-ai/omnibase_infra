# === OmniNode:Tool_Metadata ===
# metadata_version: 0.1
# name: metadata_stamping_hash_generator
# title: MetadataStampingService BLAKE3 Hash Generator
# version: 0.1.0
# namespace: omninode.services.metadata
# category: service.infrastructure.hashing
# kind: service
# role: hash_generator
# description: |
#   High-performance BLAKE3 hash generator optimized for sub-2ms hash generation
#   with memory-efficient streaming for large files, pool management, and
#   concurrent processing capabilities.
# tags: [blake3, hashing, performance, generator, optimization, concurrent]
# author: OmniNode Development Team
# license: MIT
# entrypoint: hash_generator.py
# protocols_supported: [O.N.E. v0.1]
# runtime_constraints: {sandboxed: false, privileged: false, requires_network: false, requires_gpu: false}
# dependencies: [{"name": "blake3", "version": "^0.4.1"}, {"name": "psutil", "version": "^5.9.0"}]
# environment: [python>=3.11]
# === /OmniNode:Tool_Metadata ===

"""BLAKE3 Hash Generator with sub-2ms performance optimization.

This module implements a high-performance BLAKE3 hash generator optimized
for sub-2ms hash generation with memory-efficient streaming for large files.
"""

import asyncio
import logging
import time
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import blake3
import psutil

# Configure logger for this module
logger = logging.getLogger(__name__)


class BLAKE3HashGenerator:
    """High-performance BLAKE3 hash generator with <2ms target performance."""

    def __init__(self, pool_size: int = 100, max_workers: int = 4):
        """Initialize the hash generator with optimized settings.

        Args:
            pool_size: Size of the pre-allocated hasher pool
            max_workers: Number of worker threads for parallel processing
        """
        # Pre-allocated hasher pool for zero-allocation hot path
        self.hasher_pool = asyncio.Queue(maxsize=pool_size)
        self.pool_size = pool_size

        # Optimized buffer sizes for different file sizes
        self.small_buffer_size = 8192  # 8KB for files < 1MB
        self.medium_buffer_size = 65536  # 64KB for files 1MB-10MB
        self.large_buffer_size = 1048576  # 1MB for files > 10MB

        # CPU-bound thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Performance monitoring
        self.metrics_collector = PerformanceMetricsCollector()

        # Pre-warm the hasher pool
        asyncio.create_task(self._initialize_hasher_pool())

        # Memory-mapped file cache for large files
        self.mmap_cache = weakref.WeakValueDictionary()

    async def _initialize_hasher_pool(self):
        """Pre-allocate BLAKE3 hashers to avoid allocation overhead."""
        for _ in range(self.pool_size):
            hasher = blake3.blake3()
            await self.hasher_pool.put(hasher)

    def _select_buffer_size(self, file_size: int) -> int:
        """Select optimal buffer size based on file size.

        Args:
            file_size: Size of the file in bytes

        Returns:
            Optimal buffer size for the file
        """
        if file_size < 1024 * 1024:  # < 1MB
            return self.small_buffer_size
        elif file_size < 10 * 1024 * 1024:  # < 10MB
            return self.medium_buffer_size
        else:
            return self.large_buffer_size

    async def generate_hash(
        self, file_data: bytes, file_path: Optional[str] = None
    ) -> dict[str, Any]:
        """Generate BLAKE3 hash with <2ms performance target.

        Args:
            file_data: File data to hash
            file_path: Optional file path for caching

        Returns:
            Dict containing hash, execution time, and performance metrics

        Raises:
            TypeError: If file_data is not bytes
        """
        # Validate input type
        if not isinstance(file_data, bytes):
            raise TypeError(
                f"file_data must be bytes, got {type(file_data).__name__}. "
                f"Use .encode() to convert strings to bytes."
            )

        start_time = time.perf_counter()
        start_cpu = psutil.Process().cpu_percent()

        file_size = len(file_data)

        try:
            # Performance path selection based on file size
            if file_size <= 1024:  # Very small files (≤1KB) - direct hash
                hash_result = await self._direct_hash_small_file(file_data)
            elif (
                file_size <= 1024 * 1024
            ):  # Small to medium files (≤1MB) - pooled hasher
                hash_result = await self._hash_with_pooled_hasher(file_data)
            else:  # Large files (>1MB) - streaming with thread pool
                hash_result = await self._stream_hash_large_file(file_data, file_path)

        except MemoryError as e:
            await self.metrics_collector.record_error(
                f"Memory exhausted during hash generation: {e}", file_size
            )
            raise MemoryError(
                f"Insufficient memory to hash file of size {file_size} bytes: {e}"
            )
        except OSError as e:
            await self.metrics_collector.record_error(
                f"I/O error during hash generation: {e}", file_size
            )
            raise OSError(f"File system error during hash generation: {e}")
        except TimeoutError as e:
            await self.metrics_collector.record_error(
                f"Timeout during hash generation: {e}", file_size
            )
            raise TimeoutError(
                f"Hash generation timed out for file size {file_size} bytes"
            )
        except ValueError as e:
            await self.metrics_collector.record_error(
                f"Invalid value during hash generation: {e}", file_size
            )
            raise ValueError(f"Invalid input for hash generation: {e}")
        except Exception as e:
            await self.metrics_collector.record_error(
                f"Unexpected error during hash generation: {e}", file_size
            )
            raise RuntimeError(
                f"Unexpected hash generation error for file size {file_size} bytes: {e}"
            ) from e

        # Calculate performance metrics
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        end_cpu = psutil.Process().cpu_percent()
        cpu_usage = end_cpu - start_cpu if end_cpu > start_cpu else 0

        # Record performance metrics
        await self.metrics_collector.record_hash_operation(
            execution_time_ms, file_size, cpu_usage
        )

        # Validate performance target
        if execution_time_ms > 2.0:
            await self.metrics_collector.record_performance_violation(
                execution_time_ms, file_size, "hash_generation_slow"
            )

        return {
            "hash": hash_result,
            "execution_time_ms": execution_time_ms,
            "file_size_bytes": file_size,
            "cpu_usage_percent": cpu_usage,
            "performance_grade": (
                "A"
                if execution_time_ms < 1.0
                else "B" if execution_time_ms < 2.0 else "C"
            ),
        }

    async def _direct_hash_small_file(self, file_data: bytes) -> str:
        """Optimized path for very small files - direct hash without pooling.

        Args:
            file_data: Small file data to hash

        Returns:
            Hex string of the hash
        """
        hasher = blake3.blake3()
        hasher.update(file_data)
        return hasher.hexdigest()

    async def _hash_with_pooled_hasher(self, file_data: bytes) -> str:
        """Use pre-allocated hasher from pool for medium files.

        Args:
            file_data: File data to hash

        Returns:
            Hex string of the hash
        """
        # Get hasher from pool with timeout
        try:
            hasher = await asyncio.wait_for(self.hasher_pool.get(), timeout=0.1)
        except TimeoutError:
            # Pool exhausted, create temporary hasher
            hasher = blake3.blake3()
            pool_return = False
        else:
            pool_return = True

        try:
            # Reset hasher state
            hasher.reset()
            hasher.update(file_data)
            hash_result = hasher.hexdigest()
        finally:
            # Return hasher to pool if it came from pool
            if pool_return:
                await self.hasher_pool.put(hasher)

        return hash_result

    async def _stream_hash_large_file(
        self, file_data: bytes, file_path: Optional[str] = None
    ) -> str:
        """Streaming hash for large files with thread pool processing.

        Args:
            file_data: Large file data to hash
            file_path: Optional file path for caching

        Returns:
            Hex string of the hash
        """
        file_size = len(file_data)
        buffer_size = self._select_buffer_size(file_size)

        # Use thread pool for CPU-intensive hashing
        loop = asyncio.get_event_loop()

        def _hash_in_thread():
            hasher = blake3.blake3()

            # Process in chunks for better memory usage
            for i in range(0, file_size, buffer_size):
                chunk = file_data[i : i + buffer_size]
                hasher.update(chunk)

            return hasher.hexdigest()

        # Execute in thread pool to avoid blocking the event loop
        hash_result = await loop.run_in_executor(self.thread_pool, _hash_in_thread)
        return hash_result

    async def batch_generate_hashes(
        self, files_data: list[bytes]
    ) -> list[dict[str, Any]]:
        """Generate hashes for multiple files concurrently.

        Optimized for high-throughput scenarios.

        Args:
            files_data: List of file data to hash

        Returns:
            List of hash results with performance metrics
        """
        start_time = time.perf_counter()

        # Process files concurrently with semaphore to control resource usage
        semaphore = asyncio.Semaphore(10)  # Limit concurrent operations

        async def _hash_with_semaphore(file_data: bytes) -> dict[str, Any]:
            async with semaphore:
                return await self.generate_hash(file_data)

        # Execute all hash operations concurrently
        results = await asyncio.gather(
            *[_hash_with_semaphore(file_data) for file_data in files_data],
            return_exceptions=True,
        )

        # Calculate batch metrics
        total_time_ms = (time.perf_counter() - start_time) * 1000
        successful_results = [r for r in results if not isinstance(r, Exception)]

        await self.metrics_collector.record_batch_operation(
            batch_size=len(files_data),
            successful_count=len(successful_results),
            total_time_ms=total_time_ms,
        )

        return successful_results

    async def cleanup(self):
        """Clean up resources with robust error handling and timeout protection.

        This method ensures graceful shutdown of the thread pool and proper
        cleanup of the hasher pool to prevent resource leaks.
        """
        logger.info("Starting BLAKE3HashGenerator cleanup process")
        cleanup_start_time = time.perf_counter()

        # Track cleanup statistics
        cleanup_stats = {
            "thread_pool_shutdown": False,
            "hasher_pool_cleared": False,
            "errors_encountered": [],
            "hashers_cleaned": 0,
        }

        try:
            # 1. Shutdown thread pool with timeout
            logger.debug("Shutting down thread pool executor")
            try:
                # Shutdown thread pool (timeout not supported in all Python versions)
                self.thread_pool.shutdown(wait=True)
                cleanup_stats["thread_pool_shutdown"] = True
                logger.debug("Thread pool shutdown completed successfully")
            except (OSError, RuntimeError, AttributeError) as e:
                error_msg = f"Error during thread pool shutdown: {e}"
                logger.error(error_msg)
                cleanup_stats["errors_encountered"].append(error_msg)

                # Force shutdown if timeout failed
                try:
                    logger.warning("Attempting force shutdown of thread pool")
                    self.thread_pool.shutdown(wait=False)
                except (OSError, RuntimeError, AttributeError) as force_error:
                    logger.error(f"Force shutdown also failed: {force_error}")
                    cleanup_stats["errors_encountered"].append(
                        f"Force shutdown failed: {force_error}"
                    )
                except Exception as force_error:
                    logger.error(
                        f"Unexpected error during force shutdown: {force_error}"
                    )
                    cleanup_stats["errors_encountered"].append(
                        f"Unexpected force shutdown error: {force_error}"
                    )

            # 2. Clear hasher pool to prevent memory leaks
            logger.debug("Clearing hasher pool")
            try:
                cleared_count = 0
                # Clear all hashers from the pool with timeout protection
                while not self.hasher_pool.empty():
                    try:
                        # Use asyncio.wait_for to prevent hanging on empty queue
                        await asyncio.wait_for(self.hasher_pool.get(), timeout=0.1)
                        cleared_count += 1
                    except TimeoutError:
                        # Queue is likely empty, break the loop
                        break
                    except (
                        asyncio.QueueEmpty,
                        RuntimeError,
                        AttributeError,
                    ) as pool_error:
                        logger.warning(f"Error clearing hasher from pool: {pool_error}")
                        cleanup_stats["errors_encountered"].append(
                            f"Pool clear error: {pool_error}"
                        )
                        break
                    except Exception as pool_error:
                        logger.warning(
                            f"Unexpected error clearing hasher from pool: {pool_error}"
                        )
                        cleanup_stats["errors_encountered"].append(
                            f"Unexpected pool clear error: {pool_error}"
                        )
                        break

                cleanup_stats["hashers_cleaned"] = cleared_count
                cleanup_stats["hasher_pool_cleared"] = True
                logger.debug(f"Cleared {cleared_count} hashers from pool")

            except (RuntimeError, AttributeError, MemoryError) as e:
                error_msg = f"Error during hasher pool cleanup: {e}"
                logger.error(error_msg)
                cleanup_stats["errors_encountered"].append(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error during hasher pool cleanup: {e}"
                logger.error(error_msg)
                cleanup_stats["errors_encountered"].append(error_msg)

            # 3. Clear memory-mapped cache
            try:
                if hasattr(self, "mmap_cache"):
                    cache_size = len(self.mmap_cache)
                    self.mmap_cache.clear()
                    logger.debug(
                        f"Cleared memory-mapped cache with {cache_size} entries"
                    )
            except Exception as e:
                error_msg = f"Error clearing mmap cache: {e}"
                logger.warning(error_msg)
                cleanup_stats["errors_encountered"].append(error_msg)

        except Exception as e:
            # Catch-all for any unexpected errors during cleanup
            error_msg = f"Unexpected error during cleanup: {e}"
            logger.error(error_msg)
            cleanup_stats["errors_encountered"].append(error_msg)

        finally:
            # 4. Log cleanup completion with statistics
            cleanup_duration = (time.perf_counter() - cleanup_start_time) * 1000

            if cleanup_stats["errors_encountered"]:
                logger.warning(
                    f"Cleanup completed with {len(cleanup_stats['errors_encountered'])} errors "
                    f"in {cleanup_duration:.2f}ms. "
                    f"Thread pool shutdown: {cleanup_stats['thread_pool_shutdown']}, "
                    f"Hasher pool cleared: {cleanup_stats['hasher_pool_cleared']}, "
                    f"Hashers cleaned: {cleanup_stats['hashers_cleaned']}"
                )
            else:
                logger.info(
                    f"Cleanup completed successfully in {cleanup_duration:.2f}ms. "
                    f"Hashers cleaned: {cleanup_stats['hashers_cleaned']}"
                )

            # Record cleanup metrics if metrics collector is available
            if hasattr(self, "metrics_collector") and self.metrics_collector:
                try:
                    # Create a cleanup record in the metrics
                    await self.metrics_collector.record_cleanup_operation(
                        cleanup_duration, cleanup_stats
                    )
                except Exception as metrics_error:
                    logger.debug(f"Could not record cleanup metrics: {metrics_error}")

            logger.debug("BLAKE3HashGenerator cleanup process completed")


class PerformanceMetricsCollector:
    """Collects and tracks performance metrics for BLAKE3 operations."""

    def __init__(self):
        """Initialize the metrics collector."""
        # In-memory metrics storage with rotation
        self.recent_operations = deque(maxlen=1000)
        self.performance_violations = deque(maxlen=100)
        self.error_log = deque(maxlen=100)

        # Performance statistics
        self.total_operations = 0
        self.total_execution_time = 0.0
        self.violation_count = 0

        # Lock for async safety
        self._lock = asyncio.Lock()

    async def record_hash_operation(
        self, execution_time_ms: float, file_size: int, cpu_usage: float
    ):
        """Record a hash operation with performance metrics.

        Args:
            execution_time_ms: Execution time in milliseconds
            file_size: File size in bytes
            cpu_usage: CPU usage percentage
        """
        async with self._lock:
            self.recent_operations.append(
                {
                    "timestamp": time.time(),
                    "execution_time_ms": execution_time_ms,
                    "file_size_bytes": file_size,
                    "cpu_usage_percent": cpu_usage,
                    "throughput_mbps": (
                        (file_size / 1024 / 1024) / (execution_time_ms / 1000)
                        if execution_time_ms > 0
                        else 0
                    ),
                }
            )

            self.total_operations += 1
            self.total_execution_time += execution_time_ms

    async def record_performance_violation(
        self, execution_time_ms: float, file_size: int, violation_type: str
    ):
        """Record a performance violation (>2ms execution time).

        Args:
            execution_time_ms: Execution time that violated threshold
            file_size: File size in bytes
            violation_type: Type of violation
        """
        async with self._lock:
            self.performance_violations.append(
                {
                    "timestamp": time.time(),
                    "execution_time_ms": execution_time_ms,
                    "file_size_bytes": file_size,
                    "violation_type": violation_type,
                }
            )
            self.violation_count += 1

    async def record_error(self, error_msg: str, file_size: int):
        """Record an error during hash operation.

        Args:
            error_msg: Error message
            file_size: File size in bytes
        """
        async with self._lock:
            self.error_log.append(
                {
                    "timestamp": time.time(),
                    "error": error_msg,
                    "file_size_bytes": file_size,
                }
            )

    async def record_batch_operation(
        self, batch_size: int, successful_count: int, total_time_ms: float
    ):
        """Record batch operation metrics.

        Args:
            batch_size: Number of files in batch
            successful_count: Number of successful operations
            total_time_ms: Total time for batch in milliseconds
        """
        async with self._lock:
            self.recent_operations.append(
                {
                    "timestamp": time.time(),
                    "operation_type": "batch",
                    "batch_size": batch_size,
                    "successful_count": successful_count,
                    "total_time_ms": total_time_ms,
                    "average_time_per_file": (
                        total_time_ms / batch_size if batch_size > 0 else 0
                    ),
                }
            )

    async def cleanup_old_metrics(self, max_age_seconds: int = 3600):
        """Remove metrics older than max_age to prevent memory leaks.

        Args:
            max_age_seconds: Maximum age of metrics to keep in seconds (default: 1 hour)
        """
        current_time = time.time()
        async with self._lock:
            # Filter recent operations to keep only non-expired entries
            self.recent_operations = deque(
                [
                    op
                    for op in self.recent_operations
                    if current_time - op.get("timestamp", 0) < max_age_seconds
                ],
                maxlen=1000,
            )

            # Filter performance violations to keep only non-expired entries
            self.performance_violations = deque(
                [
                    op
                    for op in self.performance_violations
                    if current_time - op.get("timestamp", 0) < max_age_seconds
                ],
                maxlen=100,
            )

            # Filter error log to keep only non-expired entries
            self.error_log = deque(
                [
                    op
                    for op in self.error_log
                    if current_time - op.get("timestamp", 0) < max_age_seconds
                ],
                maxlen=100,
            )

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get current performance statistics.

        Returns:
            Dictionary of performance statistics
        """
        async with self._lock:
            if self.total_operations == 0:
                return {"status": "no_operations"}

            recent_ops = list(self.recent_operations)
            if not recent_ops:
                return {"status": "no_recent_operations"}

            # Calculate statistics
            execution_times = [
                op["execution_time_ms"]
                for op in recent_ops
                if "execution_time_ms" in op
            ]

            return {
                "total_operations": self.total_operations,
                "average_execution_time_ms": (
                    sum(execution_times) / len(execution_times)
                    if execution_times
                    else 0
                ),
                "min_execution_time_ms": min(execution_times) if execution_times else 0,
                "max_execution_time_ms": max(execution_times) if execution_times else 0,
                "performance_violations": self.violation_count,
                "violation_rate": self.violation_count / self.total_operations,
                "operations_under_2ms": sum(1 for t in execution_times if t < 2.0),
                "performance_grade": (
                    "A"
                    if self.violation_count / self.total_operations < 0.01
                    else (
                        "B"
                        if self.violation_count / self.total_operations < 0.05
                        else "C"
                    )
                ),
            }

    async def record_cleanup_operation(
        self, cleanup_duration_ms: float, cleanup_stats: dict[str, Any]
    ):
        """Record cleanup operation metrics.

        Args:
            cleanup_duration_ms: Cleanup duration in milliseconds
            cleanup_stats: Cleanup statistics dictionary
        """
        async with self._lock:
            self.recent_operations.append(
                {
                    "timestamp": time.time(),
                    "operation_type": "cleanup",
                    "cleanup_duration_ms": cleanup_duration_ms,
                    "thread_pool_shutdown": cleanup_stats.get(
                        "thread_pool_shutdown", False
                    ),
                    "hasher_pool_cleared": cleanup_stats.get(
                        "hasher_pool_cleared", False
                    ),
                    "hashers_cleaned": cleanup_stats.get("hashers_cleaned", 0),
                    "errors_count": len(cleanup_stats.get("errors_encountered", [])),
                    "success": len(cleanup_stats.get("errors_encountered", [])) == 0,
                }
            )
