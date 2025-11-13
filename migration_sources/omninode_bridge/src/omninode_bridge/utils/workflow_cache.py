"""Memory-efficient workflow definition caching and storage system."""

import asyncio
import gzip
import hashlib
import json
import logging
import os

# Removed unsafe pickle import - now using JSON serialization for security
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkflowCacheEntry:
    """Workflow cache entry with metadata."""

    workflow_id: str
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    is_compressed: bool = False
    storage_location: str | None = None  # File path for disk storage
    memory_reference: weakref.ref | None = field(default=None, repr=False)


class MemoryEfficientWorkflowCache:
    """Memory-efficient caching system for workflow definitions with tiered storage."""

    def __init__(
        self,
        max_memory_size_mb: int = None,
        max_disk_size_mb: int = None,
        compression_threshold_kb: int = None,
        disk_storage_threshold_kb: int = None,
        cache_dir: str | None = None,
        cleanup_interval_seconds: int = 300,
    ):
        """Initialize memory-efficient workflow cache.

        Args:
            max_memory_size_mb: Maximum memory cache size in MB
            max_disk_size_mb: Maximum disk cache size in MB
            compression_threshold_kb: Compress workflows larger than this (KB)
            disk_storage_threshold_kb: Store to disk workflows larger than this (KB)
            cache_dir: Directory for disk cache (default: temp directory)
            cleanup_interval_seconds: Interval for cache cleanup (default: 5 minutes)
        """
        # Environment-based configuration
        environment = os.getenv("ENVIRONMENT", "development").lower()

        if environment == "production":
            default_memory_mb = 128
            default_disk_mb = 1024
            default_compression_kb = 100
            default_disk_threshold_kb = 500
        elif environment == "staging":
            default_memory_mb = 64
            default_disk_mb = 512
            default_compression_kb = 50
            default_disk_threshold_kb = 250
        else:  # development
            default_memory_mb = 32
            default_disk_mb = 256
            default_compression_kb = 25
            default_disk_threshold_kb = 100

        self.max_memory_size = (
            (
                max_memory_size_mb
                or int(os.getenv("WORKFLOW_CACHE_MEMORY_MB", default_memory_mb))
            )
            * 1024
            * 1024
        )
        self.max_disk_size = (
            (
                max_disk_size_mb
                or int(os.getenv("WORKFLOW_CACHE_DISK_MB", default_disk_mb))
            )
            * 1024
            * 1024
        )
        self.compression_threshold = (
            compression_threshold_kb
            or int(
                os.getenv("WORKFLOW_COMPRESSION_THRESHOLD_KB", default_compression_kb),
            )
        ) * 1024
        self.disk_threshold = (
            disk_storage_threshold_kb
            or int(os.getenv("WORKFLOW_DISK_THRESHOLD_KB", default_disk_threshold_kb))
        ) * 1024

        # Cache storage
        self.memory_cache: OrderedDict[str, Any] = OrderedDict()
        self.cache_metadata: dict[str, WorkflowCacheEntry] = {}
        self.current_memory_size = 0
        self.current_disk_size = 0

        # Disk storage setup
        self.cache_dir = (
            Path(cache_dir or os.getenv("WORKFLOW_CACHE_DIR", tempfile.gettempdir()))
            / "omninode_workflows"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        # Background cleanup
        self.cleanup_interval = cleanup_interval_seconds
        self.cleanup_task: asyncio.Task | None = None
        self._stop_cleanup = False

        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.disk_reads = 0
        self.disk_writes = 0
        self.compressions = 0
        self.decompressions = 0

        logger.info(
            f"Workflow cache initialized: memory={self.max_memory_size // (1024*1024)}MB, "
            f"disk={self.max_disk_size // (1024*1024)}MB, "
            f"compression_threshold={self.compression_threshold // 1024}KB",
        )

    async def start_background_cleanup(self) -> None:
        """Start background cleanup task."""
        if self.cleanup_task and not self.cleanup_task.done():
            return

        self._stop_cleanup = False
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Background workflow cache cleanup started")

    async def stop_background_cleanup(self) -> None:
        """Stop background cleanup task."""
        self._stop_cleanup = True
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Background workflow cache cleanup stopped")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        try:
            while not self._stop_cleanup:
                await self._cleanup_expired_entries()
                await asyncio.sleep(self.cleanup_interval)
        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled")
        except (RuntimeError, ValueError, OSError) as e:
            # Expected errors during cleanup
            logger.warning(f"Error in cleanup loop: {e}")
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error in cleanup loop: {e}", exc_info=True)

    def _get_workflow_hash(self, workflow_data: dict[str, Any]) -> str:
        """Generate a hash for workflow data to use as cache key."""
        # Create a deterministic string representation
        workflow_str = json.dumps(workflow_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(workflow_str.encode()).hexdigest()[:16]

    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            if isinstance(data, str | bytes):
                return len(data.encode() if isinstance(data, str) else data)
            # Use JSON to estimate size for workflow data (safer than pickle)
            return len(json.dumps(data, separators=(",", ":")).encode())
        except (TypeError, ValueError, AttributeError) as e:
            # Fallback estimation for non-serializable data
            logger.debug(f"Size calculation fallback for non-serializable data: {e}")
            return len(str(data).encode())

    def _compress_data(self, data: Any) -> bytes:
        """Compress workflow data."""
        try:
            serialized = json.dumps(data, separators=(",", ":")).encode()
            compressed = gzip.compress(serialized, compresslevel=6)
            self.compressions += 1
            logger.debug(
                f"Compressed data: {len(serialized)} -> {len(compressed)} bytes "
                f"({len(compressed)/len(serialized)*100:.1f}%)",
            )
            return compressed
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise

    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress workflow data."""
        try:
            decompressed = gzip.decompress(compressed_data)
            data = json.loads(decompressed.decode())
            self.decompressions += 1
            return data
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise

    async def _write_to_disk(self, workflow_id: str, data: Any) -> str:
        """Write workflow data to disk storage."""
        file_path = self.cache_dir / f"workflow_{workflow_id}.cache"

        try:
            # Compress if over threshold
            if self._calculate_size(data) > self.compression_threshold:
                compressed_data = self._compress_data(data)
                is_compressed = True
                write_data = compressed_data
            else:
                write_data = json.dumps(data, separators=(",", ":")).encode()
                is_compressed = False

            # Write atomically
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                f.write(write_data)
            temp_path.rename(file_path)

            file_size = file_path.stat().st_size
            self.current_disk_size += file_size
            self.disk_writes += 1

            logger.debug(
                f"Wrote workflow {workflow_id} to disk: {file_size} bytes, compressed={is_compressed}",
            )
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to write workflow {workflow_id} to disk: {e}")
            if file_path.exists():
                file_path.unlink()
            raise

    async def _read_from_disk(self, file_path: str, is_compressed: bool) -> Any:
        """Read workflow data from disk storage."""
        try:
            with open(file_path, "rb") as f:
                data = f.read()

            if is_compressed:
                result = self._decompress_data(data)
            else:
                result = json.loads(data.decode())

            self.disk_reads += 1
            logger.debug(f"Read workflow from disk: {file_path}")
            return result

        except Exception as e:
            logger.error(f"Failed to read workflow from disk {file_path}: {e}")
            # Clean up corrupted file
            try:
                Path(file_path).unlink()
            except (OSError, PermissionError) as cleanup_error:
                logger.warning(
                    f"Failed to cleanup corrupted file {file_path}: {cleanup_error}"
                )
            raise

    async def put(self, workflow_id: str, workflow_data: dict[str, Any]) -> None:
        """Store workflow data in cache with intelligent tiering."""
        with self._lock:
            current_time = time.time()
            data_size = self._calculate_size(workflow_data)

            # Create cache entry metadata
            entry = WorkflowCacheEntry(
                workflow_id=workflow_id,
                size_bytes=data_size,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
            )

            # Determine storage strategy based on size
            if data_size > self.disk_threshold:
                # Large workflows: store on disk
                try:
                    file_path = await self._write_to_disk(workflow_id, workflow_data)
                    entry.storage_location = file_path
                    entry.is_compressed = data_size > self.compression_threshold
                    logger.info(
                        f"Stored large workflow {workflow_id} to disk: {data_size} bytes",
                    )
                except Exception as e:
                    logger.error(f"Failed to store workflow {workflow_id} to disk: {e}")
                    return
            else:
                # Smaller workflows: try memory first
                await self._ensure_memory_capacity(data_size)

                if data_size > self.compression_threshold:
                    # Compress before storing in memory
                    compressed_data = self._compress_data(workflow_data)
                    self.memory_cache[workflow_id] = compressed_data
                    entry.is_compressed = True
                    self.current_memory_size += len(compressed_data)
                else:
                    # Store uncompressed in memory
                    self.memory_cache[workflow_id] = workflow_data
                    entry.memory_reference = weakref.ref(workflow_data)
                    self.current_memory_size += data_size

                logger.debug(
                    f"Stored workflow {workflow_id} in memory: {data_size} bytes, "
                    f"compressed={entry.is_compressed}",
                )

            # Update metadata
            self.cache_metadata[workflow_id] = entry

    async def get(self, workflow_id: str) -> dict[str, Any] | None:
        """Retrieve workflow data from cache with intelligent loading."""
        with self._lock:
            if workflow_id not in self.cache_metadata:
                self.cache_misses += 1
                return None

            entry = self.cache_metadata[workflow_id]
            current_time = time.time()

            # Update access statistics
            entry.last_accessed = current_time
            entry.access_count += 1
            self.cache_hits += 1

            try:
                # Try memory first
                if workflow_id in self.memory_cache:
                    data = self.memory_cache[workflow_id]

                    # Move to end for LRU
                    self.memory_cache.move_to_end(workflow_id)

                    if entry.is_compressed:
                        return self._decompress_data(data)
                    return data

                # Load from disk if stored there
                elif entry.storage_location:
                    data = await self._read_from_disk(
                        entry.storage_location,
                        entry.is_compressed,
                    )

                    # Consider promoting to memory if frequently accessed
                    if (
                        entry.access_count > 3
                        and entry.size_bytes < self.disk_threshold
                    ):
                        await self._promote_to_memory(workflow_id, data, entry)

                    return data

                # Check weak reference
                elif entry.memory_reference and entry.memory_reference():
                    return entry.memory_reference()

                else:
                    # Entry exists but data is missing - clean up
                    logger.warning(
                        f"Workflow {workflow_id} metadata exists but data is missing",
                    )
                    del self.cache_metadata[workflow_id]
                    return None

            except Exception as e:
                logger.error(f"Failed to retrieve workflow {workflow_id}: {e}")
                return None

    async def _promote_to_memory(
        self,
        workflow_id: str,
        data: Any,
        entry: WorkflowCacheEntry,
    ) -> None:
        """Promote frequently accessed disk-stored workflow to memory."""
        try:
            data_size = self._calculate_size(data)
            await self._ensure_memory_capacity(data_size)

            # Store in memory
            if data_size > self.compression_threshold:
                compressed_data = self._compress_data(data)
                self.memory_cache[workflow_id] = compressed_data
                self.current_memory_size += len(compressed_data)
            else:
                self.memory_cache[workflow_id] = data
                entry.memory_reference = weakref.ref(data)
                self.current_memory_size += data_size

            logger.info(f"Promoted workflow {workflow_id} to memory cache")

        except Exception as e:
            logger.error(f"Failed to promote workflow {workflow_id} to memory: {e}")

    async def _ensure_memory_capacity(self, required_size: int) -> None:
        """Ensure sufficient memory capacity by evicting old entries."""
        while (
            self.current_memory_size + required_size > self.max_memory_size
            and self.memory_cache
        ):
            # Evict least recently used item
            lru_id, lru_data = self.memory_cache.popitem(last=False)
            lru_entry = self.cache_metadata.get(lru_id)

            if lru_entry:
                data_size = lru_entry.size_bytes
                self.current_memory_size -= data_size

                # If it's a large item that should be on disk, move it there
                if data_size > self.disk_threshold and not lru_entry.storage_location:
                    try:
                        if lru_entry.is_compressed:
                            original_data = self._decompress_data(lru_data)
                        else:
                            original_data = lru_data

                        file_path = await self._write_to_disk(lru_id, original_data)
                        lru_entry.storage_location = file_path
                        logger.debug(f"Evicted workflow {lru_id} from memory to disk")
                    except Exception as e:
                        logger.error(
                            f"Failed to move evicted workflow {lru_id} to disk: {e}",
                        )
                        # Remove from metadata if can't save to disk
                        del self.cache_metadata[lru_id]
                else:
                    logger.debug(f"Evicted workflow {lru_id} from memory")

    async def _cleanup_expired_entries(self) -> None:
        """Clean up expired and least accessed entries."""
        current_time = time.time()
        max_age = int(os.getenv("WORKFLOW_CACHE_MAX_AGE_HOURS", "24")) * 3600

        expired_ids = []

        with self._lock:
            # Find expired entries
            for workflow_id, entry in self.cache_metadata.items():
                if current_time - entry.created_at > max_age:
                    expired_ids.append(workflow_id)

            # Clean up expired entries
            for workflow_id in expired_ids:
                await self._remove_entry(workflow_id)

            # Clean up disk space if over limit
            if self.current_disk_size > self.max_disk_size:
                await self._cleanup_disk_space()

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired workflow cache entries")

    async def _remove_entry(self, workflow_id: str) -> None:
        """Remove entry from cache and disk storage."""
        try:
            entry = self.cache_metadata.get(workflow_id)
            if not entry:
                return

            # Remove from memory
            if workflow_id in self.memory_cache:
                del self.memory_cache[workflow_id]
                self.current_memory_size -= entry.size_bytes

            # Remove from disk
            if entry.storage_location:
                try:
                    file_path = Path(entry.storage_location)
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        self.current_disk_size -= file_size
                except Exception as e:
                    logger.error(
                        f"Failed to remove disk file for workflow {workflow_id}: {e}",
                    )

            # Remove metadata
            del self.cache_metadata[workflow_id]

        except Exception as e:
            logger.error(f"Failed to remove workflow cache entry {workflow_id}: {e}")

    async def _cleanup_disk_space(self) -> None:
        """Clean up disk space by removing least accessed entries."""
        # Sort by access frequency and age
        sorted_entries = sorted(
            [
                (id, entry)
                for id, entry in self.cache_metadata.items()
                if entry.storage_location
            ],
            key=lambda x: (x[1].access_count, x[1].last_accessed),
        )

        # Remove least accessed entries until under limit
        for workflow_id, entry in sorted_entries:
            if self.current_disk_size <= self.max_disk_size * 0.9:  # 90% threshold
                break
            await self._remove_entry(workflow_id)

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            memory_entries = len(self.memory_cache)
            disk_entries = len(
                [e for e in self.cache_metadata.values() if e.storage_location],
            )
            total_entries = len(self.cache_metadata)

            hit_rate = (
                self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                if (self.cache_hits + self.cache_misses) > 0
                else 0
            )

            return {
                "performance_metrics": {
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "hit_rate_percent": hit_rate,
                    "disk_reads": self.disk_reads,
                    "disk_writes": self.disk_writes,
                    "compressions": self.compressions,
                    "decompressions": self.decompressions,
                },
                "storage_metrics": {
                    "memory_entries": memory_entries,
                    "disk_entries": disk_entries,
                    "total_entries": total_entries,
                    "memory_usage_bytes": self.current_memory_size,
                    "disk_usage_bytes": self.current_disk_size,
                    "memory_usage_percent": (
                        self.current_memory_size / self.max_memory_size
                    )
                    * 100,
                    "disk_usage_percent": (self.current_disk_size / self.max_disk_size)
                    * 100,
                },
                "configuration": {
                    "max_memory_size_mb": self.max_memory_size // (1024 * 1024),
                    "max_disk_size_mb": self.max_disk_size // (1024 * 1024),
                    "compression_threshold_kb": self.compression_threshold // 1024,
                    "disk_threshold_kb": self.disk_threshold // 1024,
                    "cache_directory": str(self.cache_dir),
                },
            }

    async def clear_cache(self) -> None:
        """Clear all cached workflows."""
        with self._lock:
            # Clear memory cache
            self.memory_cache.clear()
            self.current_memory_size = 0

            # Clear disk cache
            for entry in self.cache_metadata.values():
                if entry.storage_location:
                    try:
                        Path(entry.storage_location).unlink()
                    except (OSError, FileNotFoundError, PermissionError):
                        # File deletion failed, acceptable during cleanup
                        pass

            self.current_disk_size = 0
            self.cache_metadata.clear()

            # Reset metrics
            self.cache_hits = 0
            self.cache_misses = 0
            self.disk_reads = 0
            self.disk_writes = 0
            self.compressions = 0
            self.decompressions = 0

        logger.info("Workflow cache cleared")


# Global workflow cache instance
workflow_cache = MemoryEfficientWorkflowCache()


async def init_workflow_cache() -> None:
    """Initialize the global workflow cache."""
    await workflow_cache.start_background_cleanup()
    logger.info("Global workflow cache initialized")


async def shutdown_workflow_cache() -> None:
    """Shutdown the global workflow cache."""
    await workflow_cache.stop_background_cleanup()
    logger.info("Global workflow cache shut down")
