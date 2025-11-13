"""
Database sharding implementation for MetadataStampingService.

Provides horizontal database scaling through automatic sharding with
consistent hashing, health monitoring, and automatic failover capabilities.
"""

import asyncio
import hashlib
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

import asyncpg
import psutil
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ShardStatus(Enum):
    """Status of a database shard."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class ShardConfig:
    """Configuration for a database shard."""

    shard_id: str
    database_url: str
    weight: float = 1.0
    max_connections: int = 50
    min_connections: int = 5
    read_only: bool = False
    region: str = "us-west-2"
    availability_zone: str = "us-west-2a"
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ShardMetrics:
    """Performance and health metrics for a shard."""

    shard_id: str
    status: ShardStatus
    connection_count: int
    active_queries: int
    avg_response_time: float
    error_rate: float
    last_health_check: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    replication_lag: Optional[float] = None


class ConsistentHashRing:
    """Consistent hash ring for shard selection."""

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: dict[int, str] = {}
        self.nodes: set[str] = set()

    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node_id: str, weight: float = 1.0) -> None:
        """Add a node to the hash ring."""
        if node_id in self.nodes:
            return

        self.nodes.add(node_id)

        # Add virtual nodes based on weight
        virtual_node_count = int(self.virtual_nodes * weight)
        for i in range(virtual_node_count):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node_id

        logger.info(
            f"Added shard {node_id} to hash ring with {virtual_node_count} virtual nodes"
        )

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the hash ring."""
        if node_id not in self.nodes:
            return

        self.nodes.remove(node_id)

        # Remove all virtual nodes for this shard
        keys_to_remove = [k for k, v in self.ring.items() if v == node_id]
        for key in keys_to_remove:
            del self.ring[key]

        logger.info(f"Removed shard {node_id} from hash ring")

    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a key."""
        if not self.ring:
            return None

        hash_value = self._hash(key)

        # Find the first node with hash >= key hash
        for ring_hash in sorted(self.ring.keys()):
            if ring_hash >= hash_value:
                return self.ring[ring_hash]

        # Wrap around to the first node
        return self.ring[min(self.ring.keys())]

    def get_nodes(self, key: str, count: int = 3) -> list[str]:
        """Get multiple nodes for a key (for replication)."""
        if not self.ring or count <= 0:
            return []

        hash_value = self._hash(key)
        sorted_hashes = sorted(self.ring.keys())

        # Find starting position
        start_idx = 0
        for i, ring_hash in enumerate(sorted_hashes):
            if ring_hash >= hash_value:
                start_idx = i
                break

        # Collect unique nodes
        nodes = []
        seen_nodes = set()

        for i in range(len(sorted_hashes)):
            idx = (start_idx + i) % len(sorted_hashes)
            node = self.ring[sorted_hashes[idx]]

            if node not in seen_nodes:
                nodes.append(node)
                seen_nodes.add(node)

                if len(nodes) >= count:
                    break

        return nodes


class ShardRouter:
    """Routes database operations to appropriate shards."""

    def __init__(self, shards: list[ShardConfig]):
        self.shards = {shard.shard_id: shard for shard in shards}
        self.hash_ring = ConsistentHashRing()
        self.read_replicas: dict[str, list[str]] = defaultdict(list)

        # Initialize hash ring
        for shard in shards:
            if not shard.read_only:
                self.hash_ring.add_node(shard.shard_id, shard.weight)
            else:
                # Track read replicas separately
                primary_id = shard.tags.get("primary_shard", shard.shard_id)
                self.read_replicas[primary_id].append(shard.shard_id)

        logger.info(f"Initialized shard router with {len(self.shards)} shards")

    def route_write(self, key: str) -> str:
        """Route a write operation to the appropriate shard."""
        shard_id = self.hash_ring.get_node(key)
        if not shard_id:
            raise RuntimeError("No healthy shards available for write operations")

        return shard_id

    def route_read(self, key: str, prefer_replica: bool = True) -> str:
        """Route a read operation, optionally preferring read replicas."""
        primary_shard = self.hash_ring.get_node(key)
        if not primary_shard:
            raise RuntimeError("No healthy shards available for read operations")

        # Check for read replicas if preferred
        if prefer_replica and primary_shard in self.read_replicas:
            replicas = self.read_replicas[primary_shard]
            if replicas:
                # Simple round-robin selection (could be enhanced with health checks)
                replica_index = hash(key) % len(replicas)
                return replicas[replica_index]

        return primary_shard

    def get_shards_for_key(self, key: str, replica_count: int = 3) -> list[str]:
        """Get all shards that should contain a key (for replication)."""
        return self.hash_ring.get_nodes(key, replica_count)

    def get_all_shards(self, include_replicas: bool = True) -> list[str]:
        """Get all available shard IDs."""
        if include_replicas:
            return list(self.shards.keys())
        else:
            return [
                shard_id
                for shard_id in self.shards
                if not self.shards[shard_id].read_only
            ]

    def remove_shard(self, shard_id: str) -> None:
        """Remove a shard from routing (for maintenance/failures)."""
        if shard_id in self.shards:
            self.hash_ring.remove_node(shard_id)
            logger.warning(f"Removed shard {shard_id} from routing")

    def add_shard(self, shard_config: ShardConfig) -> None:
        """Add a new shard to routing."""
        self.shards[shard_config.shard_id] = shard_config
        if not shard_config.read_only:
            self.hash_ring.add_node(shard_config.shard_id, shard_config.weight)
        logger.info(f"Added new shard {shard_config.shard_id} to routing")


class ShardHealthMonitor:
    """Monitors health of database shards."""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.shard_metrics: dict[str, ShardMetrics] = {}
        self.health_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False

    async def start_monitoring(self, shard_manager: "DatabaseShardManager") -> None:
        """Start health monitoring for all shards."""
        if self.running:
            return

        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(shard_manager))
        logger.info("Started shard health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped shard health monitoring")

    async def _monitoring_loop(self, shard_manager: "DatabaseShardManager") -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                await self._check_all_shards(shard_manager)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _check_all_shards(self, shard_manager: "DatabaseShardManager") -> None:
        """Check health of all shards."""
        tasks = []
        for shard_id in shard_manager.get_all_shard_ids():
            task = asyncio.create_task(
                self._check_shard_health(shard_manager, shard_id)
            )
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_shard_health(
        self, shard_manager: "DatabaseShardManager", shard_id: str
    ) -> None:
        """Check health of a specific shard."""
        start_time = time.time()

        try:
            # Perform health check query
            async with shard_manager.get_connection(shard_id) as connection:
                await connection.fetchval("SELECT 1")

            response_time = (time.time() - start_time) * 1000

            # Get connection pool stats
            pool = shard_manager._shard_pools.get(shard_id)
            connection_count = pool.get_size() if pool else 0

            # Get system metrics (simplified)
            process = psutil.Process()
            cpu_usage = process.cpu_percent()
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / 1024 / 1024  # MB

            # Calculate error rate from history
            error_rate = self._calculate_error_rate(shard_id)

            metrics = ShardMetrics(
                shard_id=shard_id,
                status=ShardStatus.HEALTHY,
                connection_count=connection_count,
                active_queries=0,  # Would need pool introspection
                avg_response_time=response_time,
                error_rate=error_rate,
                last_health_check=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=0.0,  # Would need disk monitoring
            )

            # Determine status based on metrics
            if response_time > 1000:  # > 1 second
                metrics.status = ShardStatus.DEGRADED
            if error_rate > 0.05:  # > 5% error rate
                metrics.status = ShardStatus.DEGRADED

            self.shard_metrics[shard_id] = metrics
            self.health_history[shard_id].append((time.time(), True))

            logger.debug(
                f"Health check passed for shard {shard_id}: {response_time:.2f}ms"
            )

        except Exception as e:
            logger.error(f"Health check failed for shard {shard_id}: {e}")

            # Record failure
            self.health_history[shard_id].append((time.time(), False))

            # Update metrics with failure status
            if shard_id in self.shard_metrics:
                self.shard_metrics[shard_id].status = ShardStatus.UNHEALTHY
                self.shard_metrics[shard_id].last_health_check = time.time()

    def _calculate_error_rate(self, shard_id: str) -> float:
        """Calculate error rate from health history."""
        history = self.health_history[shard_id]
        if not history:
            return 0.0

        recent_checks = [success for _, success in history]
        if not recent_checks:
            return 0.0

        failures = sum(1 for success in recent_checks if not success)
        return failures / len(recent_checks)

    def get_shard_metrics(self, shard_id: str) -> Optional[ShardMetrics]:
        """Get current metrics for a shard."""
        return self.shard_metrics.get(shard_id)

    def get_healthy_shards(self) -> list[str]:
        """Get list of currently healthy shards."""
        healthy_shards = []
        for shard_id, metrics in self.shard_metrics.items():
            if metrics.status in [ShardStatus.HEALTHY, ShardStatus.DEGRADED]:
                healthy_shards.append(shard_id)
        return healthy_shards

    def is_shard_healthy(self, shard_id: str) -> bool:
        """Check if a specific shard is healthy."""
        metrics = self.shard_metrics.get(shard_id)
        if not metrics:
            return False
        return metrics.status in [ShardStatus.HEALTHY, ShardStatus.DEGRADED]


class DatabaseShardManager:
    """Manages database shards with automatic routing and health monitoring."""

    def __init__(self, shard_configs: list[ShardConfig]):
        self.shard_configs = {config.shard_id: config for config in shard_configs}
        self.router = ShardRouter(shard_configs)
        self.health_monitor = ShardHealthMonitor()
        self._shard_pools: dict[str, asyncpg.Pool] = {}
        self._pool_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all shard connections and start monitoring."""
        async with self._initialization_lock:
            if self._initialized:
                return

            logger.info("Initializing database shard manager")

            # Initialize connection pools for all shards
            initialization_tasks = []
            for shard_id in self.shard_configs:
                task = asyncio.create_task(self._initialize_shard_pool(shard_id))
                initialization_tasks.append(task)

            # Wait for all pools to initialize
            results = await asyncio.gather(
                *initialization_tasks, return_exceptions=True
            )

            # Collect failed shards and remove from hash ring
            failed_shards = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    shard_id = list(self.shard_configs.keys())[i]
                    logger.error(f"Failed to initialize shard {shard_id}: {result}")
                    failed_shards.append((shard_id, result))
                    # Remove failed shard from router
                    self.router.remove_shard(shard_id)

            # Fail fast if any shards failed to initialize
            if failed_shards:
                failed_shard_ids = [shard_id for shard_id, _ in failed_shards]
                error_details = "; ".join(
                    f"{shard_id}: {error!s}" for shard_id, error in failed_shards
                )
                raise RuntimeError(
                    f"Failed to initialize {len(failed_shards)} shard(s): "
                    f"{failed_shard_ids}. Details: {error_details}"
                )

            # Start health monitoring
            await self.health_monitor.start_monitoring(self)

            self._initialized = True
            logger.info("Database shard manager initialized successfully")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _initialize_shard_pool(self, shard_id: str) -> None:
        """Initialize connection pool for a specific shard."""
        async with self._pool_locks[shard_id]:
            if shard_id in self._shard_pools:
                return

            config = self.shard_configs[shard_id]

            try:
                # Parse database URL
                parsed_url = urlparse(config.database_url)

                # Create connection pool
                pool = await asyncpg.create_pool(
                    host=parsed_url.hostname,
                    port=parsed_url.port or 5432,
                    database=parsed_url.path.lstrip("/"),
                    user=parsed_url.username,
                    password=parsed_url.password,
                    min_size=config.min_connections,
                    max_size=config.max_connections,
                    command_timeout=30,
                    server_settings={
                        "application_name": f"metadata_stamping_shard_{shard_id}",
                        "tcp_keepalives_idle": "600",
                        "tcp_keepalives_interval": "30",
                        "tcp_keepalives_count": "3",
                    },
                )

                self._shard_pools[shard_id] = pool
                logger.info(f"Initialized connection pool for shard {shard_id}")

            except Exception as e:
                logger.warning(f"Failed to initialize pool for shard {shard_id}: {e}")
                # Don't add to _shard_pools, will be removed from hash ring
                raise

    @asynccontextmanager
    async def get_connection(self, shard_id: str):
        """Get a database connection for a specific shard."""
        if not self._initialized:
            await self.initialize()

        if shard_id not in self._shard_pools:
            raise ValueError(f"Shard {shard_id} not found or not initialized")

        pool = self._shard_pools[shard_id]

        try:
            async with pool.acquire() as connection:
                yield connection
        except Exception as e:
            logger.error(f"Error getting connection for shard {shard_id}: {e}")
            raise

    async def execute_on_shard(self, shard_id: str, query: str, *args) -> Any:
        """Execute a query on a specific shard."""
        async with self.get_connection(shard_id) as connection:
            return await connection.fetch(query, *args)

    async def execute_write(self, key: str, query: str, *args) -> Any:
        """Execute a write operation using shard routing."""
        shard_id = self.router.route_write(key)
        return await self.execute_on_shard(shard_id, query, *args)

    async def execute_read(
        self, key: str, query: str, *args, prefer_replica: bool = True
    ) -> Any:
        """Execute a read operation using shard routing."""
        shard_id = self.router.route_read(key, prefer_replica)
        return await self.execute_on_shard(shard_id, query, *args)

    async def execute_on_all_shards(
        self, query: str, *args, include_replicas: bool = False
    ) -> dict[str, Any]:
        """Execute a query on all shards."""
        shard_ids = self.router.get_all_shards(include_replicas)

        tasks = []
        for shard_id in shard_ids:
            if self.health_monitor.is_shard_healthy(shard_id):
                task = asyncio.create_task(
                    self.execute_on_shard(shard_id, query, *args)
                )
                tasks.append((shard_id, task))

        results = {}
        for shard_id, task in tasks:
            try:
                result = await task
                results[shard_id] = result
            except Exception as e:
                logger.error(f"Query failed on shard {shard_id}: {e}")
                results[shard_id] = None

        return results

    def get_shard_for_key(self, key: str) -> str:
        """Get the primary shard ID for a given key."""
        return self.router.route_write(key)

    def get_all_shard_ids(self) -> list[str]:
        """Get all shard IDs."""
        return list(self.shard_configs.keys())

    def get_shard_health(self) -> dict[str, ShardMetrics]:
        """Get health metrics for all shards."""
        return dict(self.health_monitor.shard_metrics)

    async def close(self) -> None:
        """Close all shard connections and stop monitoring."""
        logger.info("Closing database shard manager")

        # Stop health monitoring
        await self.health_monitor.stop_monitoring()

        # Close all connection pools
        close_tasks = []
        for shard_id, pool in self._shard_pools.items():
            task = asyncio.create_task(pool.close())
            close_tasks.append(task)

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self._shard_pools.clear()
        logger.info("Database shard manager closed")


def load_shard_configs_from_env() -> list[ShardConfig]:
    """Load shard configurations from environment variables."""
    shard_configs = []

    # Parse SHARD_DATABASE_URLS environment variable
    shard_urls_str = os.getenv("SHARD_DATABASE_URLS", "")
    if not shard_urls_str:
        raise ValueError("SHARD_DATABASE_URLS environment variable not set")

    shard_lines = [
        line.strip() for line in shard_urls_str.strip().split("\n") if line.strip()
    ]

    for line in shard_lines:
        if "=" not in line:
            continue

        shard_id, database_url = line.split("=", 1)

        # Parse additional configuration from environment
        weight = float(os.getenv(f"SHARD_{shard_id.upper()}_WEIGHT", "1.0"))
        max_connections = int(
            os.getenv(f"SHARD_{shard_id.upper()}_MAX_CONNECTIONS", "50")
        )
        min_connections = int(
            os.getenv(f"SHARD_{shard_id.upper()}_MIN_CONNECTIONS", "5")
        )
        read_only = (
            os.getenv(f"SHARD_{shard_id.upper()}_READ_ONLY", "false").lower() == "true"
        )
        region = os.getenv(
            f"SHARD_{shard_id.upper()}_REGION", os.getenv("REGION", "us-west-2")
        )
        az = os.getenv(
            f"SHARD_{shard_id.upper()}_AZ", os.getenv("AVAILABILITY_ZONE", "us-west-2a")
        )

        config = ShardConfig(
            shard_id=shard_id,
            database_url=database_url,
            weight=weight,
            max_connections=max_connections,
            min_connections=min_connections,
            read_only=read_only,
            region=region,
            availability_zone=az,
        )

        shard_configs.append(config)
        logger.info(
            f"Loaded shard config: {shard_id} (weight={weight}, read_only={read_only})"
        )

    if not shard_configs:
        raise ValueError("No valid shard configurations found")

    return shard_configs


# Global shard manager instance
_shard_manager: Optional[DatabaseShardManager] = None


async def get_shard_manager() -> DatabaseShardManager:
    """Get the global shard manager instance."""
    global _shard_manager

    if _shard_manager is None:
        shard_configs = load_shard_configs_from_env()
        _shard_manager = DatabaseShardManager(shard_configs)
        await _shard_manager.initialize()

    return _shard_manager


async def close_shard_manager() -> None:
    """Close the global shard manager instance."""
    global _shard_manager

    if _shard_manager is not None:
        await _shard_manager.close()
        _shard_manager = None
